# Project Rundown

A from-scratch, end-to-end explanation of this codebase: a genetic algorithm
(GA) for robot-cell layout optimisation, whose local-search rate, mutation
rate and crossover rate can optionally be steered at runtime by a tabular
Q-learning (reinforcement learning, RL) controller. This document walks
through the pipeline in the order data actually flows: instance generation →
the GA itself → the RL controller → how training and testing are conducted →
where the outputs land → how to run any of it.

For a shorter, reference-style overview see `README.md`. This file is the
narrative version.

---

## 1. Instance generation — `core/problems.py`

Every benchmark instance is a robot-cell layout problem: a set of machines
that must be placed (position + 90°-snapped rotation) inside a square
workspace, without collisions, such that the robot can reach every machine's
access point and travel between them in a fixed visiting order.

Instances are generated deterministically from a per-instance integer seed —
`random.Random(seed)` drives every random draw in a fixed order, so any
instance is fully reproducible from its seed alone.

**Splits** (disjoint seed ranges, so no instance is ever reused across
purposes):

| split | count | seeds | geometry | used for |
|---|---|---|---|---|
| `train` | 15 | 2000–2014 | 8 machines, workspace half-extent 15 | RL training, state-bin calibration |
| `test` | 10 | 1000–1009 | 8 machines, workspace half-extent 15 | all reported comparisons |
| `generalization` | 6 | 4000–4005 | 6–12 machines, half-extent 11–20 | out-of-distribution instances (not exercised by the main pipeline stages) |

**Per-instance generation, in order:**

1. **Machine dimensions.** Each of the 8 machines draws `width ~ U(3, 6)` and
   `height ~ U(2.5, 5)`.
2. **Shape.** Alternating by index — even indices are L-shaped, odd are
   rectangles (4 of each for an 8-machine instance).
3. **L-shape construction.** An L-shaped machine's rectangular envelope gets a
   corner notch: `cutout_width ~ U(min(1.0, width/2), width/2)` and
   `cutout_height ~ U(min(1.0, height/2), height/2)` — the notch never exceeds
   half a side.
4. **Access-point placement.** Each machine gets its own access point sampled
   `(x, y) ~ U(-1.5, 1.5)²`, relative to that machine's own center —
   independent per machine, not fixed to an edge.
5. **Visiting sequence.** Fixed at `[1, 2, ..., n]` in machine-id order — the
   GA never reorders it, only the layout (position + rotation) is optimized.

**Feasibility / difficulty.** `feasibility_rate()` Monte-Carlo samples 2,000
uniform-random layouts per instance and measures the fraction that satisfy
every hard constraint. On the shipped instances this averages roughly 4–5%,
with mean workspace occupancy around 14% — difficulty comes from
collision/reachability constraints, not sheer space pressure.

`dump_instance_catalogue()` (the `instances` pipeline stage) writes every
instance's full geometry plus these descriptive features to
`results/results_rl_ga/instances.json`.

---

## 2. The static genetic algorithm — `optimization/ga_core.py`, `feasibility/`

`ConfigurableGA` is **one** GA implementation shared by every variant compared
in this study — static, scheduled, and RL-controlled variants all run the
exact same selection/crossover/mutation/elitism/local-search code. Only two
things change between variants:

* **`ls_target`** — which individuals local search is offered to:
  `"offspring"`, `"elites"`, or `"none"`.
  * **elites**: each of the top `elitism_rate × population_size` individuals
    (carried over unchanged by elitism) may additionally get hill-climbed,
    with probability `ls_rate`, before being copied into the next generation.
  * **offspring**: each freshly bred child (after crossover + mutation) may
    get hill-climbed, with probability `ls_rate`, before joining the next
    generation.
* **`ls_controller`** — how `ls_rate` (and, for RL, mutation/crossover rate
  too) evolves over the run: `"static"` (fixed), `"schedule"`
  (deterministic, generation-dependent), or `"rl"` (the Q-learning agent,
  see §3).

**Per-generation loop** (`_next_generation`):

1. **Elitism.** The best `elitism_rate × population_size` individuals (10% of
   200 = 20) are copied forward unchanged (optionally hill-climbed, if
   `ls_target == "elites"`).
2. **Reproduction**, until the population is refilled:
   tournament selection (`tournament_k = 3`) → uniform crossover
   (`crossover_rate`) → Gaussian mutation (`mutation_rate`, per-gene jitter +
   random 90° re-orientation) → optional local search (if
   `ls_target == "offspring"`).
3. **Local search** (`local_search`) is first-improvement hill-climbing: pick
   one random gene, perturb it, keep the move only if fitness improves,
   repeat `ls_iterations` (5) times. Costs `iterations + 1 = 6` evaluations
   per call — the single most expensive operator per use, which is why
   *where* it's applied matters so much to the total evaluation budget.

**Fitness / feasibility** (`feasibility/evaluator.py`, backed by
`feasibility/collision_detector.py`): a layout is feasible only if no two
machines overlap, every machine stays inside the workspace, and the robot's
origin can reach every access point; infeasible layouts score `+inf`.
Feasible layouts are scored by total robot travel distance over the fixed
visiting sequence. `LayoutEvaluator` is a vectorised NumPy re-implementation
of the same three feasibility checks plus the travel-distance objective,
~45× faster than a naive geometric implementation and checked bit-identical
by `tests/test_fast_eval_equivalence.py`.

**Budget accounting.** Every fitness evaluation is counted, *including* the
ones consumed inside local search, and every generation records the
cumulative evaluation count and cumulative wall-clock/CPU time. This is what
lets the analysis stage compare variants at an equal evaluation budget, not
just an equal generation count — two variants at the same nominal `ls_rate`
can spend very different budgets depending on `ls_target` (elites get offered
LS at most 20×/generation, offspring at most 90×/generation, so the same
rate costs up to 4.5× more evaluations under `ls_target="offspring"`).

---

## 3. The reinforcement-learning structure — `optimization/rl_control.py`

When `ls_controller == "rl"`, a tabular Q-learning agent (`QLearningAgent`)
is queried every `control_interval` generations (default 1, i.e. every
generation) and picks one action that nudges the GA's live parameters.

**State space — 9 states.** `state = 3 × diversity_bin + improvement_bin`,
where:
* **diversity_bin** (0/1/2) comes from the population's mean pairwise
  Euclidean distance between chromosomes, discretised by calibrated
  tertile edges.
* **improvement_bin** (0/1/2) comes from the sign/magnitude of the previous
  generation's *relative* best-fitness improvement (using relative rather
  than absolute improvement lets the same thresholds transfer across
  instances with different fitness scales).

These tertile edges are calibrated once, on the **training split only**
(`train/calibrate_state_bins.py`), and saved to
`results/results_rl_ga/agents/state_bins.json` — evaluation runs load and reuse these
fixed edges rather than recomputing them, so test-split data never leaks
into the state discretisation.

**Action space — 7 actions** (`BASE_ACTIONS` in `rl_control.py`):
`mutation_rate_up`, `mutation_rate_down`, `crossover_rate_up`,
`crossover_rate_down`, `local_search_rate_up`, `local_search_rate_down`, and
`soft_reset` (partially relaxes mutation/crossover back toward their
starting values and halves the local-search rate). Rate changes are bounded
steps (e.g. LS rate moves in steps of 0.1, clipped to `[0, 1]`).

**Reward** (`compute_reward`): relative best-fitness improvement over the
previous generation if the population improved; a fixed stall penalty
(`rl_stall_penalty`, default −0.1) if it didn't; 0 if the whole population is
still infeasible (avoids poisoning the Q-table with `inf/inf`). An optional
cost term can additionally penalise the extra evaluations local search
consumed, though it's off (`rl_cost_penalty = 0.0`) by default.

**Learning rule.** Standard tabular Q-learning:
`Q(s,a) += lr × (reward + gamma × max_a' Q(s',a') − Q(s,a))`, with
epsilon-greedy action selection. During training, epsilon decays each
episode from 0.30 to a floor of 0.05; during evaluation, epsilon is fixed at
0.05 and the agent keeps learning online (`rl_learn_online = True`) unless
told otherwise.

**Ablation registry.** `optimization/variants.py` also defines
`ablation_variants()` — four variants that restrict the agent to a subset of
the 7 actions (LS-rate only, mutation-rate only, crossover-rate only, or
everything except LS rate), to isolate which control dimension matters. These
are trained only if you explicitly pass `--train-ablation-agents` to
`run_all.py`; nothing in the `main`/`timing`/`analyze`/`figures` stages
evaluates them yet.

---

## 4. The compared variants — `optimization/variants.py`

The main grid is a full cross product of **3 local-search controllers** ×
**2 local-search targets**, plus one shared no-LS reference point — **11
variants** total:

| | LS on offspring | LS on elites |
|---|---|---|
| no local search | `ga_ls00` (shared reference) | |
| static 10% / 50% / 100% | `ga_ls10/50/100_offspring` | `ga_ls10/50/100_elites` |
| deterministic schedule | `ga_sched_offspring` | `ga_sched_elites` |
| **RL controller** | `rlga_offspring` | `rlga_elites` |

All variants sharing an `(instance, seed)` pair start from the **same
initial population and the same RNG seed** ("common random numbers"), so
every variant-vs-variant comparison is paired, not just independent samples.

---

## 5. Training vs. testing

**Training** (`train/train_agents.py`, driven by the `train` stage): each of
the two RL variants (`rlga_offspring`, `rlga_elites`) gets its own
independently trained agent, repeated for **3 seeds** each (agent seeds 0/1/2)
— 6 trained Q-tables total by default. Training runs GA episodes over the
**train split** (15 instances), learning rate 0.05, epsilon decaying
0.30 → 0.05 per episode. Each trained agent (Q-table + training history) is
saved as one `.npz` file under `results/results_rl_ga/agents/`.

**State-bin calibration** (`train/calibrate_state_bins.py`, the `calibrate`
stage) runs *before* training: it samples population-diversity and
positive-relative-improvement values from the train split only, and derives
the tertile edges every RL variant's state discretisation then uses.

**Testing / the main grid** (`optimization/run_experiments.py`, the `main`
stage): all 11 variants are evaluated on the **test split** (10 instances) ×
**10 seeds** = 1,100 runs, 300 generations each, population 200. RL variants
rotate through the 3 independently trained agents (`agent_seed = run % 3`)
so the reported numbers aren't those of one lucky training repetition. Every
run's summary row goes to `results/results_rl_ga/runs.csv`; the full per-generation trace
goes to `results/results_rl_ga/raw/<variant>/<instance>_run<k>.npz`.

**Timing** (the `timing` stage) re-runs the same 11-variant grid but
sequentially (`workers=1`, fewer seeds by default) so wall-clock numbers
aren't distorted by CPU contention between parallel runs — written to
`results/results_rl_ga/runs_timing.csv`.

---

## 6. Where the results live — `results/`

```
results/
├── agents/                       trained Q-tables (.npz, 2 variants x 3 seeds),
│                                  training histories, state_bins.json,
│                                  policy_stability.json
├── raw/<variant>/                per-run anytime traces (best-so-far fitness,
│                                  evaluations, diversity, RL state/action/reward, ...)
├── analysis/                     summary_overall.csv, summary_per_instance.csv,
│                                  paired_tests.csv, budget_matched.csv,
│                                  instance_features.csv, report.md
├── figures/                      convergence plots, box plots, paired
│                                  differences, quality-vs-effort, RL behaviour
├── instances.json                the full instance catalogue with features
├── runs.csv                      one row per (variant, instance, seed) - main grid
├── runs_metadata.json            exact settings/environment used for runs.csv
├── runs_timing.csv               sequential wall-clock replication
└── runs_timing_metadata.json     settings/environment used for runs_timing.csv
```

`analysis/report.md` (produced by the `analyze` stage) is the best single
place to start reading the results — it's the same tables above, rendered
for reading rather than for scripts.

---

## 7. How to run it

### All at once

```bash
pip install -r requirements.txt
./run/run_pipeline.sh --replicate      # reuse the shipped agents, main grid only (~40 min)
./run/run_pipeline.sh                  # rebuild everything from scratch, retraining agents
./run/run_pipeline.sh --study myrun    # same, but writes to results/myrun/ instead of
                                    # overwriting the archived results/
```

`run/run_pipeline.sh` is a thin wrapper around `run/run_all.py` that also picks a
working Python interpreter and logs to `logs/<study>_<timestamp>.log`. Run
`./run/run_pipeline.sh --help` for every flag (control interval, split sizes,
seeds, generations, population size, RL stall penalty, worker count, etc).

### The underlying stages — `run_all.py`

`run_all.py` runs seven stages in order; each can be run alone with
`--only <stage>`, or skipped with `--skip <stage>`:

| stage | what it does |
|---|---|
| `instances` | writes the instance catalogue and per-instance features to `instances.json` |
| `calibrate` | derives the RL state-bin tertile edges from the training split |
| `train` | trains the RL agents (one per variant, several independent seeds) |
| `main` | runs the full 11-variant grid on the test split → `runs.csv` + `raw/` |
| `timing` | sequential wall-clock replication (one run at a time) → `runs_timing.csv` |
| `analyze` | turns `runs.csv` into summary tables and `analysis/report.md` |
| `figures` | renders every plot into `figures/` |

Run everything:
```bash
python -m run.run_all
```

Run just one stage (e.g. re-generate figures after tweaking a plot):
```bash
python -m run.run_all --only figures
```

Run a subset of stages, e.g. skip retraining and go straight to evaluation +
analysis using the already-trained agents:
```bash
python -m run.run_all --only main timing analyze figures
```

Useful flags: `--runs N` (seeds per instance in the main grid), `--generations`,
`--population-size`, `--agent-seeds`, `--control-interval`, `--rl-penalty`,
`--agent-dir` / `--bins` (point at a different trained-agent set),
`--results-dir` (write outputs somewhere other than `results/`),
`--train-ablation-agents` (also train the 4 per-parameter ablation agents,
otherwise skipped).

### Standalone figure scripts — `visualization/extra/`

Each of these is independently runnable and writes into
`results/results_rl_ga/figures/extra/` by default:

```bash
python visualization/extra/quality_vs_evaluations.py
python visualization/extra/quality_vs_walltime.py [--cpu]
python visualization/extra/paired_quality_vs_evaluations.py
python visualization/extra/convergence_by_metric.py     # gens / evals / seconds
python visualization/extra/ls_rate_by_instance.py       # what the agent settled on
python visualization/extra/q_tables_deployed.py         # deployed Q-tables
python visualization/extra/training_summary.py          # the training stage
python visualization/extra/test_run_layout.py           # re-runs a single instance
```

### Tests

```bash
python -m tests.test_stats_utils            # statistics vs brute force
python -m tests.test_problem_parity         # instances unchanged since submission
python -m tests.test_fast_eval_equivalence  # vectorised evaluator vs reference geometry
```

### Supplementary, on-demand analysis

`optimization/control_matched_rate.py` is a manual, on-demand control study
(not part of the automated stages above) that checks whether an RL variant's
apparent gain is genuine adaptation or just an artifact of settling at a
cheaper average local-search rate:

```bash
python -m optimization.control_matched_rate [--workers N]
```

Outputs go to `results/control_matched/`.
