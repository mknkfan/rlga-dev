# Robot-cell layout GA

A self-contained copy of the code that produces the study: robot-cell
layout optimisation with a genetic algorithm whose **local-search rate, mutation
rate and crossover rate are steered by a tabular Q-learning controller**,
compared against static and scheduled baselines running on the *same* GA.

The archived outputs of that study — run records, raw traces, statistics,
figures and the trained agents — ship alongside the code in `results/`.

```bash
pip install -r requirements.txt

# reproduce the archived numbers from the shipped agents  (~40 min, 22 workers)
./run_pipeline.sh --replicate

# rebuild the whole study, retraining the agents, into a new directory
./run_pipeline.sh --study rerun
```

Run every command from inside this directory (the repository root), so `python -m <module>` resolves.

---

## Layout

```
./
├── core/                      the problem
│   ├── datastruct.py            Point and Machine dataclasses: geometry,
│   │                            rotation, world-frame corners, access points
│   └── problems.py              the instance family, the disjoint train /
│                                test / generalisation splits, per-instance
│                                features (occupancy, feasibility rate, …)
│
├── feasibility/               is this layout legal, and how good is it
│   ├── collision_detector.py    exact geometry: ray-casting point-in-polygon,
│   │                            segment intersection, polygon overlap
│   └── evaluator.py             LayoutEvaluator — the vectorised NumPy
│                                re-implementation the experiments call.  Folds
│                                the same three feasibility tests (machine-machine
│                                overlap, workspace bounds, robot origin covered)
│                                and the travel-distance objective into one pass;
│                                infeasible layouts return +inf.  ~45x faster and
│                                bit-identical — see the equivalence test below.
│
├── optimization/              the search
│   ├── ga_core.py               ConfigurableGA — ONE genetic algorithm in which
│   │                            the local-search target (offspring / elites /
│   │                            none) and the parameter controller (static /
│   │                            schedule / rl) are independent options over
│   │                            shared code, plus full evaluation accounting
│   ├── rl_control.py            the Q-learning agent: 7-action space, 3x3 state
│   │                            discretisation (diversity x improvement), reward
│   ├── variants.py              the 11 compared configurations, and the
│   │                            ablation registry
│   ├── run_experiments.py       the run grid: common random numbers, shared
│   │                            initial populations, parallel workers, runs.csv
│   └── control_matched_rate.py  supplementary control — a static GA at the RL
│                                variant's own mean LS rate, and a random-policy
│                                controller
│
├── train/                     making the agents
│   ├── calibrate_state_bins.py  state thresholds from the TRAINING split only
│   └── train_agents.py          training with independent repetitions, plus
│                                cross-seed policy-agreement reporting
│
├── analysis/                  turning runs into claims
│   ├── stats_utils.py           exact Wilcoxon, bootstrap CIs, Cliff's delta,
│   │                            Cohen's dz, Holm correction — no SciPy
│   ├── analyze.py               the tables and analysis/report.md
│   └── policy_stability_report.py  how much the learned policies agree
│
├── visualization/             every figure
│   ├── plots.py                 the standard set: convergence, box plots, paired
│   │                            differences, quality-vs-effort, RL behaviour,
│   │                            feature correlations
│   └── extra/                   standalone scripts, one figure each (below)
│
├── tests/                     the correctness claims this rests on
│   ├── test_fast_eval_equivalence.py  evaluator == collision_detector geometry
│   ├── test_problem_parity.py         test instances unchanged since submission
│   ├── test_stats_utils.py            statistics vs brute-force enumeration
│   └── reference_fyp/                 the previous FYP work's implementations the two
│                                      parity tests check against
│
├── results/                   outputs
│   ├── agents/                  trained Q-tables (2 variants x 3 seeds),
│   │                            training histories, state_bins.json,
│   │                            policy_stability.json
│   ├── raw/                     per-run anytime traces, one folder per variant
│   ├── analysis/                 summary tables, paired tests, report.md
│   ├── figures/                  all plots
│   ├── instances.json            the full instance catalogue
│   ├── runs.csv                  one row per (variant, instance, seed) - main grid
│   ├── runs_metadata.json        settings/environment for runs.csv
│   ├── runs_timing.csv           sequential wall-clock replication
│   └── runs_timing_metadata.json settings/environment for runs_timing.csv
│
├── paths.py                   every default location, resolved absolutely
├── run_all.py                 the stage orchestrator
└── run_pipeline.sh            one command for the whole thing
```

---

## What the study compares

Eleven variants over 10 test instances x 10 seeds = 1,100 runs, 300 generations
of population 200 each. All variants sharing an `(instance, seed)` pair start
from the **same initial population and the same RNG seed**, so every comparison
is paired.

| | LS on offspring | LS on elites |
|---|---|---|
| no local search | `ga_ls00` (shared reference point) | |
| static 10 / 50 / 100 % | `ga_ls10/50/100_offspring` | `ga_ls10/50/100_elites` |
| deterministic schedule | `ga_sched_offspring` | `ga_sched_elites` |
| **RL controller** | `rlga_offspring` | `rlga_elites` |

The grid is a full cross product on purpose: static-vs-RL is read *within* a
local-search target, so the controller is isolated from the search architecture.

## The seven stages

`run_all.py` runs them in order; `--only <stage>` runs one.

| stage | what it does |
|---|---|
| `instances` | writes the instance catalogue and per-instance features |
| `calibrate` | state-bin thresholds from the training split |
| `train` | trains the agents, one per (variant, seed) |
| `main` | the 11-variant grid → `runs.csv` + `raw/` |
| `timing` | sequential wall-clock replication (one run per machine) → `runs_timing.csv` |
| `analyze` | statistics, tables, `analysis/report.md` |
| `figures` | all plots |

**The archived `results/` directory is the `main` + `timing` + `analyze` +
`figures` stages.** `--replicate` reuses the shipped agents but only reruns
`main` + `analyze` + `figures` (so it reproduces `runs.csv`, not
`runs_timing.csv`).

## Reproducing

Settings come from `results/runs_metadata.json`: split `test`, 10
instances, 10 seeds each, 300 generations, population 200, control interval 1,
agent seeds 0/1/2 rotated, RL stall penalty −0.1.

```bash
# the shell wrapper, or the equivalent module call (run from the repository root):
FYP_TRAIN_INSTANCES=15 FYP_TEST_INSTANCES=10 \
python -m run_all --only main timing analyze figures \
  --runs 10 --control-interval 1 --agent-seeds 0 1 2 \
  --generations 300 --population-size 200 \
  --agent-dir results/agents \
  --bins results/agents/state_bins.json \
  --results-dir results
```

`--only main analyze figures` (dropping `timing`) reproduces `runs.csv` and
everything derived from it, but not `runs_timing.csv`.

The split sizes must be **environment variables**: multiprocessing workers
rebuild the instances themselves and inherit `os.environ`, not the parent's
arguments.

> **Every stage opens its outputs in `"w"` mode, with no backup and no prompt.**
> Re-running without `--study` / `--results-dir` overwrites `results/`
> in place. Pass `--study NAME` to write somewhere else.

Verified on this copy: re-running `analyze` over the shipped `raw/` traces
reproduces all seven files in `results/analysis/` **byte for byte**,
and `visualization/extra/test_run_layout.py` re-executes five variants from
scratch and recovers the stored fitness values exactly.

## Standalone figure scripts

Each writes into `results/figures/extra/` by default and takes an
optional output path.

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

## Tests

```bash
python -m tests.test_stats_utils            # statistics vs brute force
python -m tests.test_problem_parity         # instances unchanged
python -m tests.test_fast_eval_equivalence  # ~15s, 14,400 chromosomes
```

The last one is what licenses `feasibility/evaluator.py`: it checks the
vectorised evaluator against `tests/reference_fyp/original_ga.py`, which computes
fitness through `feasibility/collision_detector.py`, over every split.
Current result: 14,400 chromosomes, max relative error 0.000e+00, 45x faster.

## Notes

* Defaults resolve **absolutely** from `paths.py` rather than relative to
  the working directory, so a stage behaves the same wherever it is launched
  from, and regardless of what the repository's containing folder is named.
* `tests/reference_fyp/` holds a frozen, pre-refactor copy of the GA and
  instance generator used from the author's FYP work.
