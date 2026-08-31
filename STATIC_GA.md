# Running the static GA — `run_static_ga.py`

A pipeline for the **static and deterministic GA variants only**. No
reinforcement learning: no `calibrate` stage, no `train` stage, no Q-table is
loaded, and no trained agent is needed to run anything here.

Its reason for existing is that the GA operators — tournament size, elitism
rate, crossover rate, mutation rate, local-search rate — are **command-line
arguments**, not fixed constants. `run_all.py` resolves each variant from a
registry inside the worker processes and can only vary generations, population
size and control interval; this script hands the worker the configuration
itself, so anything in `GAConfig` can come from a flag.

Run every command from the repository root, so `python -m run.run_static_ga`
resolves.

```bash
python -m run.run_static_ga --list-variants     # what can be run
python -m run.run_static_ga --help              # every flag
python -m run.run_static_ga                     # all 9 variants, all 5 stages
```

---

## Contents

- [Quick start](#quick-start)
- [The five stages](#the-five-stages)
- [The variants](#the-variants)
- [Choosing which variants to run](#choosing-which-variants-to-run)
- [The GA operators](#the-ga-operators)
- [Stacking runs](#stacking-runs)
- [What gets written](#what-gets-written)
- [Reference and baseline](#reference-and-baseline)
- [How long it takes](#how-long-it-takes)
- [Things that will catch you out](#things-that-will-catch-you-out)
- [Error messages](#error-messages)
- [Full flag reference](#full-flag-reference)

---

## Quick start

```bash
pip install -r requirements.txt      # numpy, and matplotlib for the figures
```

A fast smoke test (seconds, not minutes) before committing to a real run:

```bash
python -m run.run_static_ga \
  --variants no-ls ls100-elites \
  --instances 0 1 --runs 2 \
  --generations 20 --population-size 30 \
  --results-dir results/smoke
```

The real thing — every static variant, the default budget, all five stages:

```bash
python -m run.run_static_ga
```

Results land in **`results/results_rl_ga/static_ga/`** (see
[What gets written](#what-gets-written)).

---

## The five stages

They run in this order. `--only <stage> [...]` runs a subset; `--skip <stage>`
runs everything else.

| stage | what it does | needs |
|---|---|---|
| `instances` | writes `instances.json`: the instance catalogue with per-instance features | — |
| `main` | the grid of selected variants → `runs.csv` + `raw/` traces | — |
| `timing` | a **sequential** replication (one run on the machine at a time) → `runs_timing.csv` | — |
| `analyze` | statistics, paired tests, `analysis/report.md` | `runs.csv` |
| `figures` | every plot → `figures/` | `runs.csv`, `raw/`, matplotlib |

```bash
python -m run.run_static_ga --only main                 # just produce runs.csv
python -m run.run_static_ga --only analyze figures      # re-derive from an existing runs.csv
python -m run.run_static_ga --skip timing               # everything but the slow sequential pass
```

`--only analyze figures` reads the variants back out of `runs.csv`, so it
covers whatever is in the file — including runs from earlier invocations. It
does not re-run the GA, so it takes seconds.

**Why `timing` is separate.** The `main` stage runs in parallel, and parallel
runs share the machine, which inflates and adds noise to wall-clock time. The
`timing` stage reruns a smaller grid one run at a time so the wall-clock
comparison is clean. Evaluation counts and CPU time from `main` stay valid
either way. `timing` ignores `--workers` — it is always sequential — which is
why it is usually the slowest stage. Skip it if you only care about solution
quality.

---

## The variants

`--variants` takes the short names in the first column, the group aliases, or
the full variant keys in the second column. The keys match
`optimization.variants.main_variants` exactly, so a `runs.csv` from this script
stays comparable with the archived RL-GA study's rows.

| short name | variant key | local search |
|---|---|---|
| `no-ls` | `ga_ls00` | none — the reference point |
| `ls10-offspring` | `ga_ls10_offspring` | 10 % on offspring |
| `ls50-offspring` | `ga_ls50_offspring` | 50 % on offspring |
| `ls100-offspring` | `ga_ls100_offspring` | 100 % on offspring |
| `deterministic-offspring` | `ga_sched_offspring` | scheduled, on offspring |
| `ls10-elites` | `ga_ls10_elites` | 10 % on elites |
| `ls50-elites` | `ga_ls50_elites` | 50 % on elites |
| `ls100-elites` | `ga_ls100_elites` | 100 % on elites |
| `deterministic-elites` | `ga_sched_elites` | scheduled, on elites |
| `custom` | `ga_ls<NN>_<target>` | `--ls-rate` on `--ls-target` |

**Group aliases**, for when you don't want to type nine names:

| alias | expands to |
|---|---|
| `all` | all nine (the default) |
| `offspring` | the four offspring variants |
| `elites` | the four elites variants |
| `fixed` | the seven fixed-rate ones — `no-ls` plus `ls10/50/100` on both targets |
| `deterministic` | `deterministic-offspring`, `deterministic-elites` |

### What the two targets mean

Local search is first-improvement hill climbing on one randomly chosen gene,
costing `--ls-iterations + 1` objective evaluations every time it fires. All of
those evaluations are counted against the budget, which is why the higher LS
rates reach fewer generations per unit of effort.

- **`elites`** — each surviving elite gets local search with probability
  `ls_rate`, independently.
- **`offspring`** — each newly bred pair gets local search applied to one child
  with probability `ls_rate`.
- **`none`** — the `ga_ls00` reference point. It isolates how much of the
  performance comes from local search rather than from the GA itself.

### The deterministic variant

`deterministic-offspring` / `deterministic-elites` use the `schedule`
controller: the LS rate is a fixed, generation-dependent step function, with no
learning and no randomness in the rate itself. The default schedule is
`0:0.1 100:0.5 200:1.0` — 10 % for the first 100 generations, 50 % for the
next 100, 100 % thereafter. Override it with `--ls-schedule`:

```bash
python -m run.run_static_ga --variants deterministic \
  --ls-schedule 0:0.2 50:0.6 150:1.0
```

Entries are `GENERATION:RATE`, must start at generation 0, and are sorted for
you. The flag has no effect on the fixed-rate variants.

### The custom variant

For an LS rate the catalogue doesn't cover:

```bash
python -m run.run_static_ga --variants custom --ls-rate 0.25 --ls-target elites
#   -> ga_ls25_elites, "GA 25% LS [LS on elites]"
```

`--ls-rate` applies **only** to `custom` — the catalogue variants carry their
own rate, and the script refuses `--ls-rate` without `custom` rather than
silently ignoring it. `--ls-rate 0` collapses to `ga_ls00` whatever
`--ls-target` says, since 0 % on elites and 0 % on offspring are the same
algorithm.

You can mix `custom` in with catalogue variants in one invocation:

```bash
python -m run.run_static_ga --variants no-ls ls10-elites custom --ls-rate 0.25
```

---

## Choosing which variants to run

```bash
# one variant
python -m run.run_static_ga --variants ls100-elites

# several, by short name
python -m run.run_static_ga --variants no-ls ls10-elites ls100-elites

# by full key — same thing
python -m run.run_static_ga --variants ga_ls00 ga_ls10_elites ga_ls100_elites

# a whole group
python -m run.run_static_ga --variants elites

# groups and names together; duplicates are collapsed
python -m run.run_static_ga --variants elites no-ls

# the LS-rate sweep on both targets, without the deterministic ones
python -m run.run_static_ga --variants fixed
```

You can also cut the grid down along the other two axes while iterating:

```bash
python -m run.run_static_ga --instances 0 1 2 --runs 3      # 3 instances, 3 seeds
python -m run.run_static_ga --split train                   # the 15 training instances
```

`--instances` takes **indices**, not names: `0` is `test_1`, `9` is `test_10`.
The default is every instance in the split (10 for `test`, 15 for `train`).

---

## The GA operators

Every operator flag applies to **every variant in that invocation**. That is
what makes the comparison meaningful: within one run of the script the variants
differ only in their local search, so any difference between them is
attributable to it.

| flag | default | what it controls |
|---|---|---|
| `--population-size` | 200 | individuals per generation |
| `--generations` | 300 | generations per run |
| `--tournament-k` | 3 | tournament selection: `k` distinct individuals are drawn uniformly, the fittest wins. `1` makes selection uniformly random; larger `k` raises selection pressure. Must be ≤ `--population-size` |
| `--elitism-rate` | 0.1 | fraction of the population copied through unchanged. The elite count is `max(1, int(rate * population_size))`, so at least one elite always survives. Must be < 1 |
| `--crossover-rate` | 0.8 | probability that a selected pair is crossed (uniform crossover). Otherwise both parents pass through unchanged |
| `--mutation-rate` | 0.1 | **per-machine** probability of Gaussian jitter plus a random 90° re-orientation — not per-individual |
| `--ls-iterations` | 5 | hill-climbing steps each time local search fires; costs `iterations + 1` evaluations |
| `--ls-rate` | — | the LS rate for `custom` only |
| `--ls-target` | `elites` | where `custom` applies local search |
| `--ls-schedule` | `0:0.1 100:0.5 200:1.0` | the schedule for the deterministic variants only |

All rates are validated at parse time: they must be in `[0, 1]`
(`--elitism-rate` in `[0, 1)`), and the integer flags must be at least 1. You
get an error message rather than a run that quietly did something odd.

```bash
python -m run.run_static_ga \
  --variants elites \
  --tournament-k 5 --elitism-rate 0.05 \
  --crossover-rate 0.9 --mutation-rate 0.15 \
  --ls-iterations 10
```

The exact operator settings for every variant are recorded in
`runs_metadata.json`, so a result directory always says what produced it.

---

## Stacking runs

Two flags let several invocations accumulate into one analysis instead of
overwriting each other.

- **`--append`** merges into the existing `runs.csv` rather than replacing it.
  A repeated `(variant, instance, seed)` is *updated in place*, not duplicated,
  so re-running something to fix it does the right thing.
- **`--tag NAME`** suffixes every variant key and label, so a second operator
  setting sits **alongside** the first instead of replacing it.

Without a tag, running `ls10-elites` twice produces the same variant key twice
and the second run replaces the first. With a tag, you get two distinct
variants the analysis will compare against each other.

### Worked example: comparing two tournament sizes

```bash
# first setting
python -m run.run_static_ga --variants elites --tag k3 --tournament-k 3 --only main

# second setting, stacked onto the first
python -m run.run_static_ga --variants elites --tag k7 --tournament-k 7 --only main --append

# analyse and plot everything in the file together
python -m run.run_static_ga --only analyze figures
```

`runs.csv` now holds eight variants — `ga_ls10_k3_elites`, `ga_ls10_k7_elites`,
`ga_ls50_k3_elites`, and so on — and the report's paired tests compare them all
on matched `(instance, seed)` pairs.

Tags may contain letters, digits and dashes. The tag is inserted **before** the
target (`ga_ls10_k7_elites`, not `ga_ls10_elites_k7`) because the figure code
groups variants by the target their key ends with; putting it last would break
the per-target convergence plots.

### Notes on stacking

- `runs_metadata.json` accumulates too. It keeps the configuration of every
  variant across invocations and appends an `invocations` list recording each
  command line, so a stacked directory remains self-describing.
- `--append` applies to the `timing` stage's `runs_timing.csv` as well, on the
  same terms.
- Stacked variants need not cover the same instances or seeds. The analysis
  pairs on shared `(instance, seed)` pairs and reports `n_pairs`, so a variant
  run on a subset is compared only where a comparison exists. Coverage that
  differs a lot still makes the pooled medians less comparable — prefer running
  the same `--instances` and `--runs` for anything you intend to compare
  directly.
- Without `--append`, the `main` stage **replaces** `runs.csv`. Stale `raw/`
  traces from variants no longer in the CSV are left on disk but ignored.

---

## What gets written

Everything goes under `--results-dir`, which defaults to
**`results/results_rl_ga/static_ga/`**:

```
results/results_rl_ga/static_ga/
├── instances.json              the instance catalogue with features
├── runs.csv                    one row per (variant, instance, seed)
├── runs_metadata.json          operators, variants, environment, invocations
├── runs_timing.csv             the sequential timing replication
├── runs_timing_metadata.json
├── raw/<variant>/<instance>_run<N>.npz    per-generation anytime traces
├── analysis/
│   ├── report.md                 the readable version of everything below
│   ├── summary_overall.csv       one row per variant, pooled over instances
│   ├── summary_per_instance.csv  one row per (variant, instance)
│   ├── paired_tests.csv          every variant against the reference
│   ├── budget_matched.csv        equal-evaluation and equal-time comparison
│   ├── instance_features.csv     features joined with the per-instance gain
│   └── feature_gain_correlations.json
└── figures/
    ├── convergence_pooled.png    plus convergence_{offspring,elites}.png,
    │                             one per target that has variants in the run
    ├── per_instance_boxplots.png
    ├── paired_differences_<reference>.png
    ├── quality_vs_effort.png
    ├── feature_gain.png          needs at least 3 instances — the feature
    │                             correlations are undefined below that
    └── per_instance/convergence_<instance>.png
```

> **The default is `results/results_rl_ga/static_ga/`, not `results/`, on purpose.**
> `results/` holds the archived RL-GA study, and `run_all.py` overwrites it in
> place. A static-only `runs.csv` written there would silently invalidate the
> committed analysis and figures. Pass `--results-dir results` only if you
> genuinely mean to overwrite the archive.

Use separate directories for experiments you want to keep apart:

```bash
python -m run.run_static_ga --variants elites --results-dir results/static_elites
python -m run.run_static_ga --variants offspring --results-dir results/static_offspring
```

`--no-trace` skips writing `raw/`. It saves disk, but the convergence figures
and the budget-matched table both read those traces, so the `analyze` and
`figures` stages lose their most interesting output. The `timing` stage never
writes traces.

---

## Reference and baseline

The analysis and figure code was written around the RL-GA study, where the
reference variant was `rlga_elites`. That variant does not exist here, so the
script picks sensible substitutes from whatever is actually in `runs.csv`:

- **reference** — the strongest local-search variant present (`ga_ls100_elites`
  by preference, then `ga_ls100_*`, `ga_sched_*`, `ga_ls50_*`). Everything in
  the report's paired tests is compared against it.
- **baseline** — the weakest (`ga_ls00`, then `ga_ls10_offspring`, then any
  `ga_ls10_*`). The per-instance "gain" columns measure improvement from it.

Both are printed at the start of the `analyze` and `figures` stages so there is
no guessing. Override either with a full variant key:

```bash
python -m run.run_static_ga --only analyze figures \
  --reference ga_ls50_elites --baseline ga_ls00
```

The variant named must be present in `runs.csv`, or you get an error listing
what is.

---

## How long it takes

Measured from the archived study's sequential timing pass, at the default
300 generations × population 200, one run is roughly:

| variant | CPU seconds per run |
|---|---|
| `ga_ls00` | 6.7 |
| `ga_ls10_elites` | 7.2 |
| `ga_ls10_offspring` | 7.6 |
| `ga_ls50_elites` | 8.0 |
| `ga_sched_elites` | 8.8 |
| `ga_ls100_elites` | 8.9 |
| `ga_ls50_offspring` | 10.4 |
| `ga_sched_offspring` | 10.9 |
| `ga_ls100_offspring` | 13.1 |

So the full default job — 9 variants × 10 instances × 10 seeds = 900 runs — is
about **2.3 CPU-hours**, roughly 10 minutes of wall-clock on 22 workers.

The `timing` stage is the expensive one, because it is sequential by design:
9 × 10 × 3 = 270 runs at ~9 s each is about **40 minutes** whatever
`--workers` says. `--skip timing` if you don't need the wall-clock table.

Scale roughly linearly in variants × instances × seeds × generations ×
population size. `--workers` defaults to `cpu_count - 2`.

---

## Things that will catch you out

- **`--workers` does nothing for `timing`.** That stage is deliberately
  sequential; see [The five stages](#the-five-stages).
- **`--instances` takes indices, not names.** `--instances 0` means `test_1`.
- **Split sizes must be set as environment variables**, not flags, if you
  change them. Worker processes rebuild the instances themselves and inherit
  `os.environ`, not the parent's arguments:
  ```bash
  FYP_TEST_INSTANCES=4 python -m run.run_static_ga --only main
  ```
- **`--tournament-k` must be ≤ `--population-size`.** Selection draws that many
  *distinct* individuals. The script checks this before starting.
- **`--mutation-rate` is per machine, not per individual.** At `0.1` on a
  10-machine layout, most individuals get at least one machine mutated.
- **Seeds are shared across variants by construction.** The run seed is
  `42 + 100 * instance_index + run`, so every variant on a given
  `(instance, seed)` pair starts from the *same* initial population. That is
  what makes the paired statistics valid — and it means the same command is
  reproducible run to run.
- **`--only analyze` needs a `runs.csv`** in `--results-dir`. It will tell you
  if there isn't one rather than producing an empty report.
- **Figures need matplotlib.** Every other stage runs without it.
- **The `--seed` you may be looking for doesn't exist.** Use `--runs N` to get
  N seeds per instance; they are derived deterministically.

---

## Error messages

Every one of these stops before doing any work:

| message | fix |
|---|---|
| `unknown variant 'x'. Run with --list-variants...` | typo, or a variant key from the RL study |
| `--variants custom needs --ls-rate` | add `--ls-rate 0.25` |
| `--ls-rate only applies to the 'custom' variant` | drop `--ls-rate`, or add `custom` to `--variants` |
| `--tag may only contain letters, digits and dashes` | no underscores or spaces in the tag |
| `--tournament-k (30) cannot exceed --population-size (20)` | lower `k`, or raise the population |
| `--elitism-rate must be between 0 and 1 (exclusive)` | elitism of 1.0 would leave no room to breed |
| `--ls-schedule entries look like GEN:RATE` | use `0:0.1 100:0.5`, colon-separated |
| `--ls-schedule must start at generation 0` | the first entry has to be `0:<rate>` |
| `--instances [99] out of range` | the split has 10 instances, indices 0–9 |
| `<path>/runs.csv does not exist` | run the `main` stage first |
| `--reference 'x' is not in the results` | the message lists what is |
| `--skip removed every stage; nothing to do` | you skipped all five |

---

## Full flag reference

`python -m run.run_static_ga --help` prints this list. Grouped as the parser groups
them:

**Stages**

| flag | default | meaning |
|---|---|---|
| `--only STAGE [...]` | all five | run only these stages |
| `--skip STAGE [...]` | none | run everything except these |

**What to run**

| flag | default | meaning |
|---|---|---|
| `--variants NAME [...]` | `all` | short names, group aliases or full keys |
| `--list-variants` | — | print the catalogue and exit |
| `--split {train,test}` | `test` | which instance split |
| `--instances INDEX [...]` | all | restrict to these instance indices |
| `--runs N` | 10 | seeds per instance in the main grid |
| `--timing-runs N` | 3 | seeds per instance in the timing pass |
| `--workers N` | `cpu_count - 2` | parallel workers for `main` only |

**GA operators** — see [The GA operators](#the-ga-operators)

| flag | default |
|---|---|
| `--population-size N` | 200 |
| `--generations N` | 300 |
| `--tournament-k N` | 3 |
| `--elitism-rate R` | 0.1 |
| `--crossover-rate R` | 0.8 |
| `--mutation-rate R` | 0.1 |
| `--ls-iterations N` | 5 |
| `--ls-rate R` | — (`custom` only) |
| `--ls-target {offspring,elites}` | `elites` (`custom` only) |
| `--ls-schedule GEN:RATE [...]` | `0:0.1 100:0.5 200:1.0` |

**Output**

| flag | default | meaning |
|---|---|---|
| `--results-dir DIR` | `results/results_rl_ga/static_ga` | where everything is written |
| `--tag NAME` | none | suffix variant keys/labels so runs stack |
| `--append` | off | merge into an existing `runs.csv` |
| `--no-trace` | off | skip `raw/` traces |
| `--reference KEY` | auto | the variant everything is compared against |
| `--baseline KEY` | auto | the variant the gain is measured from |

---

## Relation to the rest of the project

`run_all.py` is the full seven-stage study including RL training; this script is
its static-only sibling. They write the same CSV schema and the same metadata
layout, and share `optimization/ga_core.py`, `analysis/analyze.py` and
`visualization/plots.py` — so a directory produced here can be read with the
same tools, and the variant keys line up with the archived study's.

See `README.md` for the project layout and the full study, and `RUNDOWN.md` for
the detail behind the method.
