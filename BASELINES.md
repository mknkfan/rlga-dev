# Static GA versus classical metaheuristics

The static GA is compared against four population-based metaheuristics —
differential evolution (DE), particle swarm optimisation (PSO), the artificial
bee colony (ABC) and charged system search (CSS) — on the same instances, from
the same initial populations, against the same objective and under the same
evaluation budget.

```bash
# the whole comparison: 10 test instances x 10 seeds x 5 methods
python -m run_baselines

# what each baseline's parameters are and where they come from
python -m run_baselines --list-parameters

# the properties the comparison rests on, checked rather than assumed
python -m tests.test_metaheuristics
```

Results land in `results/baselines/` — `runs.csv`, per-run traces under `raw/`,
statistics and `report.md` under `analysis/`, figures under `figures/` — in
exactly the format `results/static_ga/` uses, because the same analysis and
plotting code produces them.

---

## What is held fixed

Everything except the search strategy. The four baselines are implemented in
`optimization/metaheuristics.py` on top of the same components the GA uses, so
a difference in the results cannot come from anywhere else.

| Held fixed | Where it comes from |
|---|---|
| Representation | `[x, y, rotation]` per machine, 24 real genes for an 8-machine instance |
| Objective | `feasibility/evaluator.py` — travel distance + compactness − accessibility, `+inf` if any machine overlaps another, leaves the workspace or covers the robot origin |
| Bound repair | positions clipped to the machine's half-extent inside the workspace, rotation modulo 360 — `ConfigurableGA.apply_bounds`, reimplemented once as `PopulationOptimizer.repair` |
| Initial population | `make_initial_population(machines, bounds, 200, seed=run_seed)` — quadrant-seeded random layouts |
| Run seed | `42 + 100 * instance_index + run`, shared by every method on that pair |
| Budget | 60 000 objective evaluations (the GA's 200 × 300), counted inside every operator |
| Instances | the 10-instance `test` split, 10 independent seeds each |
| Recording | per-iteration best-so-far, cumulative evaluations, cumulative wall-clock and CPU time |

### The same starting point, literally

For a given (instance, seed) pair there is one initial population, and every
method is handed that same array of 200 layouts. `tests/test_metaheuristics.py`
asserts it: the first recorded iteration of the GA, DE, PSO, ABC and CSS agree
bit-for-bit on the best fitness, the mean fitness, the feasible fraction and the
population diversity of the starting population.

This is also what licenses the statistics in the report: results are compared as
matched pairs (Wilcoxon signed-rank on per-(instance, seed) differences), which
is only meaningful when the pairs share a starting point.

### The same budget

The GA without local search evaluates 200 individuals for each of 300
generations — 60 000 evaluations — and every baseline is capped there.

| Method | Evaluations per iteration | Iterations inside the budget |
|---|---|---|
| GA (no local search) | 200 (one population) | 300 |
| DE | 200 (one trial per member) | 300 |
| PSO | 200 (one per particle) | 300 |
| ABC | ~400 (employed + onlooker bees, plus the occasional scout) | ~150 |
| CSS | 200 (one per charged particle) | 300 |

ABC is the one method whose iteration is not one population wide, so it
completes about half as many cycles for the same effort. That is the correct
trade: the comparison is at equal *evaluations*, not equal iterations, because
an evaluation is what the objective costs. Section 5 of the generated report
additionally compares every method at the largest budget all of them reached and
at equal wall-clock time.

---

## The algorithms and their parameters

Each runs at its published default setting. Every value is overridable from the
command line (`python -m run_baselines --list-parameters`).

### Differential evolution — `de`

DE/rand/1/bin (Storn & Price, *Journal of Global Optimization* 11(4), 1997).
For each member: `v = x_r1 + F (x_r2 − x_r3)` with three distinct random members,
binomial crossover with a guaranteed inherited gene, then greedy selection.

| Parameter | Default | Note |
|---|---|---|
| `F` | 0.5 | mid-point of the recommended [0.4, 1.0] |
| `CR` | 0.9 | Storn & Price's recommendation for non-separable problems |
| `NP` | 200 | the shared population |

Acceptance is `f(trial) <= f(target)`, the original paper's own rule. It matters
more here than usual: while a member is infeasible its fitness is `+inf`, and
`<=` is what lets the trial replace it and the population drift towards
feasibility instead of freezing.

### Particle swarm optimisation — `pso`

Global-best PSO with the Clerc & Kennedy constriction constants (*IEEE
Transactions on Evolutionary Computation* 6(1), 2002).

| Parameter | Default | Note |
|---|---|---|
| `w` | 0.7298 | constriction factor χ |
| `c1`, `c2` | 1.49618 | χ · 2.05 |
| `v_max` | half of each variable's range | per-dimension clamp; initial velocities drawn uniformly inside it |

Personal and global bests update on a strict improvement, so an infeasible
particle keeps looking until it finds its first feasible layout.

### Artificial bee colony — `abc`

Karaboga's ABC (Erciyes University technical report TR06, 2005) with all three
phases: employed bees, onlooker bees and a scout.

| Parameter | Default | Note |
|---|---|---|
| `SN` (food sources) | 200 | *is* the shared initial population, so the colony is 2 × SN = 400 bees |
| `limit` | `SN × D` = 4800 | Karaboga's default abandonment threshold |

A neighbour perturbs one randomly chosen gene, `v_ij = x_ij + φ (x_ij − x_kj)`
with `φ ~ U(−1, 1)`, and replaces the source only if it is better. Onlookers
choose by roulette wheel over the textbook fitness transform — `1 / (1 + f)` for
`f ≥ 0`, `1 + |f|` otherwise — which maps an infeasible source (`f = +inf`) to
probability 0. Infeasible sources are therefore ignored by onlookers but still
worked on by employed bees and scouts.

Two consequences of the shared budget are worth stating plainly. `SN` is the
population size rather than half of it, because the food sources have to *be*
the shared initial population for the starting points to match; and with 60 000
evaluations ABC completes ~150 cycles, so a `limit` of 4800 trials is never
reached and the scout phase never fires. Both follow from the textbook defaults;
`--abc-limit` overrides the second if a stagnation-driven restart is wanted.

### Charged system search — `css`

CSS (Kaveh & Talatahari, *Acta Mechanica* 213, 2010): charged particles attract
each other by a Coulomb/Gauss law, with a charged memory of good solutions.

| Parameter | Default | Note |
|---|---|---|
| `CP` | 200 | the shared population |
| `CM` size | `CP / 4` = 50 | charged memory; also acts as a source of forces |
| `k_a` | `0.5 (1 + iter / iter_max)` | acceleration coefficient, increasing |
| `k_v` | `0.5 (1 − iter / iter_max)` | velocity coefficient, decreasing |
| `a` | 0.10 | radius separating the near and far field |
| `CMCR`, `PAR` | 0.95, 0.10 | harmony-search correction of out-of-range genes |

Charges are `q_i = (f_i − f_worst) / (f_best − f_worst)`, separation distances
are normalised by the pair midpoint's distance from the best solution, and the
moving probability is the paper's `p_ij`. Three things the paper leaves to the
implementation are pinned down here:

* **The force law is evaluated in the normalised search space** — every variable
  scaled to [0, 1] by its own range — so `a = 0.10 · max(range)` takes the
  paper's value of 0.10. In raw units the 360-degree rotation gene would set
  `a = 36`, and the `1 / a³` near-field term would freeze the swarm; the choice
  is about the variables' units, not about the algorithm. Within the normalised
  formulation the result is insensitive to it: `a` = 0.1, 0.5 and 1.0 give
  median fitness 58.4, 57.7 and 59.4 on a 6-run probe.
* **`+inf` cannot be normalised into a charge.** For the charge and moving
  probability formulas only, an infeasible particle is ranked one fitness-range
  beyond the worst feasible one. The objective, the selection and everything
  reported are untouched.
* **The force is summed over the sources, as published.** That sum grows with
  the particle count, and the paper's `k_a = 0.5` was written for its 20–40
  charged particles; at the 200 used here (the shared initial population) it
  pushes about 45 % of the position genes outside the box each iteration, where
  the harmony-search rule corrects them. Averaging the force instead removes
  that entirely (0 % corrections) but collapses the swarm onto a single point
  and searches *worse* — median fitness 82.4 against 58.4 on the same 6-run
  probe — so the published sum is the default. `--css-mean-force` repeats the
  measurement.

The charged memory records good layouts and pulls the particles towards them
through the force term; it never overwrites them. It also holds *distinct*
layouts only — a converging swarm re-finds its own best point repeatedly, and a
memory of N copies of one layout would drag every particle onto it (that
mistake collapsed the swarm to zero diversity by iteration 25 in an earlier
draft of this implementation).

---

## Reading the results

`results/baselines/analysis/report.md` is generated by the same code as the
static-GA report, with the GA as the reference:

1. **Solution quality and runtime**, pooled over instances — median [IQR], mean
   ± sd with a bootstrap CI.
2. **Paired comparison against the GA** on quality, then on runtime, then on
   effort — differences are `GA − baseline` on matched (instance, seed) pairs,
   so a negative difference means the GA is better. Two-sided Wilcoxon
   signed-rank, Holm-corrected within each family, with rank-biserial
   correlation and Cliff's delta as effect sizes.
3. **Per-instance breakdown**, so a method that wins on average but loses on a
   class of instances is visible.
4. **Equal-budget comparison** — quality at a common evaluation count and at a
   common wall-clock time, plus the effort each method needs to reach a quality
   level all of them attain.

Figures: `figures/convergence_pooled.png` (best-so-far against generations,
evaluations and wall-clock time, median with IQR band),
`figures/per_instance_boxplots.png`, `figures/paired_differences_ga_ls00.png`
and `figures/quality_vs_effort.png`.

### A caveat worth reporting

All five methods inherit the GA's constraint handling: an infeasible layout
scores `+inf` — a death penalty with no gradient towards feasibility, on a
problem where only about 8 % of random layouts are feasible. The GA's operators
were designed around that (elitism plus a mutation that re-orientates a single
machine); the four baselines meet it with their textbook operators and no
constraint-handling machinery of their own. Their numbers are therefore a fair
statement of *"what these algorithms do on this problem at their default
settings, from the same start, for the same budget"*, and not a claim about the
algorithms in general. A penalty-based or repair-based constraint handler would
be the natural next study, and would change the objective for every method
equally.

---

## Command reference

```bash
# everything, defaults
python -m run_baselines

# a subset of methods, fewer seeds
python -m run_baselines --algorithms de pso --runs 5

# against a GA with local search as well as the plain one
python -m run_baselines --ga-variants no-ls ls100-elites

# re-analyse or re-plot what is already there
python -m run_baselines --only analyze figures

# non-default algorithm parameters, stacked alongside the defaults
python -m run_baselines --algorithms de --de-f 0.8 --de-cr 0.5 --tag f08 --append

# add the baselines to the existing static-GA results directory instead
python -m run_baselines --ga-variants none --results-dir results/static_ga --append
```

`--tag` renames a variant (`de` → `de_f08`) so several parameter settings
accumulate in one `runs.csv` and appear side by side in one report; `--append`
merges into an existing one instead of replacing it.
