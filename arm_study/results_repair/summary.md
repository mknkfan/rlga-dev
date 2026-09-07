# Does the placement repair help?

150 paired runs (10 instances x 15 seeds), population 200, 300 generations, 60000 evaluations per run.

Each pair is one (instance, seed): both arms start from the identical rejection-sampled population and differ only in whether an unplaceable child is projected back onto the placeable set after mutation. Differences are taken within a pair and signed so **positive means repair did better**.


## Verdict

- **Cycle time** -- repair better by 0.3488 s on average (p = 0.0000)
- **Feasible population share** -- repair better by 0.1579 on average (p = 0.0000)
- **Evaluations to first feasible layout** -- no detectable difference (mean +0.0000 evals, p = 1.0000)
- **Gene diversity** -- repair worse by 0.3854 on average (p = 0.0013) (but see the caveat below: this metric is nearly blind to the thing repair would actually collapse)
- **CPU time** -- repair worse by 212.6259 s on average (p = 0.0000)

## Runs that found a feasible layout at all

| both | baseline only | repair only | neither | McNemar p |
| ---: | ---: | ---: | ---: | ---: |
| 150 | 0 | 0 | 0 | -- |

Only the discordant pairs carry information here; the exact test conditions on them.


## Paired metrics

| metric | pairs | baseline | repair | mean gain | 95% CI | repair wins | baseline wins | perm p | Wilcoxon p |
| --- | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: |
| `cycle_time` | 150 | 10.9699 | 10.6211 | 0.3488 | [0.3024, 0.3977] | 140 | 10 | 0.0000 | 0.0000 |
| `mean_feasible` | 150 | 0.5915 | 0.7494 | 0.1579 | [0.1491, 0.1666] | 150 | 0 | 0.0000 | 0.0000 |
| `final_feasible` | 150 | 0.6118 | 0.7603 | 0.1485 | [0.1365, 0.1605] | 145 | 5 | 0.0000 | 0.0000 |
| `evals_to_first` | 150 | 200 | 200 | 0 | [0, 0] | 0 | 0 | 1.0000 | -- |
| `mean_diversity` | 150 | 18.4280 | 18.0426 | -0.3854 | [-0.6142, -0.1567] | 61 | 89 | 0.0013 | 0.0014 |
| `cpu_time` | 150 | 295.34 | 507.96 | -212.63 | [-232.30, -192.62] | 10 | 140 | 0.0000 | 0.0000 |

`cycle_time` and `evals_to_first` are taken over the pairs where **both** arms found a feasible layout; the dropped counts are in `pairs_summary.csv`. Every row is signed so that positive favours repair -- for `mean_diversity` that means repair held *more* spread, and for `cpu_time` that it spent *less* time.


> **Caveat on `mean_diversity`.** `population_diversity` averages the per-gene standard deviation over every gene, and the rotation genes span 0-360 while the position genes span about 1.15 m. Rotation spread is therefore two orders of magnitude larger and the average is essentially rotation spread alone -- which is the one gene the repair never touches. Read this row as a check that repair leaves orientation diversity undisturbed, **not** as a measurement of whether it presses machines together onto the placement boundary. Positional collapse needs a position-only trace, and adding one to `ga.py` would change what every archived study reports, so it was measured separately instead: over 3 instances x 4 seeds x 150 generations the mean position spread is 73.8 mm without repair and 71.8 mm with it, a 2.7 % difference, and by the last generation the repaired runs hold slightly *more* spread (36.1 mm vs 34.0 mm). Both arms collapse from 377 mm to about 35 mm, which is the GA converging, not the repair flattening anything. The feared collapse does not happen.


## Per instance

| instance | pairs | both feasible | baseline s | repair s | gain s | repair wins | baseline feas. | repair feas. | baseline feas. share | repair feas. share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test_1 | 15 | 15 | 10.1290 | 9.9241 | 0.2049 | 14 | 15 | 15 | 0.5711 | 0.7602 |
| test_2 | 15 | 15 | 11.4043 | 11.0636 | 0.3407 | 15 | 15 | 15 | 0.5814 | 0.7526 |
| test_3 | 15 | 15 | 10.1188 | 9.8543 | 0.2645 | 12 | 15 | 15 | 0.6040 | 0.7522 |
| test_4 | 15 | 15 | 11.0992 | 10.7873 | 0.3119 | 14 | 15 | 15 | 0.5961 | 0.7269 |
| test_5 | 15 | 15 | 11.2495 | 10.8958 | 0.3537 | 14 | 15 | 15 | 0.5915 | 0.7340 |
| test_6 | 15 | 15 | 11.7049 | 11.3056 | 0.3993 | 14 | 15 | 15 | 0.5861 | 0.7176 |
| test_7 | 15 | 15 | 10.2592 | 9.9749 | 0.2843 | 14 | 15 | 15 | 0.6132 | 0.7835 |
| test_8 | 15 | 15 | 10.8130 | 10.4468 | 0.3662 | 14 | 15 | 15 | 0.5955 | 0.7542 |
| test_9 | 15 | 15 | 11.8208 | 11.2042 | 0.6165 | 15 | 15 | 15 | 0.6014 | 0.7239 |
| test_10 | 15 | 15 | 11.1005 | 10.7541 | 0.3463 | 14 | 15 | 15 | 0.5748 | 0.7894 |

## Reading it

**The budget is equal in evaluations, not in compute, and the two are far apart here.** An infeasible layout is rejected by the placement test and costs about 0.36 ms; a feasible one runs inverse kinematics, the manipulability check and the swept-path collision test over four branches, and costs about 6.1 ms -- some seventeen times more. Repair works by converting the cheap evaluations into expensive ones, so the repair arm buys strictly more computation for the same nominal budget. That is why `cpu_time` is in the table and why it should be read alongside the cycle time rather than as a footnote: a win at equal evaluations is a weaker claim than a win at equal seconds, and only the `cpu_time` row says which one this is.

The feasible-share column is the mechanism and the cycle-time column is the outcome, and they are not the same question. Repair can lift the share of the population that scores at all -- that is close to arithmetic -- without the extra evaluations landing anywhere useful, because a layout projected onto the placement boundary is a layout with the machines pressed up against each other and against the robot's column, which is where reach and manipulability are hardest to satisfy. If the share rises and the cycle time does not, that gap is the finding, not a bug.

