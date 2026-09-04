# Does the placement repair help?

150 paired runs (10 instances x 15 seeds), population 200, 174 generations, 34800 evaluations per run.

Each pair is one (instance, seed): both arms start from the identical rejection-sampled population and differ only in whether an unplaceable child is projected back onto the placeable set after mutation. Differences are taken within a pair and signed so **positive means repair did better**.


## Verdict

- **Cycle time** -- repair better by 0.3945 s on average (p = 0.0000)
- **Feasible population share** -- repair better by 0.1631 on average (p = 0.0000)
- **Evaluations to first feasible layout** -- no detectable difference (mean +0.0000 evals, p = 1.0000)
- **Gene diversity** -- repair worse by 0.5741 on average (p = 0.0001) (but see the caveat below: this metric is nearly blind to the thing repair would actually collapse)
- **CPU time** -- repair worse by 108.5166 s on average (p = 0.0000)

## Runs that found a feasible layout at all

| both | baseline only | repair only | neither | McNemar p |
| ---: | ---: | ---: | ---: | ---: |
| 150 | 0 | 0 | 0 | -- |

Only the discordant pairs carry information here; the exact test conditions on them.


## Paired metrics

| metric | pairs | baseline | repair | mean gain | 95% CI | repair wins | baseline wins | perm p | Wilcoxon p |
| --- | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: |
| `cycle_time` | 150 | 11.0325 | 10.6380 | 0.3945 | [0.3498, 0.4416] | 142 | 8 | 0.0000 | 0.0000 |
| `mean_feasible` | 150 | 0.5780 | 0.7411 | 0.1631 | [0.1543, 0.1718] | 150 | 0 | 0.0000 | 0.0000 |
| `final_feasible` | 150 | 0.6096 | 0.7675 | 0.1578 | [0.1449, 0.1709] | 146 | 4 | 0.0000 | 0.0000 |
| `evals_to_first` | 150 | 200 | 200 | 0 | [0, 0] | 0 | 0 | 1.0000 | -- |
| `mean_diversity` | 150 | 20.3491 | 19.7750 | -0.5741 | [-0.8448, -0.2989] | 58 | 92 | 0.0001 | 0.0001 |
| `cpu_time` | 150 | 143.42 | 251.94 | -108.52 | [-117.71, -99.36] | 6 | 144 | 0.0000 | 0.0000 |

`cycle_time` and `evals_to_first` are taken over the pairs where **both** arms found a feasible layout; the dropped counts are in `pairs_summary.csv`. Every row is signed so that positive favours repair -- for `mean_diversity` that means repair held *more* spread, and for `cpu_time` that it spent *less* time.


> **Caveat on `mean_diversity`.** `population_diversity` averages the per-gene standard deviation over every gene, and the rotation genes span 0-360 while the position genes span about 1.15 m. Rotation spread is therefore two orders of magnitude larger and the average is essentially rotation spread alone -- which is the one gene the repair never touches. Read this row as a check that repair leaves orientation diversity undisturbed, **not** as a measurement of whether it presses machines together onto the placement boundary. Positional collapse is not measured here: it would need a position-only trace in `ga.py`, and adding one changes what every archived study reports.


## Per instance

| instance | pairs | both feasible | baseline s | repair s | gain s | repair wins | baseline feas. | repair feas. | baseline feas. share | repair feas. share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test_1 | 15 | 15 | 10.1843 | 9.9404 | 0.2439 | 14 | 15 | 15 | 0.5563 | 0.7508 |
| test_2 | 15 | 15 | 11.4488 | 11.0890 | 0.3597 | 15 | 15 | 15 | 0.5625 | 0.7386 |
| test_3 | 15 | 15 | 10.1917 | 9.8762 | 0.3155 | 14 | 15 | 15 | 0.5968 | 0.7485 |
| test_4 | 15 | 15 | 11.2038 | 10.8051 | 0.3988 | 14 | 15 | 15 | 0.5816 | 0.7189 |
| test_5 | 15 | 15 | 11.2813 | 10.9103 | 0.3710 | 13 | 15 | 15 | 0.5769 | 0.7239 |
| test_6 | 15 | 15 | 11.7923 | 11.3203 | 0.4720 | 15 | 15 | 15 | 0.5764 | 0.7172 |
| test_7 | 15 | 15 | 10.3153 | 9.9812 | 0.3341 | 14 | 15 | 15 | 0.6018 | 0.7730 |
| test_8 | 15 | 15 | 10.8899 | 10.4632 | 0.4267 | 14 | 15 | 15 | 0.5809 | 0.7411 |
| test_9 | 15 | 15 | 11.8625 | 11.2300 | 0.6325 | 15 | 15 | 15 | 0.5843 | 0.7182 |
| test_10 | 15 | 15 | 11.1551 | 10.7641 | 0.3910 | 14 | 15 | 15 | 0.5629 | 0.7812 |

## Reading it

**The budget is equal in evaluations, not in compute, and the two are far apart here.** An infeasible layout is rejected by the placement test and costs about 0.36 ms; a feasible one runs inverse kinematics, the manipulability check and the swept-path collision test over four branches, and costs about 6.1 ms -- some seventeen times more. Repair works by converting the cheap evaluations into expensive ones, so the repair arm buys strictly more computation for the same nominal budget. That is why `cpu_time` is in the table and why it should be read alongside the cycle time rather than as a footnote: a win at equal evaluations is a weaker claim than a win at equal seconds, and only the `cpu_time` row says which one this is.

The feasible-share column is the mechanism and the cycle-time column is the outcome, and they are not the same question. Repair can lift the share of the population that scores at all -- that is close to arithmetic -- without the extra evaluations landing anywhere useful, because a layout projected onto the placement boundary is a layout with the machines pressed up against each other and against the robot's column, which is where reach and manipulability are hardest to satisfy. If the share rises and the cycle time does not, that gap is the finding, not a bug.

