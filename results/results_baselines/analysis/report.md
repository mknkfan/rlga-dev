# Revised experimental results

Split `test`: 10 instances x 10 independent seeds per instance x 5 algorithm variants = 500 runs.
All variants sharing an (instance, seed) pair start from the same initial population and the same random seed, so every comparison below is paired.


## 1. Solution quality and runtime (pooled over instances)

Median with interquartile range is the headline metric; the mean with its 95 % bootstrap confidence interval and the standard deviation are given alongside. The minimum column is shown for completeness only - it characterises the luckiest run, not expected performance.

| variant | fitness median [IQR] | fitness mean ± sd | fitness mean 95% CI | fitness min | time median [IQR] (s) | time mean ± sd (s) | evaluations median |
|---|---|---|---|---|---|---|---|
| GA no LS | 35.19 [33.00, 38.92] | 35.97 ± 4.23 | [35.16, 36.82] | 27.53 | 8.8 [8.2, 9.4] | 8.9 ± 1.0 | 60000 |
| Particle Swarm Optimisation | 38.34 [34.85, 42.52] | 39.15 ± 6.07 | [37.96, 40.33] | 26.15 | 3.3 [3.1, 3.5] | 3.3 ± 0.3 | 60000 |
| Artificial Bee Colony | 39.68 [37.18, 41.42] | 39.29 ± 3.14 | [38.68, 39.90] | 31.73 | 5.8 [5.6, 6.0] | 5.8 ± 0.4 | 60000 |
| Charged System Search | 58.01 [54.24, 61.47] | 58.27 ± 5.69 | [57.17, 59.39] | 45.30 | 17.3 [16.8, 17.9] | 17.3 ± 1.2 | 60000 |
| Differential Evolution | 65.18 [60.84, 68.38] | 64.46 ± 5.79 | [63.34, 65.56] | 50.64 | 3.7 [3.6, 3.9] | 3.8 ± 0.4 | 60000 |

## 2. Paired comparison against `ga_ls00` - solution quality

Difference = reference - variant on matched seeds, in fitness units, so a **negative** difference means the reference is lower (better). Two-sided Wilcoxon signed-rank test, Holm-corrected within this family; effect sizes are the matched-pairs rank-biserial correlation and Cliff's delta.

| variant | median diff | mean diff [95% CI] | reference lower / higher | p (Holm) | rank-biserial | Cliff's delta |
|---|---|---|---|---|---|---|
| Differential Evolution | -28.833 | -28.494 [-29.726, -27.194] | 100/0 | 0.0000 *** | -1.000 | -1.000 (large) |
| Charged System Search | -22.636 | -22.303 [-23.568, -21.057] | 100/0 | 0.0000 *** | -1.000 | -1.000 (large) |
| Artificial Bee Colony | -3.611 | -3.318 [-4.087, -2.513] | 82/18 | 0.0000 *** | -0.736 | -0.470 (medium) |
| Particle Swarm Optimisation | -3.147 | -3.176 [-4.494, -1.878] | 71/29 | 0.0000 *** | -0.483 | -0.308 (small) |

## 3. Paired comparison against `ga_ls00` - runtime

Difference = reference - variant on matched seeds, in seconds, so a **negative** difference means the reference is lower (cheaper). Two-sided Wilcoxon signed-rank test, Holm-corrected within this family; effect sizes are the matched-pairs rank-biserial correlation and Cliff's delta.

| variant | median diff | mean diff [95% CI] | reference lower / higher | p (Holm) | rank-biserial | Cliff's delta |
|---|---|---|---|---|---|---|
| Charged System Search | -8.339 | -8.371 [-8.612, -8.128] | 100/0 | 0.0000 *** | -1.000 | -1.000 (large) |
| Artificial Bee Colony | +2.914 | +3.042 [+2.862, +3.234] | 0/100 | 0.0000 *** | +1.000 | +0.999 (large) |
| Differential Evolution | +5.098 | +5.090 [+4.897, +5.294] | 0/100 | 0.0000 *** | +1.000 | +1.000 (large) |
| Particle Swarm Optimisation | +5.502 | +5.550 [+5.364, +5.747] | 0/100 | 0.0000 *** | +1.000 | +1.000 (large) |

## 4. Paired comparison against `ga_ls00` - computational effort

Difference = reference - variant on matched seeds, in objective-function evaluations, so a **negative** difference means the reference is lower (cheaper). Two-sided Wilcoxon signed-rank test, Holm-corrected within this family; effect sizes are the matched-pairs rank-biserial correlation and Cliff's delta.

| variant | median diff | mean diff [95% CI] | reference lower / higher | p (Holm) | rank-biserial | Cliff's delta |
|---|---|---|---|---|---|---|
| Differential Evolution | +0.000 | +0.000 [+0.000, +0.000] | 0/0 | 1.0000 n.s. | +0.000 | +0.000 (negligible) |
| Particle Swarm Optimisation | +0.000 | +0.000 [+0.000, +0.000] | 0/0 | 1.0000 n.s. | +0.000 | +0.000 (negligible) |
| Artificial Bee Colony | +0.000 | +0.000 [+0.000, +0.000] | 0/0 | 1.0000 n.s. | +0.000 | +0.000 (negligible) |
| Charged System Search | +0.000 | +0.000 [+0.000, +0.000] | 0/0 | 1.0000 n.s. | +0.000 | +0.000 (negligible) |

## 5. Equal-budget comparison

Equal generation counts are not an equal computational budget: variants with a higher local-search rate spend several times more objective-function evaluations per generation. Each (instance, seed) pair is therefore truncated at the largest budget every variant actually reached, and the best-so-far fitness at that point is compared. The last two columns give the effort needed to reach a quality level all variants attain.

| variant | fitness @ common evaluations (median) | fitness @ common wall-clock (median) | evaluations to common target (median) | seconds to common target (median) |
|---|---|---|---|---|
| GA no LS | 35.19 | 37.93 | 4100 | 0.5 |
| Particle Swarm Optimisation | 38.34 | 38.34 | 10000 | 0.5 |
| Artificial Bee Colony | 39.68 | 43.16 | 5400 | 0.5 |
| Charged System Search | 58.01 | 72.86 | 31500 | 9.2 |
| Differential Evolution | 65.18 | 67.11 | 52700 | 3.3 |

## 6. Per-instance medians (fitness)

| variant | test_1 | test_2 | test_3 | test_4 | test_5 | test_6 | test_7 | test_8 | test_9 | test_10 |
|---|---|---|---|---|---|---|---|---|---|---|
| GA no LS | 38.01 | 33.99 | 36.34 | 31.50 | 38.84 | 40.23 | 33.15 | 33.90 | 36.27 | 34.98 |
| Differential Evolution | 70.63 | 61.87 | 66.07 | 59.64 | 67.75 | 64.07 | 69.09 | 63.09 | 66.14 | 58.07 |
| Particle Swarm Optimisation | 38.96 | 36.15 | 35.79 | 31.30 | 46.58 | 39.03 | 38.25 | 38.47 | 40.62 | 36.64 |
| Artificial Bee Colony | 41.82 | 40.29 | 40.17 | 34.70 | 43.32 | 40.57 | 37.49 | 38.31 | 39.68 | 37.29 |
| Charged System Search | 58.18 | 54.84 | 56.65 | 51.57 | 60.31 | 58.17 | 59.60 | 59.50 | 58.98 | 57.57 |

## 7. Per-instance best fitness (minimum over seeds)

The best layout any seed found for each instance. This is a best-of-n statistic, so it rewards a lucky run and is not an estimate of expected performance - the medians above are. It is reported because a practitioner who can afford several restarts and keep the best layout cares about this column, and because it shows which variants can reach a good solution at all, as opposed to reaching one reliably.

| variant | test_1 | test_2 | test_3 | test_4 | test_5 | test_6 | test_7 | test_8 | test_9 | test_10 |
|---|---|---|---|---|---|---|---|---|---|---|
| GA no LS | 33.03 | 30.66 | 29.53 | 27.53 | 35.98 | 30.60 | 27.90 | 28.38 | 34.65 | 32.07 |
| Differential Evolution | 62.04 | 50.64 | 56.14 | 53.15 | 65.31 | 55.64 | 62.06 | 56.34 | 54.01 | 53.53 |
| Particle Swarm Optimisation | 32.70 | 27.95 | 31.63 | 26.15 | 38.78 | 34.92 | 32.11 | 34.64 | 34.42 | 31.45 |
| Artificial Bee Colony | 37.63 | 33.61 | 37.90 | 31.73 | 41.15 | 37.09 | 34.33 | 36.32 | 33.69 | 34.24 |
| Charged System Search | 51.36 | 48.54 | 50.99 | 45.30 | 56.49 | 51.51 | 53.34 | 54.65 | 50.74 | 49.90 |

The best layout found anywhere in the experiment scores 26.15 (Particle Swarm Optimisation on test_4). Counting instances by which variant reaches the lowest minimum, GA no LS leads with 5 of 10. Variant choice is not incidental to this column: on the median instance the gap between the best and the worst variant minimum is 85.8 % (range 60.3 - 122.4 %). That is the expected behaviour of a best-of-n statistic - it rewards the variant with the widest spread across seeds, not the most dependable one - and it is why the two tables should be read together rather than either one alone.

| variant | instances won on minimum | instances won on median |
|---|---|---|
| GA no LS | 5 (test_3, test_5, test_6, test_7, test_8) | 7 (test_1, test_2, test_5, test_7, test_8, test_9, test_10) |
| Particle Swarm Optimisation | 4 (test_1, test_2, test_4, test_10) | 3 (test_3, test_4, test_6) |
| Artificial Bee Colony | 1 (test_9) | 0 |

## 8. Where the RL-GA helps, and where it does not

| instance | median gain (%) | p | occupancy | feasibility rate | mean aspect ratio |
|---|---|---|---|---|---|
| test_7 | +52.02 | 0.002 | 0.147 | 0.0425 | 1.28 |
| test_4 | +47.19 | 0.002 | 0.115 | 0.0805 | 1.34 |
| test_8 | +46.27 | 0.002 | 0.149 | 0.0395 | 1.24 |
| test_1 | +46.19 | 0.002 | 0.157 | 0.0340 | 1.38 |
| test_9 | +45.16 | 0.002 | 0.144 | 0.0605 | 1.13 |
| test_2 | +45.07 | 0.002 | 0.134 | 0.0545 | 1.35 |
| test_3 | +45.00 | 0.002 | 0.143 | 0.0505 | 1.21 |
| test_5 | +42.67 | 0.002 | 0.161 | 0.0300 | 1.40 |
| test_10 | +39.76 | 0.002 | 0.137 | 0.0555 | 1.11 |
| test_6 | +37.21 | 0.002 | 0.141 | 0.0530 | 1.24 |

Pearson correlation between instance features and the relative gain:

| feature | correlation with gain (%) |
|---|---|
| mean_access_offset | +0.632 |
| machine_area_cv | +0.368 |
| mean_aspect_ratio | +0.296 |
| max_diagonal_over_extent | +0.284 |
| max_aspect_ratio | +0.192 |
| random_feasibility_rate | -0.041 |
| free_space_ratio | +0.016 |
| total_machine_area | -0.016 |
