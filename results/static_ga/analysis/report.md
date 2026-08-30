# Revised experimental results

Split `test`: 10 instances x 10 independent seeds per instance x 9 algorithm variants = 900 runs.
All variants sharing an (instance, seed) pair start from the same initial population and the same random seed, so every comparison below is paired.


## 1. Solution quality and runtime (pooled over instances)

Median with interquartile range is the headline metric; the mean with its 95 % bootstrap confidence interval and the standard deviation are given alongside. The minimum column is shown for completeness only - it characterises the luckiest run, not expected performance.

| variant | fitness median [IQR] | fitness mean ± sd | fitness mean 95% CI | fitness min | time median [IQR] (s) | time mean ± sd (s) | evaluations median |
|---|---|---|---|---|---|---|---|
| GA 100% LS [LS on offspring] | 34.63 [32.21, 37.25] | 35.08 ± 4.25 | [34.27, 35.92] | 25.74 | 16.1 [13.6, 18.3] | 16.4 ± 2.9 | 222000 |
| GA scheduled LS [LS on offspring] | 34.72 [32.48, 37.83] | 35.08 ± 3.74 | [34.35, 35.81] | 25.75 | 12.7 [11.1, 14.3] | 12.8 ± 2.2 | 146403 |
| GA 50% LS [LS on elites] | 34.89 [32.80, 38.03] | 35.66 ± 4.71 | [34.76, 36.61] | 26.33 | 9.8 [8.8, 11.2] | 10.0 ± 1.6 | 77997 |
| GA 50% LS [LS on offspring] | 35.05 [32.93, 37.89] | 35.34 ± 3.93 | [34.58, 36.11] | 27.05 | 12.3 [11.4, 13.7] | 12.6 ± 1.9 | 140973 |
| GA no LS | 35.19 [33.00, 38.92] | 35.97 ± 4.23 | [35.16, 36.82] | 27.53 | 8.3 [7.5, 9.1] | 8.4 ± 1.2 | 60000 |
| GA 100% LS [LS on elites] | 35.37 [33.03, 37.65] | 35.20 ± 4.22 | [34.38, 36.01] | 25.85 | 11.0 [9.9, 12.1] | 11.2 ± 1.8 | 96000 |
| GA scheduled LS [LS on elites] | 35.51 [33.34, 37.93] | 35.70 ± 3.75 | [34.98, 36.44] | 27.64 | 9.8 [8.9, 10.8] | 9.8 ± 1.4 | 79197 |
| GA 10% LS [LS on offspring] | 35.58 [33.37, 37.98] | 35.55 ± 3.76 | [34.83, 36.28] | 26.08 | 9.1 [8.5, 10.0] | 9.2 ± 1.3 | 76251 |
| GA 10% LS [LS on elites] | 35.91 [33.05, 38.73] | 35.97 ± 3.82 | [35.23, 36.71] | 27.74 | 8.6 [7.9, 9.2] | 8.6 ± 1.0 | 63585 |

## 2. Paired comparison against `ga_ls100_elites` - solution quality

Difference = reference - variant on matched seeds, in fitness units, so a **negative** difference means the reference is lower (better). Two-sided Wilcoxon signed-rank test, Holm-corrected within this family; effect sizes are the matched-pairs rank-biserial correlation and Cliff's delta.

| variant | median diff | mean diff [95% CI] | reference lower / higher | p (Holm) | rank-biserial | Cliff's delta |
|---|---|---|---|---|---|---|
| GA 10% LS [LS on offspring] | -0.535 | -0.347 [-1.180, +0.489] | 53/47 | 1.0000 n.s. | -0.084 | -0.028 (negligible) |
| GA scheduled LS [LS on elites] | -0.393 | -0.506 [-1.350, +0.344] | 53/47 | 1.0000 n.s. | -0.098 | -0.027 (negligible) |
| GA 50% LS [LS on elites] | -0.336 | -0.463 [-1.453, +0.540] | 53/47 | 1.0000 n.s. | -0.078 | +0.000 (negligible) |
| GA no LS | -0.215 | -0.771 [-1.720, +0.176] | 54/46 | 1.0000 n.s. | -0.162 | -0.063 (negligible) |
| GA 10% LS [LS on elites] | -0.203 | -0.774 [-1.595, +0.025] | 52/48 | 1.0000 n.s. | -0.157 | -0.072 (negligible) |
| GA 100% LS [LS on offspring] | +0.219 | +0.115 [-0.740, +0.986] | 49/51 | 1.0000 n.s. | +0.028 | +0.077 (negligible) |
| GA scheduled LS [LS on offspring] | +0.238 | +0.122 [-0.704, +0.973] | 46/54 | 1.0000 n.s. | +0.036 | +0.043 (negligible) |
| GA 50% LS [LS on offspring] | +0.315 | -0.147 [-0.985, +0.732] | 48/52 | 1.0000 n.s. | +0.023 | +0.016 (negligible) |

## 3. Paired comparison against `ga_ls100_elites` - runtime

Difference = reference - variant on matched seeds, in seconds, so a **negative** difference means the reference is lower (cheaper). Two-sided Wilcoxon signed-rank test, Holm-corrected within this family; effect sizes are the matched-pairs rank-biserial correlation and Cliff's delta.

| variant | median diff | mean diff [95% CI] | reference lower / higher | p (Holm) | rank-biserial | Cliff's delta |
|---|---|---|---|---|---|---|
| GA 100% LS [LS on offspring] | -5.101 | -5.193 [-5.786, -4.584] | 99/1 | 0.0000 *** | -0.999 | -0.906 (large) |
| GA scheduled LS [LS on offspring] | -1.707 | -1.605 [-2.041, -1.153] | 75/25 | 0.0000 *** | -0.663 | -0.427 (medium) |
| GA 50% LS [LS on offspring] | -1.519 | -1.452 [-1.871, -1.023] | 73/27 | 0.0000 *** | -0.671 | -0.447 (medium) |
| GA 50% LS [LS on elites] | +0.902 | +1.174 [+0.769, +1.588] | 31/69 | 0.0000 *** | +0.555 | +0.368 (medium) |
| GA scheduled LS [LS on elites] | +1.314 | +1.342 [+0.976, +1.715] | 28/72 | 0.0000 *** | +0.694 | +0.433 (medium) |
| GA 10% LS [LS on offspring] | +1.635 | +1.943 [+1.621, +2.278] | 6/94 | 0.0000 *** | +0.935 | +0.631 (large) |
| GA 10% LS [LS on elites] | +2.574 | +2.584 [+2.257, +2.924] | 3/97 | 0.0000 *** | +0.993 | +0.815 (large) |
| GA no LS | +2.723 | +2.732 [+2.376, +3.101] | 4/96 | 0.0000 *** | +0.977 | +0.815 (large) |

## 4. Paired comparison against `ga_ls100_elites` - computational effort

Difference = reference - variant on matched seeds, in objective-function evaluations, so a **negative** difference means the reference is lower (cheaper). Two-sided Wilcoxon signed-rank test, Holm-corrected within this family; effect sizes are the matched-pairs rank-biserial correlation and Cliff's delta.

| variant | median diff | mean diff [95% CI] | reference lower / higher | p (Holm) | rank-biserial | Cliff's delta |
|---|---|---|---|---|---|---|
| GA 100% LS [LS on offspring] | -126000.000 | -126000.000 [-126000.000, -126000.000] | 100/0 | 0.0000 *** | -1.000 | -1.000 (large) |
| GA scheduled LS [LS on offspring] | -50403.000 | -50424.780 [-50491.085, -50356.260] | 100/0 | 0.0000 *** | -1.000 | -1.000 (large) |
| GA 50% LS [LS on offspring] | -44973.000 | -44975.520 [-45071.641, -44876.939] | 100/0 | 0.0000 *** | -1.000 | -1.000 (large) |
| GA scheduled LS [LS on elites] | +16803.000 | +16802.220 [+16771.799, +16833.361] | 0/100 | 0.0000 *** | +1.000 | +1.000 (large) |
| GA 50% LS [LS on elites] | +18003.000 | +17988.000 [+17936.100, +18039.480] | 0/100 | 0.0000 *** | +1.000 | +1.000 (large) |
| GA 10% LS [LS on offspring] | +19749.000 | +19768.680 [+19706.877, +19830.780] | 0/100 | 0.0000 *** | +1.000 | +1.000 (large) |
| GA 10% LS [LS on elites] | +32415.000 | +32417.640 [+32391.659, +32443.080] | 0/100 | 0.0000 *** | +1.000 | +1.000 (large) |
| GA no LS | +36000.000 | +36000.000 [+36000.000, +36000.000] | 0/100 | 0.0000 *** | +1.000 | +1.000 (large) |

## 5. Equal-budget comparison

Equal generation counts are not an equal computational budget: variants with a higher local-search rate spend several times more objective-function evaluations per generation. Each (instance, seed) pair is therefore truncated at the largest budget every variant actually reached, and the best-so-far fitness at that point is compared. The last two columns give the effort needed to reach a quality level all variants attain.

| variant | fitness @ common evaluations (median) | fitness @ common wall-clock (median) | evaluations to common target (median) | seconds to common target (median) |
|---|---|---|---|---|
| GA no LS | 35.19 | 35.38 | 17300 | 2.2 |
| GA 50% LS [LS on elites] | 35.19 | 35.08 | 17899 | 2.1 |
| GA 10% LS [LS on offspring] | 35.66 | 35.64 | 19751 | 2.2 |
| GA 100% LS [LS on elites] | 35.73 | 35.63 | 18760 | 2.0 |
| GA scheduled LS [LS on offspring] | 35.81 | 35.36 | 19751 | 2.2 |
| GA scheduled LS [LS on elites] | 35.88 | 35.63 | 17090 | 2.2 |
| GA 10% LS [LS on elites] | 35.96 | 36.07 | 17090 | 2.2 |
| GA 50% LS [LS on offspring] | 36.35 | 35.43 | 32840 | 2.5 |
| GA 100% LS [LS on offspring] | 37.88 | 35.70 | 47190 | 2.9 |

## 6. Per-instance medians (fitness)

| variant | test_1 | test_2 | test_3 | test_4 | test_5 | test_6 | test_7 | test_8 | test_9 | test_10 |
|---|---|---|---|---|---|---|---|---|---|---|
| GA no LS | 38.01 | 33.99 | 36.34 | 31.50 | 38.84 | 40.23 | 33.15 | 33.90 | 36.27 | 34.98 |
| GA 10% LS [LS on offspring] | 37.80 | 35.66 | 35.79 | 30.29 | 38.90 | 36.66 | 35.08 | 36.07 | 37.23 | 33.45 |
| GA 50% LS [LS on offspring] | 39.66 | 32.36 | 33.98 | 30.69 | 39.73 | 35.75 | 34.80 | 33.37 | 38.02 | 33.36 |
| GA 100% LS [LS on offspring] | 37.19 | 34.08 | 34.48 | 29.55 | 40.77 | 36.03 | 33.51 | 33.53 | 36.00 | 33.02 |
| GA scheduled LS [LS on offspring] | 38.28 | 35.83 | 35.52 | 29.58 | 39.21 | 36.59 | 33.81 | 35.29 | 36.77 | 33.13 |
| GA 10% LS [LS on elites] | 39.56 | 35.50 | 34.74 | 30.66 | 39.83 | 36.76 | 33.46 | 34.45 | 36.69 | 35.94 |
| GA 50% LS [LS on elites] | 39.57 | 33.81 | 34.90 | 29.71 | 40.15 | 34.31 | 33.46 | 36.09 | 36.51 | 34.89 |
| GA 100% LS [LS on elites] | 38.27 | 34.85 | 37.41 | 29.15 | 40.25 | 37.20 | 34.36 | 34.47 | 35.83 | 34.19 |
| GA scheduled LS [LS on elites] | 40.26 | 34.92 | 34.69 | 31.43 | 39.78 | 36.03 | 33.43 | 34.30 | 36.31 | 37.47 |

## 7. Per-instance best fitness (minimum over seeds)

The best layout any seed found for each instance. This is a best-of-n statistic, so it rewards a lucky run and is not an estimate of expected performance - the medians above are. It is reported because a practitioner who can afford several restarts and keep the best layout cares about this column, and because it shows which variants can reach a good solution at all, as opposed to reaching one reliably.

| variant | test_1 | test_2 | test_3 | test_4 | test_5 | test_6 | test_7 | test_8 | test_9 | test_10 |
|---|---|---|---|---|---|---|---|---|---|---|
| GA no LS | 33.03 | 30.66 | 29.53 | 27.53 | 35.98 | 30.60 | 27.90 | 28.38 | 34.65 | 32.07 |
| GA 10% LS [LS on offspring] | 34.08 | 29.93 | 31.99 | 26.08 | 34.63 | 32.64 | 26.28 | 30.01 | 33.20 | 29.94 |
| GA 50% LS [LS on offspring] | 35.39 | 28.35 | 28.46 | 27.05 | 34.81 | 33.10 | 30.46 | 28.40 | 33.10 | 29.23 |
| GA 100% LS [LS on offspring] | 34.30 | 29.61 | 30.98 | 26.23 | 34.32 | 31.73 | 28.79 | 25.74 | 31.08 | 30.18 |
| GA scheduled LS [LS on offspring] | 34.23 | 30.62 | 31.93 | 25.75 | 33.88 | 31.52 | 26.23 | 31.62 | 33.23 | 30.88 |
| GA 10% LS [LS on elites] | 34.33 | 31.38 | 29.25 | 27.74 | 32.54 | 31.95 | 29.93 | 31.19 | 32.70 | 32.08 |
| GA 50% LS [LS on elites] | 31.78 | 28.40 | 28.98 | 27.61 | 33.95 | 31.67 | 26.33 | 28.42 | 32.58 | 32.27 |
| GA 100% LS [LS on elites] | 30.94 | 26.78 | 31.69 | 26.21 | 33.92 | 28.80 | 29.98 | 25.85 | 33.06 | 27.97 |
| GA scheduled LS [LS on elites] | 34.16 | 31.01 | 28.77 | 27.64 | 32.04 | 32.29 | 28.96 | 31.09 | 31.92 | 30.94 |

The best layout found anywhere in the experiment scores 25.74 (GA 100% LS [LS on offspring] on test_8). Counting instances by which variant reaches the lowest minimum, GA 100% LS [LS on elites] leads with 4 of 10. Variant choice is not incidental to this column: on the median instance the gap between the best and the worst variant minimum is 14.6 % (range 7.7 - 22.8 %). That is the expected behaviour of a best-of-n statistic - it rewards the variant with the widest spread across seeds, not the most dependable one - and it is why the two tables should be read together rather than either one alone.

| variant | instances won on minimum | instances won on median |
|---|---|---|
| GA 100% LS [LS on elites] | 4 (test_1, test_2, test_6, test_10) | 2 (test_4, test_9) |
| GA 100% LS [LS on offspring] | 2 (test_8, test_9) | 2 (test_1, test_10) |
| GA scheduled LS [LS on offspring] | 2 (test_4, test_7) | 0 |
| GA 50% LS [LS on offspring] | 1 (test_3) | 3 (test_2, test_3, test_8) |
| GA scheduled LS [LS on elites] | 1 (test_5) | 0 |
| GA no LS | 0 | 2 (test_5, test_7) |
| GA 50% LS [LS on elites] | 0 | 1 (test_6) |

## 8. Where the RL-GA helps, and where it does not

| instance | median gain (%) | p | occupancy | feasibility rate | mean aspect ratio |
|---|---|---|---|---|---|
| test_6 | +7.53 | 0.432 | 0.141 | 0.0530 | 1.24 |
| test_4 | +7.45 | 0.322 | 0.115 | 0.0805 | 1.34 |
| test_10 | +2.25 | 0.492 | 0.137 | 0.0555 | 1.11 |
| test_9 | +1.21 | 0.375 | 0.144 | 0.0605 | 1.13 |
| test_1 | -0.70 | 0.625 | 0.157 | 0.0340 | 1.38 |
| test_8 | -1.69 | 0.492 | 0.149 | 0.0395 | 1.24 |
| test_2 | -2.56 | 1.000 | 0.134 | 0.0545 | 1.35 |
| test_3 | -2.96 | 0.492 | 0.143 | 0.0505 | 1.21 |
| test_7 | -3.64 | 0.695 | 0.147 | 0.0425 | 1.28 |
| test_5 | -3.64 | 0.846 | 0.161 | 0.0300 | 1.40 |

Pearson correlation between instance features and the relative gain:

| feature | correlation with gain (%) |
|---|---|
| max_diagonal_over_extent | -0.720 |
| random_feasibility_rate | +0.699 |
| free_space_ratio | +0.648 |
| occupancy | -0.648 |
| total_machine_area | -0.648 |
| mean_machine_area | -0.648 |
| machine_area_cv | -0.570 |
| mean_access_offset | -0.330 |
