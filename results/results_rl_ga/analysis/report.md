# Revised experimental results

Split `test`: 10 instances x 10 independent seeds per instance x 11 algorithm variants = 1100 runs.
All variants sharing an (instance, seed) pair start from the same initial population and the same random seed, so every comparison below is paired.


## 1. Solution quality and runtime (pooled over instances)

Median with interquartile range is the headline metric; the mean with its 95 % bootstrap confidence interval and the standard deviation are given alongside. The minimum column is shown for completeness only - it characterises the luckiest run, not expected performance.

| variant | fitness median [IQR] | fitness mean ± sd | fitness mean 95% CI | fitness min | time median [IQR] (s) | time mean ± sd (s) | evaluations median |
|---|---|---|---|---|---|---|---|
| GA 100% LS [LS on offspring] | 34.63 [32.21, 37.25] | 35.08 ± 4.25 | [34.27, 35.92] | 25.74 | 15.4 [14.1, 17.0] | 15.5 ± 2.3 | 222000 |
| GA scheduled LS [LS on offspring] | 34.72 [32.48, 37.83] | 35.08 ± 3.74 | [34.35, 35.81] | 25.75 | 12.1 [10.7, 13.4] | 12.2 ± 1.8 | 146403 |
| RL-GA [LS on offspring] | 34.85 [32.52, 38.02] | 35.42 ± 4.82 | [34.48, 36.37] | 23.58 | 11.8 [9.7, 13.2] | 11.8 ± 2.5 | 128628 |
| GA 50% LS [LS on elites] | 34.89 [32.80, 38.03] | 35.66 ± 4.71 | [34.76, 36.61] | 26.33 | 9.5 [8.8, 10.5] | 9.6 ± 1.3 | 77997 |
| GA 50% LS [LS on offspring] | 35.05 [32.93, 37.89] | 35.34 ± 3.93 | [34.58, 36.11] | 27.05 | 12.0 [10.8, 13.3] | 12.1 ± 1.7 | 140973 |
| GA no LS | 35.19 [33.00, 38.92] | 35.97 ± 4.23 | [35.16, 36.82] | 27.53 | 8.1 [7.6, 8.8] | 8.2 ± 0.9 | 60000 |
| RL-GA [LS on elites] | 35.31 [32.51, 38.80] | 35.58 ± 4.23 | [34.75, 36.39] | 27.78 | 10.0 [9.0, 11.4] | 10.2 ± 1.6 | 86160 |
| GA 100% LS [LS on elites] | 35.37 [33.03, 37.65] | 35.20 ± 4.22 | [34.38, 36.01] | 25.85 | 10.5 [9.7, 11.7] | 10.7 ± 1.5 | 96000 |
| GA scheduled LS [LS on elites] | 35.51 [33.34, 37.93] | 35.70 ± 3.75 | [34.98, 36.44] | 27.64 | 9.5 [8.6, 10.4] | 9.6 ± 1.3 | 79197 |
| GA 10% LS [LS on offspring] | 35.58 [33.37, 37.98] | 35.55 ± 3.76 | [34.83, 36.28] | 26.08 | 8.9 [8.0, 9.4] | 8.9 ± 1.0 | 76251 |
| GA 10% LS [LS on elites] | 35.91 [33.05, 38.73] | 35.97 ± 3.82 | [35.23, 36.71] | 27.74 | 8.2 [7.8, 9.0] | 8.4 ± 1.0 | 63585 |

## 2. Paired comparison against `rlga_elites` - solution quality

Difference = reference - variant on matched seeds, in fitness units, so a **negative** difference means the reference is lower (better). Two-sided Wilcoxon signed-rank test, Holm-corrected within this family; effect sizes are the matched-pairs rank-biserial correlation and Cliff's delta.

| variant | median diff | mean diff [95% CI] | reference lower / higher | p (Holm) | rank-biserial | Cliff's delta |
|---|---|---|---|---|---|---|
| GA 10% LS [LS on elites] | -0.789 | -0.397 [-1.231, +0.484] | 58/42 | 1.0000 n.s. | -0.167 | -0.059 (negligible) |
| GA no LS | -0.474 | -0.393 [-1.277, +0.503] | 56/44 | 1.0000 n.s. | -0.105 | -0.058 (negligible) |
| GA scheduled LS [LS on elites] | -0.357 | -0.128 [-0.973, +0.748] | 53/47 | 1.0000 n.s. | -0.089 | -0.018 (negligible) |
| GA 10% LS [LS on offspring] | -0.306 | +0.030 [-0.839, +0.935] | 52/48 | 1.0000 n.s. | -0.015 | -0.019 (negligible) |
| GA 50% LS [LS on elites] | -0.093 | -0.086 [-1.077, +0.910] | 50/50 | 1.0000 n.s. | -0.017 | +0.010 (negligible) |
| GA scheduled LS [LS on offspring] | +0.172 | +0.499 [-0.378, +1.419] | 47/53 | 1.0000 n.s. | +0.086 | +0.050 (negligible) |
| GA 100% LS [LS on offspring] | +0.296 | +0.492 [-0.481, +1.472] | 44/56 | 1.0000 n.s. | +0.149 | +0.073 (negligible) |
| GA 50% LS [LS on offspring] | +0.392 | +0.231 [-0.693, +1.181] | 47/53 | 1.0000 n.s. | +0.039 | +0.023 (negligible) |
| GA 100% LS [LS on elites] | +0.493 | +0.377 [-0.597, +1.341] | 45/55 | 1.0000 n.s. | +0.088 | +0.015 (negligible) |
| RL-GA [LS on offspring] | +0.530 | +0.153 [-0.857, +1.186] | 46/54 | 1.0000 n.s. | +0.048 | +0.028 (negligible) |

## 3. Paired comparison against `rlga_elites` - runtime

Difference = reference - variant on matched seeds, in seconds, so a **negative** difference means the reference is lower (cheaper). Two-sided Wilcoxon signed-rank test, Holm-corrected within this family; effect sizes are the matched-pairs rank-biserial correlation and Cliff's delta.

| variant | median diff | mean diff [95% CI] | reference lower / higher | p (Holm) | rank-biserial | Cliff's delta |
|---|---|---|---|---|---|---|
| GA 100% LS [LS on offspring] | -4.970 | -5.336 [-5.847, -4.814] | 97/3 | 0.0000 *** | -0.997 | -0.945 (large) |
| GA scheduled LS [LS on offspring] | -1.991 | -2.062 [-2.498, -1.618] | 79/21 | 0.0000 *** | -0.797 | -0.589 (large) |
| GA 50% LS [LS on offspring] | -1.933 | -1.916 [-2.321, -1.496] | 84/16 | 0.0000 *** | -0.797 | -0.582 (large) |
| RL-GA [LS on offspring] | -1.388 | -1.675 [-2.262, -1.092] | 72/28 | 0.0000 *** | -0.583 | -0.405 (medium) |
| GA 100% LS [LS on elites] | -0.419 | -0.532 [-0.919, -0.130] | 62/38 | 0.0111 * | -0.302 | -0.215 (small) |
| GA scheduled LS [LS on elites] | +0.369 | +0.588 [+0.243, +0.935] | 36/64 | 0.0111 * | +0.335 | +0.222 (small) |
| GA 50% LS [LS on elites] | +0.623 | +0.541 [+0.175, +0.906] | 39/61 | 0.0111 * | +0.326 | +0.200 (small) |
| GA 10% LS [LS on offspring] | +1.255 | +1.267 [+0.934, +1.614] | 26/74 | 0.0000 *** | +0.680 | +0.483 (large) |
| GA 10% LS [LS on elites] | +1.517 | +1.720 [+1.404, +2.042] | 13/87 | 0.0000 *** | +0.908 | +0.642 (large) |
| GA no LS | +2.089 | +1.996 [+1.679, +2.315] | 10/90 | 0.0000 *** | +0.934 | +0.736 (large) |

## 4. Paired comparison against `rlga_elites` - computational effort

Difference = reference - variant on matched seeds, in objective-function evaluations, so a **negative** difference means the reference is lower (cheaper). Two-sided Wilcoxon signed-rank test, Holm-corrected within this family; effect sizes are the matched-pairs rank-biserial correlation and Cliff's delta.

| variant | median diff | mean diff [95% CI] | reference lower / higher | p (Holm) | rank-biserial | Cliff's delta |
|---|---|---|---|---|---|---|
| GA 100% LS [LS on offspring] | -135840.000 | -135940.500 [-136846.563, -135044.968] | 100/0 | 0.0000 *** | -1.000 | -1.000 (large) |
| GA scheduled LS [LS on offspring] | -60351.000 | -60365.280 [-61264.201, -59478.770] | 100/0 | 0.0000 *** | -1.000 | -1.000 (large) |
| GA 50% LS [LS on offspring] | -54762.000 | -54916.020 [-55824.974, -54015.115] | 100/0 | 0.0000 *** | -1.000 | -1.000 (large) |
| RL-GA [LS on offspring] | -42384.000 | -42935.640 [-49495.707, -36244.949] | 85/15 | 0.0000 *** | -0.926 | -0.759 (large) |
| GA 100% LS [LS on elites] | -9840.000 | -9940.500 [-10846.563, -9044.969] | 100/0 | 0.0000 *** | -1.000 | -1.000 (large) |
| GA scheduled LS [LS on elites] | +6933.000 | +6861.720 [+5964.420, +7750.861] | 11/89 | 0.0000 *** | +0.952 | +0.793 (large) |
| GA 50% LS [LS on elites] | +8160.000 | +8047.500 [+7133.991, +8945.709] | 7/93 | 0.0000 *** | +0.974 | +0.851 (large) |
| GA 10% LS [LS on offspring] | +9693.000 | +9828.180 [+8914.136, +10724.344] | 3/97 | 0.0000 *** | +0.994 | +0.939 (large) |
| GA 10% LS [LS on elites] | +22527.000 | +22477.140 [+21573.840, +23373.278] | 0/100 | 0.0000 *** | +1.000 | +1.000 (large) |
| GA no LS | +26160.000 | +26059.500 [+25153.437, +26955.032] | 0/100 | 0.0000 *** | +1.000 | +1.000 (large) |

## 5. Paired comparison against `rlga_offspring` - solution quality

Difference = reference - variant on matched seeds, in fitness units, so a **negative** difference means the reference is lower (better). Two-sided Wilcoxon signed-rank test, Holm-corrected within this family; effect sizes are the matched-pairs rank-biserial correlation and Cliff's delta.

| variant | median diff | mean diff [95% CI] | reference lower / higher | p (Holm) | rank-biserial | Cliff's delta |
|---|---|---|---|---|---|---|
| GA 10% LS [LS on elites] | -1.384 | -0.550 [-1.396, +0.330] | 58/42 | 1.0000 n.s. | -0.174 | -0.085 (negligible) |
| GA scheduled LS [LS on elites] | -1.125 | -0.281 [-1.111, +0.588] | 63/37 | 1.0000 n.s. | -0.136 | -0.049 (negligible) |
| GA no LS | -0.647 | -0.546 [-1.533, +0.455] | 53/47 | 1.0000 n.s. | -0.133 | -0.073 (negligible) |
| RL-GA [LS on elites] | -0.530 | -0.153 [-1.186, +0.857] | 54/46 | 1.0000 n.s. | -0.048 | -0.028 (negligible) |
| GA 100% LS [LS on elites] | -0.359 | +0.225 [-0.766, +1.211] | 54/46 | 1.0000 n.s. | +0.035 | -0.011 (negligible) |
| GA 100% LS [LS on offspring] | -0.338 | +0.339 [-0.676, +1.401] | 53/47 | 1.0000 n.s. | +0.017 | +0.042 (negligible) |
| GA 50% LS [LS on elites] | -0.099 | -0.239 [-1.304, +0.806] | 52/48 | 1.0000 n.s. | -0.059 | -0.022 (negligible) |
| GA 10% LS [LS on offspring] | -0.034 | -0.122 [-1.091, +0.860] | 51/49 | 1.0000 n.s. | -0.073 | -0.048 (negligible) |
| GA 50% LS [LS on offspring] | +0.034 | +0.078 [-0.819, +1.010] | 49/51 | 1.0000 n.s. | -0.024 | -0.004 (negligible) |
| GA scheduled LS [LS on offspring] | +0.421 | +0.346 [-0.596, +1.310] | 45/55 | 1.0000 n.s. | +0.045 | +0.019 (negligible) |

## 6. Paired comparison against `rlga_offspring` - runtime

Difference = reference - variant on matched seeds, in seconds, so a **negative** difference means the reference is lower (cheaper). Two-sided Wilcoxon signed-rank test, Holm-corrected within this family; effect sizes are the matched-pairs rank-biserial correlation and Cliff's delta.

| variant | median diff | mean diff [95% CI] | reference lower / higher | p (Holm) | rank-biserial | Cliff's delta |
|---|---|---|---|---|---|---|
| GA 100% LS [LS on offspring] | -4.224 | -3.661 [-4.305, -2.998] | 84/16 | 0.0000 *** | -0.867 | -0.703 (large) |
| GA 50% LS [LS on offspring] | -0.748 | -0.241 [-0.824, +0.354] | 60/40 | 0.2529 n.s. | -0.173 | -0.108 (negligible) |
| GA scheduled LS [LS on offspring] | -0.410 | -0.387 [-1.000, +0.227] | 55/45 | 0.2529 n.s. | -0.176 | -0.137 (negligible) |
| GA 100% LS [LS on elites] | +0.947 | +1.142 [+0.592, +1.690] | 39/61 | 0.0011 ** | +0.411 | +0.267 (small) |
| RL-GA [LS on elites] | +1.388 | +1.675 [+1.092, +2.262] | 28/72 | 0.0000 *** | +0.583 | +0.405 (medium) |
| GA 50% LS [LS on elites] | +1.715 | +2.215 [+1.683, +2.752] | 25/75 | 0.0000 *** | +0.746 | +0.546 (large) |
| GA scheduled LS [LS on elites] | +2.033 | +2.263 [+1.728, +2.820] | 23/77 | 0.0000 *** | +0.764 | +0.564 (large) |
| GA 10% LS [LS on offspring] | +2.511 | +2.942 [+2.408, +3.482] | 9/91 | 0.0000 *** | +0.924 | +0.732 (large) |
| GA 10% LS [LS on elites] | +3.150 | +3.395 [+2.886, +3.914] | 7/93 | 0.0000 *** | +0.967 | +0.824 (large) |
| GA no LS | +3.572 | +3.671 [+3.138, +4.218] | 4/96 | 0.0000 *** | +0.985 | +0.870 (large) |

## 7. Paired comparison against `rlga_offspring` - computational effort

Difference = reference - variant on matched seeds, in objective-function evaluations, so a **negative** difference means the reference is lower (cheaper). Two-sided Wilcoxon signed-rank test, Holm-corrected within this family; effect sizes are the matched-pairs rank-biserial correlation and Cliff's delta.

| variant | median diff | mean diff [95% CI] | reference lower / higher | p (Holm) | rank-biserial | Cliff's delta |
|---|---|---|---|---|---|---|
| GA 100% LS [LS on offspring] | -93372.000 | -93004.860 [-99355.000, -86699.360] | 100/0 | 0.0000 *** | -1.000 | -1.000 (large) |
| GA scheduled LS [LS on offspring] | -17706.000 | -17429.640 [-23795.631, -11117.849] | 68/32 | 0.0000 *** | -0.546 | -0.360 (medium) |
| GA 50% LS [LS on offspring] | -11907.000 | -11980.380 [-18371.101, -5675.936] | 63/37 | 0.0008 *** | -0.386 | -0.260 (small) |
| GA 100% LS [LS on elites] | +32628.000 | +32995.140 [+26644.999, +39300.641] | 17/83 | 0.0000 *** | +0.852 | +0.660 (large) |
| RL-GA [LS on elites] | +42384.000 | +42935.640 [+36244.949, +49495.707] | 15/85 | 0.0000 *** | +0.926 | +0.759 (large) |
| GA scheduled LS [LS on elites] | +49413.000 | +49797.360 [+43462.312, +56095.882] | 8/92 | 0.0000 *** | +0.975 | +0.834 (large) |
| GA 50% LS [LS on elites] | +50724.000 | +50983.140 [+44636.208, +57280.971] | 7/93 | 0.0000 *** | +0.979 | +0.860 (large) |
| GA 10% LS [LS on offspring] | +52602.000 | +52763.820 [+46412.935, +59075.480] | 6/94 | 0.0000 *** | +0.986 | +0.879 (large) |
| GA 10% LS [LS on elites] | +64977.000 | +65412.780 [+59070.678, +71721.048] | 0/100 | 0.0000 *** | +1.000 | +1.000 (large) |
| GA no LS | +68628.000 | +68995.140 [+62644.999, +75300.640] | 0/100 | 0.0000 *** | +1.000 | +1.000 (large) |

## 8. Equal-budget comparison

Equal generation counts are not an equal computational budget: variants with a higher local-search rate spend several times more objective-function evaluations per generation. Each (instance, seed) pair is therefore truncated at the largest budget every variant actually reached, and the best-so-far fitness at that point is compared. The last two columns give the effort needed to reach a quality level all variants attain.

| variant | fitness @ common evaluations (median) | fitness @ common wall-clock (median) | evaluations to common target (median) | seconds to common target (median) |
|---|---|---|---|---|
| GA no LS | 35.19 | 35.26 | 15200 | 1.9 |
| GA 50% LS [LS on elites] | 35.19 | 34.91 | 17163 | 1.9 |
| GA 10% LS [LS on offspring] | 35.66 | 35.63 | 17877 | 2.1 |
| RL-GA [LS on elites] | 35.67 | 35.60 | 17075 | 2.0 |
| GA 100% LS [LS on elites] | 35.73 | 35.63 | 17480 | 1.7 |
| GA scheduled LS [LS on offspring] | 35.81 | 35.36 | 17877 | 2.1 |
| GA scheduled LS [LS on elites] | 35.88 | 35.63 | 14740 | 1.9 |
| GA 10% LS [LS on elites] | 35.96 | 35.98 | 14740 | 2.0 |
| RL-GA [LS on offspring] | 36.15 | 35.20 | 29862 | 2.6 |
| GA 50% LS [LS on offspring] | 36.35 | 35.46 | 30233 | 2.4 |
| GA 100% LS [LS on offspring] | 37.88 | 35.67 | 43860 | 2.8 |

## 9. Per-instance medians (fitness)

| variant | test_1 | test_2 | test_3 | test_4 | test_5 | test_6 | test_7 | test_8 | test_9 | test_10 |
|---|---|---|---|---|---|---|---|---|---|---|
| GA no LS | 38.01 | 33.99 | 36.34 | 31.50 | 38.84 | 40.23 | 33.15 | 33.90 | 36.27 | 34.98 |
| GA 10% LS [LS on offspring] | 37.80 | 35.66 | 35.79 | 30.29 | 38.90 | 36.66 | 35.08 | 36.07 | 37.23 | 33.45 |
| GA 50% LS [LS on offspring] | 39.66 | 32.36 | 33.98 | 30.69 | 39.73 | 35.75 | 34.80 | 33.37 | 38.02 | 33.36 |
| GA 100% LS [LS on offspring] | 37.19 | 34.08 | 34.48 | 29.55 | 40.77 | 36.03 | 33.51 | 33.53 | 36.00 | 33.02 |
| GA scheduled LS [LS on offspring] | 38.28 | 35.83 | 35.52 | 29.58 | 39.21 | 36.59 | 33.81 | 35.29 | 36.77 | 33.13 |
| RL-GA [LS on offspring] | 37.95 | 33.09 | 34.25 | 30.46 | 40.03 | 35.19 | 32.42 | 33.50 | 36.44 | 36.57 |
| GA 10% LS [LS on elites] | 39.56 | 35.50 | 34.74 | 30.66 | 39.83 | 36.76 | 33.46 | 34.45 | 36.69 | 35.94 |
| GA 50% LS [LS on elites] | 39.57 | 33.81 | 34.90 | 29.71 | 40.15 | 34.31 | 33.46 | 36.09 | 36.51 | 34.89 |
| GA 100% LS [LS on elites] | 38.27 | 34.85 | 37.41 | 29.15 | 40.25 | 37.20 | 34.36 | 34.47 | 35.83 | 34.19 |
| GA scheduled LS [LS on elites] | 40.26 | 34.92 | 34.69 | 31.43 | 39.78 | 36.03 | 33.43 | 34.30 | 36.31 | 37.47 |
| RL-GA [LS on elites] | 35.68 | 32.64 | 38.10 | 30.31 | 40.38 | 35.68 | 35.40 | 34.88 | 37.25 | 35.01 |

## 10. Per-instance best fitness (minimum over seeds)

The best layout any seed found for each instance. This is a best-of-n statistic, so it rewards a lucky run and is not an estimate of expected performance - the medians above are. It is reported because a practitioner who can afford several restarts and keep the best layout cares about this column, and because it shows which variants can reach a good solution at all, as opposed to reaching one reliably.

| variant | test_1 | test_2 | test_3 | test_4 | test_5 | test_6 | test_7 | test_8 | test_9 | test_10 |
|---|---|---|---|---|---|---|---|---|---|---|
| GA no LS | 33.03 | 30.66 | 29.53 | 27.53 | 35.98 | 30.60 | 27.90 | 28.38 | 34.65 | 32.07 |
| GA 10% LS [LS on offspring] | 34.08 | 29.93 | 31.99 | 26.08 | 34.63 | 32.64 | 26.28 | 30.01 | 33.20 | 29.94 |
| GA 50% LS [LS on offspring] | 35.39 | 28.35 | 28.46 | 27.05 | 34.81 | 33.10 | 30.46 | 28.40 | 33.10 | 29.23 |
| GA 100% LS [LS on offspring] | 34.30 | 29.61 | 30.98 | 26.23 | 34.32 | 31.73 | 28.79 | 25.74 | 31.08 | 30.18 |
| GA scheduled LS [LS on offspring] | 34.23 | 30.62 | 31.93 | 25.75 | 33.88 | 31.52 | 26.23 | 31.62 | 33.23 | 30.88 |
| RL-GA [LS on offspring] | 33.36 | 29.91 | 27.56 | 23.58 | 37.60 | 31.93 | 26.50 | 28.88 | 28.88 | 30.87 |
| GA 10% LS [LS on elites] | 34.33 | 31.38 | 29.25 | 27.74 | 32.54 | 31.95 | 29.93 | 31.19 | 32.70 | 32.08 |
| GA 50% LS [LS on elites] | 31.78 | 28.40 | 28.98 | 27.61 | 33.95 | 31.67 | 26.33 | 28.42 | 32.58 | 32.27 |
| GA 100% LS [LS on elites] | 30.94 | 26.78 | 31.69 | 26.21 | 33.92 | 28.80 | 29.98 | 25.85 | 33.06 | 27.97 |
| GA scheduled LS [LS on elites] | 34.16 | 31.01 | 28.77 | 27.64 | 32.04 | 32.29 | 28.96 | 31.09 | 31.92 | 30.94 |
| RL-GA [LS on elites] | 32.77 | 28.93 | 28.86 | 27.78 | 33.21 | 31.98 | 29.58 | 29.58 | 33.48 | 31.53 |

The best layout found anywhere in the experiment scores 23.58 (RL-GA [LS on offspring] on test_4). Counting instances by which variant reaches the lowest minimum, GA 100% LS [LS on elites] leads with 4 of 10. Variant choice is not incidental to this column: on the median instance the gap between the best and the worst variant minimum is 16.6 % (range 14.4 - 22.8 %). That is the expected behaviour of a best-of-n statistic - it rewards the variant with the widest spread across seeds, not the most dependable one - and it is why the two tables should be read together rather than either one alone.

| variant | instances won on minimum | instances won on median |
|---|---|---|
| GA 100% LS [LS on elites] | 4 (test_1, test_2, test_6, test_10) | 2 (test_4, test_9) |
| RL-GA [LS on offspring] | 3 (test_3, test_4, test_9) | 1 (test_7) |
| GA 100% LS [LS on offspring] | 1 (test_8) | 1 (test_10) |
| GA scheduled LS [LS on elites] | 1 (test_5) | 0 |
| GA scheduled LS [LS on offspring] | 1 (test_7) | 0 |
| GA no LS | 0 | 1 (test_5) |
| GA 50% LS [LS on elites] | 0 | 1 (test_6) |
| GA 50% LS [LS on offspring] | 0 | 3 (test_2, test_3, test_8) |
| RL-GA [LS on elites] | 0 | 1 (test_1) |

## 11. Where the RL-GA helps, and where it does not

| instance | median gain (%) | p | occupancy | feasibility rate | mean aspect ratio |
|---|---|---|---|---|---|
| test_2 | +8.47 | 0.232 | 0.134 | 0.0545 | 1.35 |
| test_1 | +5.59 | 0.557 | 0.157 | 0.0340 | 1.38 |
| test_8 | +3.30 | 0.322 | 0.149 | 0.0395 | 1.24 |
| test_6 | +2.67 | 0.695 | 0.141 | 0.0530 | 1.24 |
| test_9 | -0.06 | 1.000 | 0.144 | 0.0605 | 1.13 |
| test_4 | -0.09 | 1.000 | 0.115 | 0.0805 | 1.34 |
| test_7 | -0.90 | 0.432 | 0.147 | 0.0425 | 1.28 |
| test_5 | -3.81 | 0.557 | 0.161 | 0.0300 | 1.40 |
| test_10 | -4.65 | 0.193 | 0.137 | 0.0555 | 1.11 |
| test_3 | -6.47 | 0.846 | 0.143 | 0.0505 | 1.21 |

Pearson correlation between instance features and the relative gain:

| feature | correlation with gain (%) |
|---|---|
| max_aspect_ratio | +0.644 |
| mean_aspect_ratio | +0.416 |
| max_diagonal_over_extent | +0.379 |
| machine_area_cv | +0.364 |
| mean_access_offset | +0.093 |
| occupancy | -0.053 |
| total_machine_area | -0.053 |
| mean_machine_area | -0.053 |
