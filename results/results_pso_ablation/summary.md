# Cognitive x social coefficients - PSO

3 coefficient settings, 20 runs on each of 10 'test' instances (600 runs recorded).

Shared parameters: population 200, 300 iterations, inertia w=0.7298, velocity clamp 0.5 x range, budget 60000 evaluations.

`c1` is the cognitive coefficient (pull towards the particle's own best) and `c2` the social one (pull towards the swarm's best). The settings share `c1 + c2`, so the total acceleration is fixed and only its split between the two attractors changes.

**Fitness is minimised** (lower is better), and infeasible runs are excluded from the fitness statistics; `n_feasible` reports how many runs in each cell were feasible. Evaluation counts and times cover every run.

`mean_rank` ranks the settings against each other within every (instance, seed) pair - all of which start from the same initial population - and averages those ranks. It is the scale-free comparison; the pooled fitness columns mix instances whose fitness scales differ.

Lowest mean fitness: **c1=2, c2=2** (mean 52.4089, median 51.4612).
Best mean rank: **c1=2, c2=2** (1.55).


## Per setting

| c1 | c2 | n | feasible | fitness mean | fitness sd | fitness median | fitness min | evals mean | wall mean (s) | CPU mean (s) | mean rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 200 | 200/200 | 55.2075 | 6.1724 | 54.4399 | 41.3590 | 60000 | 2.68 | 2.64 | 2.03 |
| 2 | 2 | 200 | 200/200 | 52.4089 | 5.4711 | 51.4612 | 39.8802 | 60000 | 2.69 | 2.66 | 1.55 |
| 3 | 1 | 200 | 200/200 | 57.6018 | 5.4772 | 57.0569 | 47.0901 | 60000 | 2.72 | 2.69 | 2.42 |

## Pairwise comparisons

Two-sided Wilcoxon signed-rank on the paired (instance, seed) runs, Holm-Bonferroni adjusted over the pairs below. A **negative** mean difference means the left setting is better. `seed w/l` counts the paired runs the left setting won and lost; `instance w/l` counts the instances where its mean fitness was lower - the count that does not pool different fitness scales.

| comparison | n pairs | mean diff | 95% CI | median diff | p (Holm) | | Cliff's delta | seed w/l | instance w/l |
|---|---|---|---|---|---|---|---|---|---|
| c1=1, c2=3 vs c1=2, c2=2 | 200 | 2.7987 | [1.7059, 3.9012] | 2.1466 | 1.39e-05 | *** | 0.272 (small) | 71/129 | 1/9 |
| c1=1, c2=3 vs c1=3, c2=1 | 200 | -2.3943 | [-3.4398, -1.3701] | -2.9665 | 2.43e-05 | *** | -0.246 (small) | 123/77 | 9/1 |
| c1=2, c2=2 vs c1=3, c2=1 | 200 | -5.1929 | [-6.1161, -4.2378] | -5.2620 | 1.99e-18 | *** | -0.504 (large) | 161/39 | 10/0 |

## Mean fitness per instance

| instance | c1=1, c2=3 | c1=2, c2=2 | c1=3, c2=1 |
|---|---|---|---|
| test_1 | 57.887 | 53.065 | 59.758 |
| test_2 | 53.052 | 51.499 | 55.880 |
| test_3 | 53.133 | 52.087 | 54.643 |
| test_4 | 50.227 | 50.495 | 54.449 |
| test_5 | 60.257 | 57.264 | 64.052 |
| test_6 | 55.912 | 50.328 | 58.195 |
| test_7 | 56.886 | 52.764 | 59.841 |
| test_8 | 56.555 | 53.227 | 55.433 |
| test_9 | 54.350 | 52.100 | 55.668 |
| test_10 | 53.817 | 51.260 | 58.100 |

## Median fitness per instance

| instance | c1=1, c2=3 | c1=2, c2=2 | c1=3, c2=1 |
|---|---|---|---|
| test_1 | 56.528 | 52.799 | 59.556 |
| test_2 | 51.739 | 51.385 | 56.912 |
| test_3 | 53.058 | 50.757 | 54.446 |
| test_4 | 49.251 | 50.362 | 53.028 |
| test_5 | 59.567 | 55.441 | 63.939 |
| test_6 | 55.734 | 49.187 | 58.476 |
| test_7 | 55.402 | 51.425 | 59.624 |
| test_8 | 57.639 | 52.893 | 54.914 |
| test_9 | 52.663 | 51.209 | 55.451 |
| test_10 | 53.357 | 50.575 | 58.516 |

## Feasible runs per instance

| instance | c1=1, c2=3 | c1=2, c2=2 | c1=3, c2=1 |
|---|---|---|---|
| test_1 | 20 | 20 | 20 |
| test_2 | 20 | 20 | 20 |
| test_3 | 20 | 20 | 20 |
| test_4 | 20 | 20 | 20 |
| test_5 | 20 | 20 | 20 |
| test_6 | 20 | 20 | 20 |
| test_7 | 20 | 20 | 20 |
| test_8 | 20 | 20 | 20 |
| test_9 | 20 | 20 | 20 |
| test_10 | 20 | 20 | 20 |

---

Wall-clock time is inflated and noisy whenever the study was run with `--workers > 1`, because parallel runs share the machine; CPU time and the evaluation count are the reliable effort measures in that case.
