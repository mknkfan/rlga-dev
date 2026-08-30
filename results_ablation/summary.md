# Crossover x mutation ablation - static GA

Grid: 5 crossover rates x 4 mutation rates = 20 combinations, 20 runs on each of 10 'test' instances (4000 runs recorded).

Fixed operators: population 200, 300 generations, elitism 0.1, tournament k=3, local search off.

**Fitness is minimised** (lower is better), and infeasible runs are excluded from the fitness statistics; `n_feasible` reports how many runs in each cell were feasible. Evaluation counts and times cover every run.

The crossover rate is the probability that a selected pair undergoes uniform crossover; the mutation rate is the probability that any one machine (x, y, rotation) receives Gaussian jitter and a random re-orientation.

`mean_rank` ranks the combinations against each other within every (instance, seed) pair - all of which start from the same initial population - and averages those ranks. It is the scale-free comparison; the pooled fitness columns mix instances whose fitness scales differ.

Lowest mean fitness: **crossover 0.9, mutation 0.1** (mean 35.4797, median 34.7754).


## Per combination

| crossover | mutation | n | feasible | fitness mean | fitness median | evals mean | evals median | wall mean (s) | wall median (s) | CPU mean (s) | CPU median (s) | mean rank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.9 | 0.1 | 200 | 200/200 | 35.4797 | 34.7754 | 60000 | 60000 | 8.36 | 8.31 | 8.25 | 8.20 | 3.35 |
| 0.9 | 0.01 | 200 | 200/200 | 39.6266 | 39.1397 | 60000 | 60000 | 9.76 | 9.47 | 9.63 | 9.34 | 6.77 |
| 0.9 | 0.001 | 200 | 200/200 | 46.4268 | 46.2489 | 60000 | 60000 | 9.12 | 8.85 | 9.00 | 8.71 | 11.45 |
| 0.9 | 0.0001 | 200 | 200/200 | 56.2073 | 56.0541 | 60000 | 60000 | 8.96 | 8.48 | 8.85 | 8.38 | 15.68 |
| 0.7 | 0.1 | 200 | 200/200 | 36.4473 | 36.6798 | 60000 | 60000 | 8.48 | 8.48 | 8.37 | 8.38 | 4.07 |
| 0.7 | 0.01 | 200 | 200/200 | 39.4684 | 39.5278 | 60000 | 60000 | 9.87 | 9.60 | 9.75 | 9.44 | 6.49 |
| 0.7 | 0.001 | 200 | 200/200 | 47.2899 | 47.4352 | 60000 | 60000 | 9.28 | 8.84 | 9.16 | 8.71 | 11.97 |
| 0.7 | 0.0001 | 200 | 200/200 | 60.5398 | 60.1902 | 60000 | 60000 | 9.06 | 8.61 | 8.94 | 8.53 | 17.01 |
| 0.5 | 0.1 | 200 | 200/200 | 36.5491 | 36.7215 | 60000 | 60000 | 8.65 | 8.60 | 8.53 | 8.48 | 4.18 |
| 0.5 | 0.01 | 200 | 200/200 | 40.0628 | 40.0377 | 60000 | 60000 | 9.97 | 9.74 | 9.84 | 9.66 | 7.16 |
| 0.5 | 0.001 | 200 | 200/200 | 48.6308 | 48.4699 | 60000 | 60000 | 9.27 | 9.05 | 9.15 | 8.91 | 12.68 |
| 0.5 | 0.0001 | 200 | 200/200 | 64.2525 | 64.4181 | 60000 | 60000 | 8.81 | 8.43 | 8.69 | 8.34 | 17.79 |
| 0.3 | 0.1 | 200 | 200/200 | 37.2024 | 37.2965 | 60000 | 60000 | 8.73 | 8.61 | 8.61 | 8.52 | 4.53 |
| 0.3 | 0.01 | 200 | 200/200 | 41.3540 | 41.0857 | 60000 | 60000 | 9.86 | 9.76 | 9.73 | 9.63 | 8.02 |
| 0.3 | 0.001 | 200 | 200/200 | 50.3852 | 50.4888 | 60000 | 60000 | 9.30 | 8.98 | 9.18 | 8.83 | 13.35 |
| 0.3 | 0.0001 | 200 | 200/200 | 69.2425 | 69.1522 | 60000 | 60000 | 8.82 | 8.46 | 8.70 | 8.36 | 18.59 |
| 0.1 | 0.1 | 200 | 200/200 | 37.3865 | 37.4690 | 60000 | 60000 | 8.79 | 8.64 | 8.68 | 8.56 | 4.93 |
| 0.1 | 0.01 | 200 | 200/200 | 41.5021 | 40.9784 | 60000 | 60000 | 9.85 | 9.71 | 9.72 | 9.55 | 8.16 |
| 0.1 | 0.001 | 200 | 200/200 | 52.1740 | 51.9444 | 60000 | 60000 | 9.21 | 8.99 | 9.09 | 8.87 | 14.29 |
| 0.1 | 0.0001 | 200 | 200/200 | 77.0490 | 76.3203 | 60000 | 60000 | 8.80 | 8.46 | 8.68 | 8.35 | 19.55 |

## Mean fitness - crossover (rows) x mutation (columns)

| crossover \ mutation | 0.1 | 0.01 | 0.001 | 0.0001 |
|---|---|---|---|---|
| **0.1** | 37.3865 | 41.5021 | 52.1740 | 77.0490 |
| **0.3** | 37.2024 | 41.3540 | 50.3852 | 69.2425 |
| **0.5** | 36.5491 | 40.0628 | 48.6308 | 64.2525 |
| **0.7** | 36.4473 | 39.4684 | 47.2899 | 60.5398 |
| **0.9** | 35.4797 | 39.6266 | 46.4268 | 56.2073 |

## Median fitness - crossover (rows) x mutation (columns)

| crossover \ mutation | 0.1 | 0.01 | 0.001 | 0.0001 |
|---|---|---|---|---|
| **0.1** | 37.4690 | 40.9784 | 51.9444 | 76.3203 |
| **0.3** | 37.2965 | 41.0857 | 50.4888 | 69.1522 |
| **0.5** | 36.7215 | 40.0377 | 48.4699 | 64.4181 |
| **0.7** | 36.6798 | 39.5278 | 47.4352 | 60.1902 |
| **0.9** | 34.7754 | 39.1397 | 46.2489 | 56.0541 |

## Mean rank - crossover (rows) x mutation (columns)

| crossover \ mutation | 0.1 | 0.01 | 0.001 | 0.0001 |
|---|---|---|---|---|
| **0.1** | 4.93 | 8.16 | 14.29 | 19.55 |
| **0.3** | 4.53 | 8.02 | 13.35 | 18.59 |
| **0.5** | 4.18 | 7.16 | 12.68 | 17.79 |
| **0.7** | 4.07 | 6.49 | 11.97 | 17.01 |
| **0.9** | 3.35 | 6.77 | 11.45 | 15.68 |

---

Wall-clock time is inflated and noisy whenever the grid was run with `--workers > 1`, because parallel runs share the machine; CPU time and the evaluation count are the reliable effort measures in that case.
