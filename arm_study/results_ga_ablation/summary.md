# Crossover x mutation ablation -- static GA on robot cycle time

Grid: 5 crossover rates x 6 mutation rates = 30 combinations, 20 runs on each of 10 'test' instances (6000 runs recorded).

Fixed operators: population 200, 300 generations (60000 evaluations), elitism 10%, tournament k=3, mutation width 0.1 of each gene's range. **No local search, no learned control.**

Objective: the time for one full cycle (home -> every loading port in process order -> home) with a 0.3 s dwell per station, under per-joint rate and acceleration limits, subject to `sigma_min` >= 0.14 m/rad at every port.

**Cycle time is minimised** (seconds, lower is better), and infeasible runs are excluded from the cycle-time statistics; `n_feasible` reports how many runs in each cell produced a layout at all. Evaluation counts and times cover every run.

The crossover rate is the probability that a selected pair undergoes uniform crossover; the mutation rate is the probability that any one machine (x, y, rotation) receives Gaussian jitter and a re-orientation.

`mean_rank` ranks the combinations against each other within every (instance, seed) pair -- all of which start from the same initial population -- and averages those ranks. It is the scale-free comparison; the pooled cycle-time columns mix instances whose scales differ.

Lowest mean cycle time: **crossover 0.9, mutation 0.2** (mean 5.220 s, median 5.207 s, feasible 200/200).

Best mean rank: **crossover 0.9, mutation 0.2** (8.89 of 30 cells).


## Per combination

| crossover | mutation | n | feasible | cycle mean (s) | sd | median | best | min sigma mean | evals mean | wall mean (s) | CPU mean (s) | mean rank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.9 | 0.3 | 200 | 200/200 | 5.288 | 0.207 | 5.283 | 4.838 | 0.2242 | 54200 | 16.4 | 16.2 | 11.52 |
| 0.9 | 0.2 | 200 | 200/200 | 5.220 | 0.196 | 5.207 | 4.781 | 0.2308 | 54200 | 20.1 | 19.9 | 8.89 |
| 0.9 | 0.1 | 200 | 200/200 | 5.238 | 0.215 | 5.236 | 4.708 | 0.2325 | 54200 | 30.9 | 30.5 | 9.74 |
| 0.9 | 0.05 | 200 | 200/200 | 5.306 | 0.228 | 5.296 | 4.858 | 0.2312 | 54200 | 36.5 | 36.0 | 11.90 |
| 0.9 | 0.01 | 200 | 200/200 | 5.462 | 0.248 | 5.456 | 4.984 | 0.2293 | 54200 | 39.8 | 39.3 | 17.70 |
| 0.9 | 0.001 | 200 | 200/200 | 5.781 | 0.306 | 5.742 | 5.180 | 0.2231 | 54200 | 39.1 | 38.6 | 24.57 |
| 0.7 | 0.3 | 200 | 200/200 | 5.296 | 0.200 | 5.290 | 4.848 | 0.2247 | 54200 | 17.0 | 16.8 | 12.21 |
| 0.7 | 0.2 | 200 | 200/200 | 5.238 | 0.197 | 5.253 | 4.804 | 0.2285 | 54200 | 21.1 | 20.8 | 9.47 |
| 0.7 | 0.1 | 200 | 200/200 | 5.243 | 0.203 | 5.226 | 4.783 | 0.2294 | 54200 | 31.6 | 31.2 | 9.28 |
| 0.7 | 0.05 | 200 | 200/200 | 5.333 | 0.249 | 5.291 | 4.831 | 0.2306 | 54200 | 37.2 | 36.8 | 13.02 |
| 0.7 | 0.01 | 200 | 200/200 | 5.550 | 0.312 | 5.501 | 4.919 | 0.2255 | 54200 | 40.3 | 39.8 | 20.00 |
| 0.7 | 0.001 | 200 | 200/200 | 5.880 | 0.318 | 5.816 | 5.286 | 0.2194 | 54200 | 39.2 | 38.7 | 25.82 |
| 0.5 | 0.3 | 200 | 200/200 | 5.275 | 0.216 | 5.276 | 4.840 | 0.2266 | 54200 | 17.5 | 17.3 | 11.22 |
| 0.5 | 0.2 | 200 | 200/200 | 5.245 | 0.207 | 5.254 | 4.794 | 0.2247 | 54200 | 22.3 | 22.0 | 9.46 |
| 0.5 | 0.1 | 200 | 200/200 | 5.270 | 0.224 | 5.245 | 4.860 | 0.2287 | 54200 | 32.1 | 31.7 | 10.55 |
| 0.5 | 0.05 | 200 | 200/200 | 5.346 | 0.275 | 5.286 | 4.823 | 0.2273 | 54200 | 37.5 | 37.1 | 12.92 |
| 0.5 | 0.01 | 200 | 200/200 | 5.555 | 0.308 | 5.522 | 4.993 | 0.2294 | 54200 | 40.8 | 40.3 | 19.98 |
| 0.5 | 0.001 | 200 | 200/200 | 5.979 | 0.370 | 5.960 | 5.124 | 0.2238 | 54200 | 39.2 | 38.7 | 26.65 |
| 0.3 | 0.3 | 200 | 200/200 | 5.314 | 0.215 | 5.308 | 4.893 | 0.2271 | 54200 | 18.2 | 18.0 | 12.63 |
| 0.3 | 0.2 | 200 | 200/200 | 5.260 | 0.220 | 5.249 | 4.823 | 0.2291 | 54200 | 23.1 | 22.8 | 10.23 |
| 0.3 | 0.1 | 200 | 200/200 | 5.296 | 0.263 | 5.262 | 4.737 | 0.2296 | 54200 | 32.5 | 32.0 | 10.89 |
| 0.3 | 0.05 | 200 | 200/200 | 5.426 | 0.305 | 5.370 | 4.780 | 0.2314 | 54200 | 37.8 | 37.3 | 15.11 |
| 0.3 | 0.01 | 200 | 200/200 | 5.607 | 0.375 | 5.564 | 4.869 | 0.2282 | 54200 | 40.6 | 40.0 | 20.77 |
| 0.3 | 0.001 | 200 | 200/200 | 6.074 | 0.421 | 6.014 | 5.263 | 0.2209 | 54200 | 38.8 | 38.3 | 27.44 |
| 0.1 | 0.3 | 200 | 200/200 | 5.305 | 0.196 | 5.283 | 4.858 | 0.2249 | 54200 | 19.1 | 18.9 | 12.34 |
| 0.1 | 0.2 | 200 | 200/200 | 5.307 | 0.243 | 5.276 | 4.853 | 0.2290 | 54200 | 24.1 | 23.8 | 11.96 |
| 0.1 | 0.1 | 200 | 200/200 | 5.311 | 0.250 | 5.299 | 4.759 | 0.2320 | 54200 | 33.0 | 32.6 | 11.89 |
| 0.1 | 0.05 | 200 | 200/200 | 5.426 | 0.276 | 5.398 | 4.925 | 0.2286 | 54200 | 37.7 | 37.2 | 16.00 |
| 0.1 | 0.01 | 200 | 200/200 | 5.690 | 0.377 | 5.646 | 4.948 | 0.2260 | 54200 | 40.4 | 39.9 | 22.23 |
| 0.1 | 0.001 | 200 | 200/200 | 6.236 | 0.413 | 6.222 | 5.388 | 0.2225 | 54200 | 38.8 | 38.3 | 28.61 |

## Mean cycle time (s) -- crossover (rows) x mutation (columns)

| crossover \ mutation | 0.3 | 0.2 | 0.1 | 0.05 | 0.01 | 0.001 |
|---|---|---|---|---|---|---|
| **0.1** | 5.305 | 5.307 | 5.311 | 5.426 | 5.690 | 6.236 |
| **0.3** | 5.314 | 5.260 | 5.296 | 5.426 | 5.607 | 6.074 |
| **0.5** | 5.275 | 5.245 | 5.270 | 5.346 | 5.555 | 5.979 |
| **0.7** | 5.296 | 5.238 | 5.243 | 5.333 | 5.550 | 5.880 |
| **0.9** | 5.288 | 5.220 | 5.238 | 5.306 | 5.462 | 5.781 |

## Median cycle time (s) -- crossover (rows) x mutation (columns)

| crossover \ mutation | 0.3 | 0.2 | 0.1 | 0.05 | 0.01 | 0.001 |
|---|---|---|---|---|---|---|
| **0.1** | 5.283 | 5.276 | 5.299 | 5.398 | 5.646 | 6.222 |
| **0.3** | 5.308 | 5.249 | 5.262 | 5.370 | 5.564 | 6.014 |
| **0.5** | 5.276 | 5.254 | 5.245 | 5.286 | 5.522 | 5.960 |
| **0.7** | 5.290 | 5.253 | 5.226 | 5.291 | 5.501 | 5.816 |
| **0.9** | 5.283 | 5.207 | 5.236 | 5.296 | 5.456 | 5.742 |

## Mean rank -- crossover (rows) x mutation (columns)

| crossover \ mutation | 0.3 | 0.2 | 0.1 | 0.05 | 0.01 | 0.001 |
|---|---|---|---|---|---|---|
| **0.1** | 12.34 | 11.96 | 11.89 | 16.00 | 22.23 | 28.61 |
| **0.3** | 12.63 | 10.23 | 10.89 | 15.11 | 20.77 | 27.44 |
| **0.5** | 11.22 | 9.46 | 10.55 | 12.92 | 19.98 | 26.65 |
| **0.7** | 12.21 | 9.47 | 9.28 | 13.02 | 20.00 | 25.82 |
| **0.9** | 11.52 | 8.89 | 9.74 | 11.90 | 17.70 | 24.57 |

## Feasible fraction -- crossover (rows) x mutation (columns)

| crossover \ mutation | 0.3 | 0.2 | 0.1 | 0.05 | 0.01 | 0.001 |
|---|---|---|---|---|---|---|
| **0.1** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **0.3** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **0.5** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **0.7** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **0.9** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

---

Wall-clock time is inflated and noisy whenever the grid was run with `--workers > 1`, because parallel runs share the machine; CPU time and the evaluation count are the reliable effort measures in that case. The results themselves do not depend on the worker count: a run's seed is fixed by its (instance, run) pair alone.
