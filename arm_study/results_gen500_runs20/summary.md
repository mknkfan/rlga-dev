# Static GA on robot cycle time -- Puma 560 workcell

10 'test' instances x 20 seeds = 200 runs. **Cycle time is minimised** (seconds); an infeasible layout scores `+inf` and is excluded from the statistics, with `n_feasible` recording how many runs of each instance produced a layout at all.

## What is being optimised

The objective is the time for one full cycle: home -> every loading port in process order -> home, under a trapezoidal velocity profile per move, plus a 0.3 s dwell at each station. Each move is time-scaled so that no joint exceeds its rate or acceleration limit, and the move takes as long as the slowest joint needs -- so the cost of a layout depends on which joints a move loads, not merely on how far the tool travels.

A layout is feasible only if all four hold:

1. footprints inside the cell, clear of the robot column, and mutually disjoint;
2. every port reachable with a vertical top-down approach;
3. every joint inside its travel at every port;
4. `sigma_min` of the position Jacobian >= 0.14 m/rad at every port, which is the anti-singularity condition;

and a **single** arm posture must satisfy 2-4 at every station, since a real cell does not reconfigure the arm mid-cycle.

## Robot

4-DOF arm on Puma 560 link geometry: shoulder 0.6718 m up, upper arm 0.4318 m, forearm 0.4323 m (= hypot(0.0203, 0.4318)), shoulder offset 0.15005 m, tool 0.1 m. Reach 0.224-0.864 m from the shoulder. Joint rate limits 1.75, 1.75, 1.75, 3.05 rad/s; accelerations 6, 6, 6, 10 rad/s^2.

## GA

Population 200, 500 generations (100000 evaluations), tournament k=3, uniform crossover p=0.9, mutation p=0.1 per machine, elitism 10%. **No local search, no learned control.**

Across every feasible run: cycle time 5.209 +- 0.213 s, best 4.633 s (`test_8` run 6, posture right/down).


## Per instance

| instance | feasible | cycle mean (s) | sd | median | best | min sigma mean | evals | CPU mean (s) |
|---|---|---|---|---|---|---|---|---|
| test_1 | 20/20 | 5.207 | 0.250 | 5.138 | 4.860 | 0.2300 | 90200 | 50.9 |
| test_2 | 20/20 | 5.290 | 0.179 | 5.265 | 5.021 | 0.2278 | 90200 | 49.8 |
| test_3 | 20/20 | 5.033 | 0.119 | 5.002 | 4.862 | 0.2449 | 90200 | 50.9 |
| test_4 | 20/20 | 5.136 | 0.178 | 5.131 | 4.888 | 0.2359 | 90200 | 49.3 |
| test_5 | 20/20 | 5.237 | 0.140 | 5.225 | 4.993 | 0.2337 | 90200 | 48.8 |
| test_6 | 20/20 | 5.386 | 0.153 | 5.416 | 5.137 | 0.2310 | 90200 | 48.7 |
| test_7 | 20/20 | 5.020 | 0.151 | 5.027 | 4.771 | 0.2389 | 90200 | 49.0 |
| test_8 | 20/20 | 5.090 | 0.252 | 5.021 | 4.633 | 0.2445 | 90200 | 51.4 |
| test_9 | 20/20 | 5.296 | 0.093 | 5.294 | 5.134 | 0.2281 | 90200 | 51.1 |
| test_10 | 20/20 | 5.391 | 0.151 | 5.354 | 5.181 | 0.2202 | 90200 | 48.1 |

## Posture chosen by the best layout

| posture | runs |
|---|---|
| right/down | 200 |

---

Wall-clock time is inflated when the study is run with `--workers > 1`; CPU time and the evaluation count are the reliable effort measures in that case.
