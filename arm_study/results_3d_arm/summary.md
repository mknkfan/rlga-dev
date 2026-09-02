# Static GA on robot cycle time -- Puma 560 workcell

10 'test' instances x 10 seeds = 100 runs. **Cycle time is minimised** (seconds); an infeasible layout scores `+inf` and is excluded from the statistics, with `n_feasible` recording how many runs of each instance produced a layout at all.

## What is being optimised

The objective is the time for one full cycle: home -> every loading port in process order -> home, under a trapezoidal velocity profile per move, plus a 0.3 s dwell at each station. Each move is time-scaled so that no joint exceeds its rate or acceleration limit, and the move takes as long as the slowest joint needs -- so the cost of a layout depends on which joints a move loads, not merely on how far the tool travels.

A layout is feasible only if all four hold:

1. footprints inside the cell, clear of the robot column, and mutually disjoint;
2. every port reachable with a vertical top-down approach;
3. every joint inside its travel at every port;
4. `sigma_min` of the position Jacobian >= 0.14 m/rad at every port, which is the anti-singularity condition;

and a **single** arm posture must satisfy 2-4 at every station, since a real cell does not reconfigure the arm mid-cycle.

## Robot

4-DOF arm on Puma 560 link geometry: shoulder 0.6718 m up, upper arm 0.4318 m, forearm 0.4323 m (= hypot(0.0203, 0.4318)), shoulder offset 0.15005 m, tool 0.1 m. Reach 0.349-0.864 m from the shoulder. Joint rate limits 1.74533, 1.65806, 1.74533, 2.26893 rad/s; accelerations 6, 6, 6, 10 rad/s^2.

Joint travel is the manufacturer's, mapped into this model's angle convention by a fixed per-joint offset and otherwise unaltered: q1 -160.0 to 160.0 deg (joint 1, -160 to 160), q2 -45.0 to 225.0 deg (joint 2, -45 to 225), q3 -137.7 to 132.3 deg (joint 3, -225 to 45), q4 -97.3 to 102.7 deg (joint 5, -100 to 100). The two roll joints, 4 and 6, are locked at zero because a top-down pick does not need them.

## GA

Population 200, 300 generations (60000 evaluations), tournament k=3, uniform crossover p=0.9, mutation p=0.1 per machine, elitism 10%. **No local search, no learned control.**

Across every feasible run: cycle time 5.270 +- 0.245 s, best 4.816 s (`test_8` run 3, posture left/up).


## Per instance

| instance | feasible | cycle mean (s) | sd | median | best | min sigma mean | evals | CPU mean (s) |
|---|---|---|---|---|---|---|---|---|
| test_1 | 10/10 | 5.322 | 0.241 | 5.271 | 5.035 | 0.2335 | 54200 | 23.5 |
| test_2 | 10/10 | 5.232 | 0.141 | 5.235 | 5.023 | 0.2263 | 54200 | 28.1 |
| test_3 | 10/10 | 5.202 | 0.393 | 5.058 | 4.895 | 0.2312 | 54200 | 26.0 |
| test_4 | 10/10 | 5.230 | 0.241 | 5.208 | 4.894 | 0.2271 | 54200 | 23.1 |
| test_5 | 10/10 | 5.343 | 0.165 | 5.274 | 5.160 | 0.2328 | 54200 | 23.2 |
| test_6 | 10/10 | 5.393 | 0.095 | 5.383 | 5.231 | 0.2231 | 54200 | 23.0 |
| test_7 | 10/10 | 5.055 | 0.119 | 5.051 | 4.881 | 0.2335 | 54200 | 20.8 |
| test_8 | 10/10 | 5.106 | 0.202 | 5.086 | 4.816 | 0.2369 | 54200 | 21.3 |
| test_9 | 10/10 | 5.474 | 0.283 | 5.383 | 5.233 | 0.2112 | 54200 | 24.5 |
| test_10 | 10/10 | 5.347 | 0.174 | 5.343 | 5.105 | 0.2191 | 54200 | 22.0 |

## Posture chosen by the best layout

| posture | runs |
|---|---|
| right/down | 74 |
| left/up | 26 |

---

Wall-clock time is inflated when the study is run with `--workers > 1`; CPU time and the evaluation count are the reliable effort measures in that case.
