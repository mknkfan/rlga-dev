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

4-DOF arm on Puma 560 link geometry: shoulder 0.6718 m up, upper arm 0.4318 m, forearm 0.4323 m (= hypot(0.0203, 0.4318)), shoulder offset 0.15005 m, tool 0.1 m. Reach 0.224-0.864 m from the shoulder. Joint rate limits 1.75, 1.75, 1.75, 3.05 rad/s; accelerations 6, 6, 6, 10 rad/s^2.

## GA

Population 200, 300 generations (60000 evaluations), tournament k=3, uniform crossover p=0.9, mutation p=0.1 per machine, elitism 10%. **No local search, no learned control.**

Across every feasible run: cycle time 5.233 +- 0.214 s, best 4.708 s (`test_8` run 6, posture right/down).


## Per instance

| instance | feasible | cycle mean (s) | sd | median | best | min sigma mean | evals | CPU mean (s) |
|---|---|---|---|---|---|---|---|---|
| test_1 | 10/10 | 5.219 | 0.187 | 5.164 | 4.956 | 0.2289 | 54200 | 23.5 |
| test_2 | 10/10 | 5.377 | 0.195 | 5.306 | 5.127 | 0.2302 | 54200 | 24.3 |
| test_3 | 10/10 | 5.046 | 0.112 | 5.018 | 4.887 | 0.2423 | 54200 | 24.6 |
| test_4 | 10/10 | 5.166 | 0.171 | 5.169 | 4.898 | 0.2386 | 54200 | 24.3 |
| test_5 | 10/10 | 5.246 | 0.126 | 5.222 | 5.024 | 0.2427 | 54200 | 24.3 |
| test_6 | 10/10 | 5.451 | 0.181 | 5.486 | 5.143 | 0.2209 | 54200 | 23.1 |
| test_7 | 10/10 | 5.035 | 0.202 | 4.985 | 4.782 | 0.2406 | 54200 | 26.2 |
| test_8 | 10/10 | 5.130 | 0.275 | 5.057 | 4.708 | 0.2493 | 54200 | 26.8 |
| test_9 | 10/10 | 5.284 | 0.076 | 5.282 | 5.146 | 0.2312 | 54200 | 25.5 |
| test_10 | 10/10 | 5.376 | 0.129 | 5.353 | 5.208 | 0.2197 | 54200 | 23.2 |

## Posture chosen by the best layout

| posture | runs |
|---|---|
| right/down | 100 |

---

Wall-clock time is inflated when the study is run with `--workers > 1`; CPU time and the evaluation count are the reliable effort measures in that case.
