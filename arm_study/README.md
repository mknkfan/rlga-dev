# arm_study — layout optimisation against robot cycle time

A static GA that places machines in a robot workcell to minimise the **cycle
time of a Puma 560**, subject to reachability and a manipulability floor.

This directory is **self-contained**: it imports nothing from the rest of the
repository, only `numpy` and `matplotlib`. Copy it anywhere and it runs.

```bash
python -m arm_study.selftest          # 54 checks on the physics and the search
python -m arm_study.run_ga --workers 12 --animate
python -m arm_study.run_ga --instances 0 --runs 3 --generations 60   # quick look
python -m arm_study.run_ga --only aggregate figures                  # rebuild from raw/
python -m arm_study.run_ablation --workers 12   # crossover x mutation grid
python -m arm_study.run_ablation --dry-run      # its plan and CPU cost first
```

---

## What changed from the distance-based study

| | distance study | this one |
|---|---|---|
| objective | Euclidean travel distance | **cycle time (s)** under trapezoidal joint motion |
| robot | a point that moves | 4-DOF arm on Puma 560 geometry, modelled as a solid |
| feasibility | overlap, workspace bounds | **+ reachability, joint travel, `σ_min` floor, arm/machine clearance** |
| cost of a move | proportional to distance | set by the *slowest joint*, so it depends on which joints the move loads |

The consequence: two layouts with identical total tool travel can differ
noticeably in cycle time, because a move that swings the waist across the cell
is paced by the waist's rate limit while a move that mostly reaches in and out
is paced by the shoulder or elbow.

## The objective

One cycle is `home → every loading port in process order → home`, plus a
0.3 s dwell at each station.

**The arm retracts between stations.** Each station is bracketed by a *lift
pose* — the same tool position raised to 6 cm above the tallest machine in the
cell — so the cycle is descend / dwell / retract / traverse, which is what a
real cell does. This is not decoration: with direct point-to-point moves the
tool interpolates in joint space between two ports and drags the forearm
straight through whatever machine stands in between. Measured on `test_1`,
**0 of 199** layouts that are clean at their stations survive the whole cycle
under direct moves; **141 of 145** do with the retract. `--no-retract`
restores the direct moves for comparison.

Each move is a coordinated point-to-point motion: all joints start and stop
together and follow the straight line between the two configurations, written
`q(t) = q_start + s(t)·Δq` with a single path parameter `s` running 0→1. `s`
gets the **trapezoidal profile** — accelerate, cruise, decelerate — degenerating
to a triangle when the move is too short to reach cruise. Joint `i` then moves
at `|Δq_i|·ṡ`, so every joint stays inside its own limits iff

```
ṡ  ≤ minᵢ (v_i / |Δq_i|)        s̈ ≤ minᵢ (a_i / |Δq_i|)
```

and the fastest legal move takes those with equality. The joint attaining the
minimum paces the move; the rest coast. Segment time is then the textbook
closed form: `T = 1/ṡ + ṡ/s̈` (trapezoid) or `T = 2√(1/s̈)` (triangle).

## Feasibility — `+∞` if any of these fails

1. **Placement** — footprints inside the cell, clear of the robot column, and
   mutually disjoint with a 20 mm gap.
2. **Reachability** — the arm can put its tool on every port with a vertical
   top-down approach.
3. **Joint travel** — every joint inside its range at every port.
4. **Manipulability** — `σ_min` of the position Jacobian ≥ **0.14 m/rad** at
   every port. This is the anti-singularity condition.
5. **Clearance** — no part of the robot comes within 5 mm of any machine, at
   every station *and* at samples along every move. The arm is bounded by a
   chain of spheres measured off the solid model in `armmesh.py`, so a layout
   that passes here *cannot* show the robot inside a box in any figure. The
   one exemption is the approach corridor: the wrist and gripper are ignored
   for a machine the tool is directly above — inside its footprint, at or
   above its top face — because the tool tip rests on that top surface by
   definition, and counting it would make every legal pick a collision. The
   forearm and upper arm still have to clear that machine, and the gripper
   still has to clear every other one. `--no-path-check` narrows this to the
   stations; `--no-arm-check` drops it entirely.
6. **One posture for the whole cycle** — a *single* IK branch must satisfy 2–5
   at every station. A real cell does not reconfigure the arm mid-cycle;
   without this the optimiser buys cycle time with elbow flips no controller
   would execute.

Hard wall rather than a penalty, deliberately: a penalty needs a scale
relating metres of overlap to seconds of cycle time, and any such choice is an
unstated preference that ends up in the answer.

## The robot

Puma 560 is a **6-DOF** arm. The task here is a top-down pick, so tool roll
about the approach axis is irrelevant and the two roll joints (the
manufacturer's 4 and 6) are **locked at zero**, leaving four driven joints.

The reference is the standard-DH table published for the Puma 560 (Unimation;
the same table Corke's Robotics Toolbox ships as `mdl_puma560`). This model's
angles differ from it by a fixed offset per joint and by nothing else, with
`ψ₃ = atan2(a3, d4) = 2.6926°`:

| this model | function | Puma joint | conversion | travel (model) | travel (Puma) |
|---|---|---|---|---|---|
| `q1` | waist | 1 | `θ₁ = q1` | −160 … +160° | −160 … +160° |
| `q2` | shoulder pitch | 2 | `θ₂ = q2` | −45 … +225° | −45 … +225° |
| `q3` | elbow pitch | 3 | `θ₃ = q3 − 90° + ψ₃` | −137.69 … +132.31° | −225 … +45° |
| `q4` | wrist pitch | 5 | `θ₅ = q4 − ψ₃` | −97.31 … +102.69° | −100 … +100° |

**The travel is the manufacturer's, unmodified** — mapped through those
offsets and neither rounded nor symmetrised, so a configuration this model
calls legal is one the real arm can hold and one it calls illegal the real arm
cannot. `PUMA_TRAVEL` holds the table verbatim; `to_puma_angles()` and
`from_puma_angles()` convert both ways and are exact inverses.

Rate limits are likewise the published maximum joint speeds — 100, 95, 100 and
130 °/s for joints 1, 2, 3 and 5. Unimation never published accelerations, so
those alone are representative values, and they stay parameters rather than
constants so the one soft number in the model is visible.

Two things the self-test pins down, both of which were wrong before:

* the `q3` offset is `90° − ψ₃`, not `ψ₃` — the manufacturer measures joint 3
  to the `d4` axis and `d4` runs *across* the elbow, so their `q3 = 0` is
  nearly perpendicular to the upper arm, not collinear with it;
* the shoulder offset `d3` is on the **negative** side of the arm plane (at
  `q1 = 0` the arm reaches along `+x` and the shoulder sits at `y = −d3`).
  The old sign mirrored the robot, which is harmless while the travel is
  symmetric and is not harmless now that it is the real, asymmetric travel.

Joint 2 travels past `+180°`, so IK solutions can no longer be folded into
`(−π, π]` — a genuine `+200°` shoulder would come back as `−160°` and be
rejected. Each joint is folded into the `2π` window centred on **its own**
travel instead.

Published link geometry (m): `a2 = 0.4318`, `a3 = 0.0203`, `d3 = 0.15005`,
`d4 = 0.4318`, shoulder height `d1 = 0.6718`, tool 0.10. The forearm is one
rigid link of length `L3 = hypot(a3, d4) = 0.4323` — exactly the real
elbow-to-wrist distance.

`robot.puma_dh_transform()` is an independent implementation of the published
DH chain, and the self-test checks this model's tool pose against it over
4 000 random configurations: **max error 4.4e-16 m**. The two models are the
same robot, not merely similar ones.

**Why the IK is closed-form.** The task is position `(x, y, z)` *plus* the
vertical-approach constraint — four constraints on four joints. So a target
has a finite set of exact solutions rather than a null space: no iteration, no
optimisation inside the objective, which is what makes 60 000 evaluations per
run affordable (~105 µs each). Four branches exist (shoulder left/right ×
elbow up/down).

**Why `σ_min` is taken on a 3×3.** The vertical approach pins
`q4 = −π/2 − q2 − q3`, so `q4` is not a degree of freedom the task can spend.
The motion actually available to the tool is 3-D, and the honest Jacobian is
`∂(x,y,z)/∂(q1,q2,q3)` along that constraint — all entries in m/rad, so its
singular values have one unit. Its determinant is

```
det J = a₂ · L₃ · sin(q₃) · √(ρ² + d₃²)
```

so the arm is singular exactly at **elbow lock-out** (`q3 = 0`) and **full
fold** (`q3 = ±π`). Note the shoulder offset `d3` keeps the second factor
away from zero — which is precisely why the real Puma has that offset: the
shoulder singularity over the base axis is designed out. The self-test
confirms `σ_min → 0` at both predicted singularities and nowhere else.

The 4×4 `task_jacobian` is also provided and finite-difference-checked, but is
not used for feasibility: its rows mix metres and radians, so its singular
values have no single unit.

## The cell

Sized to the robot, which is the point — reachability is only a real
constraint if the cell is one a Puma could plausibly serve. Floor 1.4 m
square, 6 benchtop machines 0.10–0.18 m across and 0.10–0.40 m tall, arm in
the middle on a 0.67 m pedestal. The corners sit at radius 0.99 m, well
outside the arm's envelope, so they are genuinely unusable.

Only an annulus is workable: measured on this arm under the manufacturer's
travel, a top-down pick is possible for port radii ≈ 0.15–0.74 m at the lowest
machine height (0.108 m) and 0.30–0.86 m at the highest (0.395 m), and
`σ_min ≥ 0.14` removes a further 16 % of those poses — the outer band where
the elbow approaches lock-out, which tightens the two bands to 0.21–0.67 m and
0.30–0.80 m.

Each machine's loading port sits **off-centre** on its top face, so the
rotation gene is not decorative: rotating a machine moves the point the arm
must reach, and with it the joint angles, the manipulability and the time.

Difficulty, over 2 000 random layouts per instance (mean of the 10 test
instances): **3.2 % feasible** — 88.1 % rejected on placement, 4.9 % on joint
travel, 1.9 % on `σ_min`, 1.8 % on reach (which now includes ports the arm
cannot retract from), 0.1 % on clearance. Clearance is the smallest of them on
*random* layouts, which is exactly why it had to be a constraint rather than a
diagnostic: it binds on the layouts the optimiser is drawn to, not on the ones
it starts from. Before it existed, 28 of the 100 converged layouts had the arm
inside a machine at a station and **all 100** had it inside one somewhere in
the cycle.

## The GA

Exactly the static study's no-LS variant: population 200, 300 generations
(60 000 evaluations), tournament `k=3`, uniform crossover `p=0.9`, mutation
`p=0.1` per machine (Gaussian jitter + random re-orientation), elitism 10 %.
**No local search, no RL.**

One departure from a textbook GA, and it is deliberate: the **initial
population is rejection-sampled for placement only**. Drawn uniformly, under
1 % of layouts clear all the constraints, so a random population of 200 would
contain almost no finite fitness and selection would have nothing to rank.
Seeding gives the search a foothold in the geometrically legal region; it says
nothing about reachability or manipulability, which the GA still has to find
for itself. `seeded_fraction` is recorded in every run so this can never
quietly mask a broken instance.

## Results (10 test instances × 10 seeds = 100 runs)

`results_retract/` — the current tree, produced by
`python -m arm_study.run_ga --runs 10 --workers 12 --animate --results-dir results_retract`.

All 100 runs feasible. Cycle time **10.993 ± 0.629 s**, best **9.742 s**
(`test_3`). Median best-so-far falls 14.29 → 11.13 s, and the feasible share
of the population climbs from ~27 % to ~62 % and holds — the GA learns to stay
inside the constraints rather than repeatedly falling out of them.

Audited independently afterwards, against the **drawn** robot rather than the
bounding spheres and at 41 samples per segment rather than the 12 the evaluator
tests at: **100/100 layouts are clear of every machine over the whole cycle**,
worst approach +5.0 mm, median +15.0 mm.

The posture is no longer unanimous — 73 runs chose **right/down** and 27
**left/up** — which is the retract showing up in the answer: lifting to a
common height above the cell narrows the gap between the two elbow
configurations that the direct-move objective used to punish.

Realised `σ_min` at the stations averages 0.235 against a floor of 0.14
(minimum 0.148), so the manipulability constraint is active *during* the search
but the optimum sits comfortably inside it.

`results_3d_arm/` is the previous tree: same corrected robot, but direct
point-to-point moves and no clearance criterion, so its cycle times (5.270 ±
0.245 s) are the cost of a motion the arm cannot actually execute. It is kept
for the comparison in the next section. `results/`, `results_gen500_runs20/`
and `results_ga_ablation/` are older still — they predate the joint-travel
correction as well — and none of their numbers describe this code.

## What the clearance criterion and the retract actually bought

Measured on the **drawn solid** — every vertex of the 164-polygon robot against
every machine box, with only the approach corridor exempt — not on the bounding
spheres the evaluator uses. Negative means the robot is inside a machine.

| | at the stations | over the whole cycle |
|---|---|---|
| direct moves, no clearance criterion (`results_3d_arm`) | 28/100 layouts inside a machine, worst −21.7 mm | **100/100 inside**, worst −88.0 mm, median −43.5 mm |
| direct moves + clearance at the stations | 30/30 clear, worst +8.2 mm, median +20.8 mm | **30/30 inside**, worst −86.3 mm, median −42.3 mm |
| retract + clearance over the whole cycle (`results_retract`) | 100/100 clear, worst +5.7 mm, median +16.7 mm | **100/100 clear**, worst +5.0 mm, median +15.0 mm |

The middle row is why the retract had to happen. Fixing the *stations* fixes
the poses the criteria govern and does nothing at all for the poses in between,
because the problem there is the motion, not the layout: with direct
joint-space moves the whole-cycle clearance test is not merely strict but
**unsatisfiable** — the GA finds no feasible layout in 300 generations, on any
instance. A Puma serving six machines packed into a 1.4 m cell cannot traverse
from one port to the next without putting an arm through something.

The retract costs cycle time and nothing else: the tool has to climb 6 cm above
the tallest machine and come back down at every station, so the cycle is about
twice as long as the (unachievable) direct-move figure. That is the real price
of collision-free motion in this cell, and it is now in the objective rather
than hidden behind an off-by-default switch.

### What is still not modelled

* **Only the machines.** The arm is checked against the machine boxes and its
  own pedestal keep-out disc, not against itself, the cell walls, or cabling.
* **The lift height is one number for the whole cell**, set by the tallest
  machine. A per-move height — only as high as whatever stands between *these
  two* ports — would be faster and is the obvious next refinement.
* **`sigma_min` is not required at a lift pose.** It is a via point the arm
  passes through, not one it controls fine motion at; imposing the threshold
  there would reject layouts for a condition the task never meets. Reach and
  joint travel *are* required, and a port the arm cannot retract from is
  rejected.
* **Samples, not a swept volume.** Clearance along a move is tested at
  `path_samples = 12` interior points per segment. That is a dense enough net
  for links this size against boxes this size, but it is sampling, not a proof.

## Files

| file | what it holds |
|---|---|
| `geometry.py` | `Machine`, rotated-rectangle overlap (SAT), bounds, keep-out, point-to-box distance |
| `robot.py` | `Puma560Arm`: FK, closed-form IK, Jacobians, `σ_min`, link frames, points along the links, the manufacturer's travel and DH table |
| `armmesh.py` | the solid Puma 560: parts per link frame, shading, the viewer's payload, and the bounding spheres the clearance test uses |
| `trajectory.py` | trapezoidal time scaling, segment time, cycle time, sampling, which waypoints are stations |
| `problems.py` | the instance family and the three disjoint splits |
| `evaluator.py` | `CycleTimeEvaluator` — feasibility wall, the retracted tour, arm/machine clearance, objective |
| `ga.py` | `ConfigurableGA`, `GAConfig`, seeded initial population |
| `visualize.py` | layout, 3-D trajectory, joint/rate profiles, convergence, GIF |
| `viewer.py` | the standalone drag-to-rotate HTML viewer |
| `run_ga.py` | CLI: run → aggregate → figures |
| `run_ablation.py` | CLI: the crossover × mutation ablation, run → aggregate |
| `selftest.py` | 22 checks against finite differences and closed forms |

Outputs land in `results/`: `runs.csv`, `summary.md`, `metadata.json`,
`raw/<instance>_run<k>.json`, and per instance
`figures/<instance>/{layout.png, trajectory.png, cycle.html}` (plus `cycle.gif`
with `--animate`), and `figures/convergence.png`.

The ablation writes its own tree under `results_ablation/`: `runs.csv`,
`summary.csv`, `summary_per_instance.csv`, `summary.md`, `metadata.json` and
`raw/<combo>/<instance>_run<k>.json`, one directory per cell of the grid.

## Seeing the layout in 3-D

The robot is a **solid model**, not a polyline: pedestal, rotating turret,
shoulder and elbow castings, the two arm links as flattened tapered sections,
the wrist yoke and a two-finger gripper — about 170 polygons, lit by one
directional light plus ambient. `armmesh.py` builds it, and every part is
expressed in one of the five frames `Puma560Arm.link_frames()` returns, so the
picture and the maths are the same object: the upper arm runs from the
shoulder frame's origin to `x = a2`, the forearm from the elbow frame's origin
to `x = L3`, and the gripper fingers end exactly at `x = tool`, which is the
point the IK solves for. Change a link length and the drawing follows.

This matters beyond looks. A polyline gives the eye no occlusion cue and no
sense of which way a link faces, so a pose reaching *away* from the camera
looked identical to one reaching towards it, and the arm never read as an
object standing among the machines.

A single fixed isometric viewpoint is the other half of the problem, so there
are three answers:

* **`cycle.html`** — written for every instance by default. Drag to orbit,
  scroll to zoom, shift-drag to pan, **Top** for a plan view that settles any
  overlap question outright, and a scrubber for the cycle. The **⏩** button
  sets playback speed (0.25× to 16×; shift-click or `[` to slow down, `]` to
  speed up, space to pause) — a retracted cycle is ~470 sampled instants, so
  19 s a loop at 1×, and most of the watching is done at 4× or 8×. Self-contained: no
  libraries, no network, opens straight off the filesystem. **Every** drawable
  — machine faces, robot facets, each tool-path segment, each port marker —
  goes into one list sorted back to front per frame (painter's algorithm), so
  the arm hides the machines it is in front of and is hidden by the ones it is
  behind, from any angle. The model travels once in link-local coordinates
  with only the four joint angles per frame; the page rebuilds the link
  transforms itself, which is why the page is ~50 kB rather than megabytes and
  why the arm re-lights as the camera turns. `--no-viewer` skips it.
* **`cycle.gif`** (`--animate`) — the camera orbits 360° while the cycle
  plays, so the parallax resolves the depths even though a GIF cannot be
  interactive. `--spin 0` restores a fixed viewpoint; `--spin 180` for half a
  turn.
* **`trajectory.png`** shows the solid arm **at each station**, faint to solid
  in service order, with the tool path and the wrist-centre track carrying the
  motion between them. The stations are the poses the feasibility criteria
  actually govern; evenly spaced instants would show mid-move poses the
  criteria say nothing about (see the limitation above).

Machines and robot go into a **single** matplotlib `Poly3DCollection`.
Matplotlib depth-sorts within a collection but not between collections, so a
robot in its own collection would sit always in front of the boxes or always
behind them, whichever was added last. One collection is what makes the arm
occlude and be occluded correctly.

One caveat remains on the matplotlib figures: the tool path is a `Line3D` and
still sorts separately from any collection, so it can show through a box it is
behind. The animation therefore draws the boxes at alpha 0.80 to keep the path
readable. `cycle.html` has no such problem — its tool path is cut into
per-segment drawables that sort with everything else.

## What the self-test verifies

The retracted tour is home, (lift, station, lift) per machine, home, with the
dwells on the stations and every lift pose exactly at the retract height ·
direct moves sweep the arm through a machine in 199 of 199 station-clean
layouts while the retract clears 141 of 145 · no vertex of the *drawn* robot is
inside a machine anywhere in a feasible cycle · the collision spheres provably
contain the drawn solid ·
Tool pose against an independent implementation of the manufacturer's DH table
to 4e-16 m · the angle conversions exact inverses · each joint's travel equal
to the published travel of the Puma joint it drives · the limit test agreeing
in both angle conventions over 30 000 configurations · a +200° shoulder
surviving the wrap · the solid model's tool tip landing where `fk` puts it ·
FK/IK round-trip to 2e-15 · both Jacobians against finite differences to 1e-9
· `σ_min → 0` at exactly the two singularities the determinant predicts · the
trapezoid never exceeds any joint's rate or acceleration limit and lands
exactly on target · segment time matches the textbook closed form · SAT
overlap agrees with dense point sampling · the clearance test rejects layouts
that ignoring it accepts · feasible solutions genuinely satisfy every
criterion · cycle time
equals the sum of segment times plus dwells · the GA improves monotonically
and respects its budget · runs are reproducible from their seed.
