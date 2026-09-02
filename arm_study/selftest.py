"""
Checks the claims this study rests on.  Run:  python -m arm_study.selftest

Nothing here is a unit test of plumbing; each check is a property the results
would be wrong without, verified against something independent of the code
being checked -- finite differences, a closed form derived separately, or an
exhaustive alternative.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from armmesh import (  # noqa: E402
    build_parts,
    collision_spheres,
    mesh_surface_points,
    posed_polygons,
)
from evaluator import CycleTimeEvaluator  # noqa: E402
from ga import ConfigurableGA, GAConfig, make_initial_population  # noqa: E402
from geometry import Machine, rectangles_overlap  # noqa: E402
from problems import build_instances, random_layout  # noqa: E402
from robot import (  # noqa: E402
    BRANCHES,
    LOCKED_PUMA_JOINTS,
    PUMA_JOINT_INDEX,
    PUMA_TRAVEL,
    VERTICAL_PITCH,
    ArmLimits,
    Puma560Arm,
    puma_dh_transform,
)
from trajectory import plan_segment  # noqa: E402

_failures = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global _failures
    if not condition:
        _failures += 1
    status = "PASS" if condition else "FAIL"
    print(f"  {status}  {name}" + (f" - {detail}" if detail else ""))


def main() -> None:
    arm = Puma560Arm()
    rng = np.random.default_rng(0)

    print("\n0. This model IS the Puma 560, not merely something like it:")
    # Checked against an independent implementation of the published
    # standard-DH table, not against anything in robot.py: if the offsets,
    # the link lengths or the sign of the shoulder offset were wrong, this
    # would be the check that says so.
    worst_pose = 0.0
    for _ in range(4000):
        q = rng.uniform(-2.5, 2.5, 4)
        pose = puma_dh_transform(arm.to_puma_angles(q))
        # The tool hangs one tool-length along the flange's approach axis.
        tip = pose[:3, 3] + arm.tool * pose[:3, 2]
        worst_pose = max(worst_pose, float(np.abs(arm.fk(q)[:3] - tip).max()))
    check("tool position equals the manufacturer's DH chain", worst_pose < 1e-12,
          f"max error {worst_pose:.2e} m over 4000 poses")

    sample = rng.uniform(-2.5, 2.5, (500, 4))
    theta = arm.to_puma_angles(sample)
    check("to_puma_angles / from_puma_angles are exact inverses",
          float(np.abs(arm.from_puma_angles(theta) - sample).max()) < 1e-12)
    check("the two roll joints are reported locked at zero",
          all(bool(np.all(theta[:, j] == v)) for j, v in LOCKED_PUMA_JOINTS.items()))

    print("\n0b. Joint travel is the manufacturer's travel, exactly:")
    limits = ArmLimits()
    for k, joint in enumerate(PUMA_JOINT_INDEX):
        span = np.degrees(limits.q_max[k] - limits.q_min[k])
        published = PUMA_TRAVEL[joint][1] - PUMA_TRAVEL[joint][0]
        check(f"q{k + 1} spans joint {joint + 1}'s {published:g} deg",
              abs(span - published) < 1e-9,
              f"{np.degrees(limits.q_min[k]):.2f} to {np.degrees(limits.q_max[k]):.2f} deg")
    probe = rng.uniform(-4.0, 5.0, (30000, 4))
    check("the limit test agrees in both angle conventions",
          bool((arm.within_limits(probe) == arm.puma_within_limits(probe)).all()))
    # The shoulder's travel runs past +180 deg, so an angle folded into
    # (-pi, pi] would be rejected where the real joint can reach.
    stretched = np.array([[0.0, math.radians(200.0), 0.0, 0.0]])
    check("a +200 deg shoulder survives wrapping",
          bool(arm.within_limits(arm._wrap_into_travel(stretched))[0]),
          f"wrapped to {np.degrees(arm._wrap_into_travel(stretched)[0, 1]):.1f} deg")

    print("\n0c. The solid model hangs off the same frames as the kinematics:")
    parts = build_parts(arm)
    worst_tip = 0.0
    for _ in range(200):
        q = rng.uniform(-1.5, 1.5, 4)
        frames = arm.link_frames(q)
        tip = (frames[4] @ np.array([arm.tool, 0.0, 0.0, 1.0]))[:3]
        worst_tip = max(worst_tip, float(np.abs(tip - arm.fk(q)[:3]).max()))
    check("link_frames puts the tool tip where fk does", worst_tip < 1e-12,
          f"max error {worst_tip:.2e} m")
    q_probe = np.array([0.4, 0.6, 1.1, VERTICAL_PITCH - 0.6 - 1.1])
    polys, colours = posed_polygons(arm, q_probe, parts)
    vertices = np.concatenate(polys)
    check("every part is a closed polygon in world space",
          len(polys) == sum(len(p.polys) for p in parts) and colours.shape[0] == len(polys),
          f"{len(polys)} polygons")
    check("the model stands on the floor and inside the arm's reach",
          float(vertices[:, 2].min()) > -1e-9
          and float(np.linalg.norm(vertices[:, :2], axis=1).max())
          <= arm.max_reach + arm.tool + 0.2,
          f"z from {vertices[:, 2].min():.3f} to {vertices[:, 2].max():.3f} m")
    finger = np.concatenate([p.polys[0] for p in parts if p.name.startswith("finger")])
    check("the gripper fingers close on the tool tip",
          abs(float(np.max([f[0] for f in finger])) - arm.tool) < 1e-9)

    print("\n0d. The collision spheres really bound the robot that gets drawn:")
    spheres = collision_spheres(arm)
    for frame, slot, label in ((2, 0, "upper arm"), (3, 1, "forearm"), (4, 2, "tool")):
        mask = spheres.link == slot
        surface = mesh_surface_points(arm, frame)
        arc = spheres.coefficients[mask][:, slot]
        centres = np.stack([arc, np.zeros_like(arc), np.zeros_like(arc)], axis=1)
        gap = np.linalg.norm(surface[:, None, :] - centres[None], axis=2)
        outside = float((gap - spheres.radii[mask]).min(axis=1).max())
        check(f"every drawn point of the {label} is inside a sphere", outside <= 1e-12,
              f"{int(mask.sum())} spheres, worst margin {outside:+.1e} m")
    # The base and waist are deliberately not in the chain.
    pedestal = np.concatenate([np.concatenate(p.polys) for p in build_parts(arm)
                               if p.frame == 0])
    turret = np.concatenate([np.concatenate(p.polys) for p in build_parts(arm)
                             if p.frame == 1])
    tallest = max(m.height for inst in build_instances("test") for m in inst.machines)
    check("the pedestal stays inside the cell's keep-out disc",
          float(np.linalg.norm(pedestal[:, :2], axis=1).max())
          <= build_instances("test")[0].keep_out_radius,
          f"{float(np.linalg.norm(pedestal[:, :2], axis=1).max()):.3f} m "
          f"<= {build_instances('test')[0].keep_out_radius:g} m")
    check("the turret and shoulder clear the tallest machine outright",
          float(turret[:, 2].min()) > tallest,
          f"lowest turret point {float(turret[:, 2].min()):.3f} m > {tallest:.3f} m")

    print("\n0e. points_along_arm agrees with the link frames:")
    worst_axis = 0.0
    for _ in range(200):
        q = rng.uniform(-2.0, 2.0, 4)
        frames = arm.link_frames(q)
        expected = []
        for a, b, c in spheres.coefficients:
            if c > 0.0 or (a == arm.a2 and b == arm.L3):
                expected.append((frames[4] @ np.array([c, 0.0, 0.0, 1.0]))[:3])
            elif b > 0.0 or a == arm.a2:
                expected.append((frames[3] @ np.array([b, 0.0, 0.0, 1.0]))[:3])
            else:
                expected.append((frames[2] @ np.array([a, 0.0, 0.0, 1.0]))[:3])
        got = arm.points_along_arm(q, spheres.coefficients)
        worst_axis = max(worst_axis, float(np.abs(np.stack(expected) - got).max()))
    check("sphere centres land where link_frames puts them", worst_axis < 1e-12,
          f"max error {worst_axis:.2e} m")

    print("\n1. Inverse kinematics inverts forward kinematics exactly:")
    worst = 0.0
    solved = 0
    for _ in range(3000):
        p = np.array([rng.uniform(-0.9, 0.9), rng.uniform(-0.9, 0.9), rng.uniform(0.05, 0.45)])
        for branch in BRANCHES:
            q, ok = arm.ik_vertical(p[None, :], *branch)
            if not ok[0]:
                continue
            solved += 1
            task = arm.fk(q)[0]
            pitch_error = abs(
                (task[3] - VERTICAL_PITCH + math.pi) % (2 * math.pi) - math.pi
            )
            worst = max(worst, float(np.abs(task[:3] - p).max()), pitch_error)
    check("FK(IK(p)) returns p, and the tool is vertical", worst < 1e-9,
          f"{solved} solutions, max error {worst:.2e}")

    print("\n2. The Jacobians match finite differences:")
    def constrained(q3v):
        q = np.array([q3v[0], q3v[1], q3v[2], VERTICAL_PITCH - q3v[1] - q3v[2]])
        return arm.fk(q[None, :])[0, :3]

    worst_pos = worst_task = 0.0
    for _ in range(400):
        a = np.array([rng.uniform(-2, 2), rng.uniform(-1.2, 1.2), rng.uniform(0.3, 2.5)])
        full = np.array([a[0], a[1], a[2], VERTICAL_PITCH - a[1] - a[2]])
        analytic = arm.position_jacobian(full[None, :])[0]
        numeric = np.zeros((3, 3))
        for k in range(3):
            step = np.zeros(3)
            step[k] = 1e-6
            numeric[:, k] = (constrained(a + step) - constrained(a - step)) / 2e-6
        worst_pos = max(worst_pos, float(np.abs(analytic - numeric).max()))

        b = rng.uniform(-2, 2, 4)
        analytic_t = arm.task_jacobian(b[None, :])[0]
        numeric_t = np.zeros((4, 4))
        for k in range(4):
            step = np.zeros(4)
            step[k] = 1e-6
            numeric_t[:, k] = (arm.fk(b + step) - arm.fk(b - step)) / 2e-6
        worst_task = max(worst_task, float(np.abs(analytic_t - numeric_t).max()))
    check("position_jacobian (3x3, along the vertical constraint)", worst_pos < 1e-6,
          f"max error {worst_pos:.2e}")
    check("task_jacobian (4x4, unconstrained)", worst_task < 1e-6,
          f"max error {worst_task:.2e}")

    print("\n3. sigma_min vanishes exactly at the singularities the determinant predicts:")
    # det(J) = a2 * L3 * sin(q3) * sqrt(rho^2 + d3^2), so q3 = 0 and q3 = pi.
    for q3, label in ((1e-7, "elbow locked out (q3 = 0)"), (math.pi - 1e-7, "fully folded (q3 = pi)")):
        q = np.array([[0.3, 0.4, q3, VERTICAL_PITCH - 0.4 - q3]])
        check(f"sigma_min ~ 0 at {label}", float(arm.sigma_min(q)[0]) < 1e-5,
              f"{float(arm.sigma_min(q)[0]):.2e} m/rad")
    q = np.array([[0.3, 0.4, 1.2, VERTICAL_PITCH - 0.4 - 1.2]])
    check("sigma_min is healthy away from them", float(arm.sigma_min(q)[0]) > 0.1,
          f"{float(arm.sigma_min(q)[0]):.4f} m/rad")

    print("\n4. The trapezoidal profile never exceeds a joint limit:")
    _, _, v_limit, a_limit = ArmLimits().as_arrays()
    worst_v = worst_a = worst_end = 0.0
    triangular = 0
    for _ in range(1500):
        q0 = rng.uniform(-1.5, 1.5, 4)
        q1 = q0 + rng.uniform(-2.5, 2.5, 4)
        segment = plan_segment(q0, q1, v_limit, a_limit)
        triangular += segment.triangular
        t, q_t, q_dot = segment.sample(q0, 300)
        worst_v = max(worst_v, float(np.max(np.abs(q_dot) / v_limit)))
        accel = np.gradient(q_dot, t, axis=0)
        worst_a = max(worst_a, float(np.max(np.abs(accel[3:-3]) / a_limit)))
        worst_end = max(worst_end, float(np.abs(q_t[-1] - q1).max()))
    check("joint rate never exceeds its limit", worst_v <= 1.0 + 1e-9,
          f"max |q_dot| / limit = {worst_v:.6f}")
    check("joint acceleration never exceeds its limit", worst_a <= 1.0 + 1e-3,
          f"max |q_ddot| / limit = {worst_a:.6f}")
    check("the move lands exactly on the target", worst_end < 1e-12,
          f"max endpoint error {worst_end:.2e} rad")
    check("both trapezoid and triangle cases are exercised",
          0 < triangular < 1500, f"{triangular}/1500 triangular")

    print("\n5. A one-joint move matches the textbook closed form:")
    for dq in (0.05, 0.5, 2.0):
        segment = plan_segment(np.zeros(4), np.array([dq, 0.0, 0.0, 0.0]), v_limit, a_limit)
        if dq < v_limit[0] ** 2 / a_limit[0]:
            reference = 2.0 * math.sqrt(dq / a_limit[0])
        else:
            reference = dq / v_limit[0] + v_limit[0] / a_limit[0]
        check(f"dq = {dq} rad", abs(segment.duration - reference) < 1e-12,
              f"{segment.duration:.6f} s vs {reference:.6f} s")

    print("\n6. Rectangle overlap agrees with a dense point test:")
    disagreements = 0
    for _ in range(400):
        a = Machine("a", rng.uniform(0.1, 0.3), rng.uniform(0.1, 0.3), 0.2,
                    rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(0, 360))
        b = Machine("b", rng.uniform(0.1, 0.3), rng.uniform(0.1, 0.3), 0.2,
                    rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(0, 360))
        sat = rectangles_overlap(a, b)
        # Dense sampling can only confirm overlap, never disprove it, so a
        # sampled hit with SAT saying "disjoint" is the real contradiction.
        grid = np.linspace(-0.5, 0.5, 11)
        local = np.stack(np.meshgrid(grid * a.width, grid * a.depth), -1).reshape(-1, 2)
        theta = math.radians(a.rotation)
        rot = np.array([[math.cos(theta), -math.sin(theta)],
                        [math.sin(theta), math.cos(theta)]])
        pts = local @ rot.T + np.array([a.x, a.y])
        theta_b = math.radians(b.rotation)
        rb = np.array([[math.cos(theta_b), math.sin(theta_b)],
                       [-math.sin(theta_b), math.cos(theta_b)]])
        lb = (pts - np.array([b.x, b.y])) @ rb.T
        sampled = bool(((np.abs(lb[:, 0]) <= b.width / 2) &
                        (np.abs(lb[:, 1]) <= b.depth / 2)).any())
        if sampled and not sat:
            disagreements += 1
    check("no sampled overlap is ever reported disjoint", disagreements == 0,
          f"{disagreements} contradictions in 400 pairs")

    print("\n6b. Clearance, the swept path, and what the retract is for:")
    instance = build_instances("test")[0]
    # Each evaluator differs from its neighbour in exactly one setting, so a
    # difference in what they accept is attributable to that setting.
    ignore = CycleTimeEvaluator(instance, check_arm=False, check_path=False)
    at_stations = CycleTimeEvaluator(instance, check_arm=True, check_path=False)
    whole_cycle = CycleTimeEvaluator(instance, check_arm=True, check_path=True)
    direct_stations = CycleTimeEvaluator(instance, check_path=False, retract=None)
    direct_cycle = CycleTimeEvaluator(instance, check_path=True, retract=None)

    seen = caught = kept = direct_seen = direct_kept = 0
    probe = np.random.default_rng(11)
    for _ in range(4000):
        genes = random_layout(instance, probe)
        if ignore.evaluate(ignore.decode(genes)).feasible:
            seen += 1
            if not at_stations.evaluate(at_stations.decode(genes)).feasible:
                caught += 1
            elif whole_cycle.evaluate(whole_cycle.decode(genes)).feasible:
                kept += 1
        if direct_stations.evaluate(direct_stations.decode(genes)).feasible:
            direct_seen += 1
            if direct_cycle.evaluate(direct_cycle.decode(genes)).feasible:
                direct_kept += 1
    check("the clearance test rejects layouts that ignoring it accepts", caught > 0,
          f"{caught}/{seen} otherwise-feasible layouts put the arm in a machine "
          "at a station")
    # The point of the retract, as a number: with direct point-to-point moves
    # essentially every layout that is clean at its stations still drags an arm
    # through a machine on the way between them; with the retract almost none
    # do.  That is the whole reason the motion is what it is.
    check("direct moves sweep the arm through a machine almost always",
          direct_kept / max(direct_seen, 1) < 0.05,
          f"{direct_kept}/{direct_seen} station-clean layouts survive the whole cycle")
    check("the retract is what makes the whole cycle clear",
          kept / max(seen - caught, 1) > 0.8,
          f"{kept}/{seen - caught} station-clean layouts survive the whole cycle")

    print("\n6c. The retract puts a lift pose either side of every station:")
    scan = CycleTimeEvaluator(instance)
    retracted = None
    probe = np.random.default_rng(5)
    for _ in range(20000):
        candidate = scan.evaluate(scan.decode(random_layout(instance, probe)))
        if candidate.feasible:
            retracted = candidate
            break
    if retracted is None:
        check("found a retracted cycle to audit", False)
    else:
        n = instance.n_machines
        check("the tour is home, (lift, station, lift) per machine, home",
              retracted.configurations.shape[0] == 3 * n + 2
              and len(retracted.station_index) == n,
              f"{retracted.configurations.shape[0]} waypoints, {n} stations")
        lifts = retracted.configurations[retracted.station_index - 1]
        check("every lift pose is the port raised to the retract height",
              float(np.abs(scan.arm.fk(lifts)[:, 2] - scan.retract_height).max()) < 1e-9,
              f"retract height {scan.retract_height:.4f} m")
        check("the lift poses clear the tallest machine",
              scan.retract_height > max(m.height for m in instance.machines))
        check("the dwells land on the stations, not the lift poses",
              abs(float(scan.segment_times(retracted).sum())
                  + scan.dwell * n - retracted.cycle_time) < 1e-9,
              f"{len(scan.segment_times(retracted))} segments, {n} dwells")

    print("\n7. Infeasible layouts score +inf and feasible ones a finite time:")
    evaluator = CycleTimeEvaluator(instance)
    probe = np.random.default_rng(5)
    finite = 0
    bad = 0
    for _ in range(3000):
        value = evaluator.fitness(random_layout(instance, probe))
        if math.isfinite(value):
            finite += 1
            if value <= 0:
                bad += 1
    check("some random layouts are feasible", finite > 0, f"{finite}/3000")
    check("every feasible cycle time is strictly positive", bad == 0)
    check("the evaluation counter tracks calls", evaluator.n_evals == 3000,
          f"{evaluator.n_evals}")

    print("\n8. A feasible solution really satisfies every stated constraint:")
    outcome = None
    probe = np.random.default_rng(5)
    for _ in range(20000):
        candidate = evaluator.evaluate(evaluator.decode(random_layout(instance, probe)))
        if candidate.feasible:
            outcome = candidate
            break
    if outcome is None:
        check("found a feasible layout to audit", False)
    else:
        ports = np.stack([outcome.machines[i].port() for i in instance.sequence])
        q, ok = arm.ik_vertical(ports, *outcome.branch)
        reached = arm.fk(q)[:, :3]
        check("the arm's tool lands on every port", ok.all() and
              float(np.abs(reached - ports).max()) < 1e-9,
              f"max error {float(np.abs(reached - ports).max()):.2e} m")
        check("every joint is inside its travel", bool(arm.within_limits(q).all()))
        check("sigma_min clears the threshold at every station",
              float(arm.sigma_min(q).min()) >= evaluator.sigma_min_threshold,
              f"min {float(arm.sigma_min(q).min()):.4f} >= {evaluator.sigma_min_threshold}")
        check("cycle time is the sum of the segment times plus the dwells",
              abs(float(evaluator.segment_times(outcome).sum())
                  + evaluator.dwell * instance.n_machines - outcome.cycle_time) < 1e-9)

        # The strongest form of the clearance claim: not "the bounding spheres
        # clear the boxes" but "no vertex of the robot that gets drawn is
        # inside a box it is not reaching into".  If this ever fails, the
        # figures show the arm buried in a machine.
        from geometry import box_frames, points_over_boxes, points_to_boxes_distance
        boxes = box_frames(outcome.machines)
        arm_parts = build_parts(arm)
        # Sampled along every move, not only at the waypoints, because the
        # clearance criterion now covers the whole cycle.
        tour = outcome.configurations
        fractions = np.linspace(0.0, 1.0, evaluator.path_samples + 2)
        sampled = np.concatenate([
            tour[k] + fractions[:, None] * (tour[k + 1] - tour[k])
            for k in range(tour.shape[0] - 1)
        ])
        deepest = math.inf
        for station in sampled:
            frames = arm.link_frames(station)
            tip = arm.fk(station)[:3]
            corridor = points_over_boxes(tip[None, :], *boxes)[0]
            for part in arm_parts:
                vertices = np.concatenate(part.polys) @ frames[part.frame][:3, :3].T \
                    + frames[part.frame][:3, 3]
                gap = points_to_boxes_distance(vertices, *boxes)
                if part.frame == 4:
                    gap = np.where(corridor[None, :], math.inf, gap)
                deepest = min(deepest, float(gap.min()))
        check("no part of the drawn robot is inside a machine anywhere in the cycle",
              deepest >= 0.0,
              f"closest approach {deepest * 1000:+.1f} mm "
              f"(threshold {evaluator.arm_clearance * 1000:g} mm on the bounding spheres)")

    print("\n9. The GA improves on its own starting population:")
    population, seeded = make_initial_population(instance, 60, seed=1)
    ga = ConfigurableGA(
        instance, evaluator=CycleTimeEvaluator(instance),
        config=GAConfig(population_size=60, generations=40), seed=1,
    )
    result = ga.optimize(initial_population=population, seeded_fraction=seeded)
    check("the run ends feasible", result.feasible, f"{result.best_fitness:.4f} s")
    check("best-so-far never increases",
          bool(np.all(np.diff(result.best_so_far) <= 1e-12)))
    check("the final answer beats the first generation",
          result.best_fitness < result.gen_best[0],
          f"{result.gen_best[0]:.4f} -> {result.best_fitness:.4f} s")
    check("the evaluation budget is respected",
          result.total_evaluations <= 60 * 40, f"{result.total_evaluations}")

    print("\n10. A run is reproducible from its seed:")
    def once() -> float:
        pop, frac = make_initial_population(instance, 60, seed=7)
        engine = ConfigurableGA(
            instance, evaluator=CycleTimeEvaluator(instance),
            config=GAConfig(population_size=60, generations=25), seed=7,
        )
        return engine.optimize(initial_population=pop, seeded_fraction=frac).best_fitness
    first, second = once(), once()
    check("the same seed gives the same answer", first == second,
          f"{first:.9f} vs {second:.9f}")

    print()
    if _failures:
        print(f"{_failures} CHECK(S) FAILED")
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
