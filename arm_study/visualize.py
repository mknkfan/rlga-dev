"""
Figures for a solved layout: the cell, the arm's path through it, and the
motion profiles that produced the cycle time.

Three figures, because they answer three different questions:

``layout.png``      where did the machines end up, and is that sensible given
                    the reachable annulus?
``trajectory.png``  what does the arm actually do -- the tool path in 3-D, the
                    joint angles, and the trapezoidal rate profiles with the
                    limits drawn on so it is visible which joint paces each
                    move.
``convergence.png`` did the GA converge, and how fast did the population
                    become feasible?

Optionally ``cycle.gif``, which is the same trajectory animated.  It is off by
default because writing it costs more than the three static figures together.

Everything here is matplotlib only and reads a
:class:`~arm_study.evaluator.LayoutOutcome`; nothing re-solves anything.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

from armmesh import build_parts, posed_polygons, shade, viewer_payload
from evaluator import CycleTimeEvaluator, LayoutOutcome
from ga import RunResult
from problems import Instance
from robot import BRANCHES, Puma560Arm
from trajectory import dwell_indices, plan_cycle, sample_cycle

DPI = 170
_JOINT_NAMES = ("q1 waist", "q2 shoulder", "q3 elbow", "q4 wrist pitch")
_JOINT_COLOURS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
#: Base colour of a machine box, lit by the same light as the robot.
_MACHINE_RGB = (0.298, 0.447, 0.690)


# ----------------------------------------------------------------------
# the reachable annulus
# ----------------------------------------------------------------------


def reachable_radii(
    arm: Puma560Arm, height: float, threshold: float, samples: int = 500
) -> Tuple[float, float]:
    """Inner and outer radius the arm can serve at a given port height.

    Swept numerically rather than solved: the limit is set by whichever of the
    joint travel, the annulus and ``sigma_min`` binds first, and sweeping is
    both shorter and harder to get wrong than case analysis.
    """
    radii = np.linspace(0.05, arm.max_reach + arm.tool, samples)
    points = np.stack([radii, np.zeros_like(radii), np.full_like(radii, height)], axis=1)
    usable = np.zeros(radii.size, dtype=bool)
    for branch in BRANCHES:
        q, ok = arm.ik_vertical(points, *branch)
        usable |= ok & arm.within_limits(q) & (arm.sigma_min(q) >= threshold)
    if not usable.any():
        return (0.0, 0.0)
    return float(radii[usable].min()), float(radii[usable].max())


# ----------------------------------------------------------------------
# the cell, seen from above
# ----------------------------------------------------------------------


def draw_layout(
    ax,
    instance: Instance,
    outcome: LayoutOutcome,
    arm: Puma560Arm,
    threshold: float,
) -> None:
    """Top-down plan: machines, ports, the service order, the annulus."""
    min_x, max_x, min_y, max_y = instance.bounds
    machines = outcome.machines or instance.machines

    # The annulus at the mean port height: the band the arm can actually work
    # in.  Drawn first so everything else sits on top of it.
    mean_height = float(np.mean([m.height for m in machines]))
    inner, outer = reachable_radii(arm, mean_height, threshold)
    ax.add_patch(Circle(instance.base_xy, outer, facecolor="#2ca02c", alpha=0.07, edgecolor="none"))
    ax.add_patch(Circle(instance.base_xy, inner, facecolor="white", edgecolor="none", zorder=1.5))
    for radius, style in ((inner, ":"), (outer, "--")):
        ax.add_patch(
            Circle(instance.base_xy, radius, fill=False, edgecolor="#2ca02c",
                   linestyle=style, linewidth=1.1, zorder=1.6)
        )

    # The robot's own column.
    ax.add_patch(
        Circle(instance.base_xy, instance.keep_out_radius, facecolor="#bbbbbb",
               edgecolor="#555555", alpha=0.85, zorder=2)
    )

    for index, machine in enumerate(machines):
        corners = machine.corners()
        order = instance.sequence.index(index) + 1
        ax.add_patch(
            Polygon(corners, closed=True, facecolor="#4c72b0", edgecolor="#22344f",
                    alpha=0.55, linewidth=1.2, zorder=3)
        )
        ax.text(machine.x, machine.y, f"{machine.name}\n#{order}", ha="center", va="center",
                fontsize=7.5, zorder=5, color="#10203a")
        port = machine.port()
        ax.plot(port[0], port[1], "o", color="#d62728", markersize=4.5, zorder=6)

    # The order the ports are served, home at both ends.
    home = np.asarray(instance.home_point, dtype=float)
    path = np.vstack([home[:2],
                      np.stack([machines[i].port()[:2] for i in instance.sequence]),
                      home[:2]])
    ax.plot(path[:, 0], path[:, 1], "-", color="#d62728", linewidth=1.4, alpha=0.85, zorder=4)
    ax.plot(home[0], home[1], "*", color="#ff7f0e", markersize=14,
            markeredgecolor="#7a3d00", zorder=7, label="home")

    pad = 0.06
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.add_patch(
        plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                      fill=False, edgecolor="#333333", linewidth=1.3, zorder=2)
    )
    ax.set_title(
        f"{instance.name}: cycle {outcome.cycle_time:.3f} s, posture {outcome.branch_name}\n"
        f"shaded band = reachable at sigma_min >= {threshold:g} "
        f"(r {inner:.2f}-{outer:.2f} m at mean port height)",
        fontsize=9,
    )
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    ax.grid(alpha=0.2, linewidth=0.5)


def figure_layout(
    instance: Instance,
    outcome: LayoutOutcome,
    evaluator: CycleTimeEvaluator,
    path: str,
) -> None:
    """``layout.png``: the plan, plus the per-station manipulability margin."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.0),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})
    draw_layout(axes[0], instance, outcome, evaluator.arm, evaluator.sigma_min_threshold)

    ax = axes[1]
    if outcome.feasible and outcome.sigma_min is not None:
        names = [instance.machines[i].name for i in instance.sequence]
        positions = np.arange(len(names))
        bars = ax.bar(positions, outcome.sigma_min, color="#4c72b0", alpha=0.85, width=0.62)
        ax.axhline(evaluator.sigma_min_threshold, color="#d62728", linestyle="--", linewidth=1.4,
                   label=f"threshold {evaluator.sigma_min_threshold:g}")
        tightest = int(np.argmin(outcome.sigma_min))
        bars[tightest].set_color("#d62728")
        bars[tightest].set_alpha(0.9)
        for k, value in enumerate(outcome.sigma_min):
            ax.text(k, value + 0.004, f"{value:.3f}", ha="center", fontsize=7.5)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{n}\n#{k+1}" for k, n in enumerate(names)], fontsize=8)
        ax.set_ylabel(r"$\sigma_{min}$ of the position Jacobian (m/rad)")
        ax.set_ylim(0, max(float(outcome.sigma_min.max()) * 1.25,
                           evaluator.sigma_min_threshold * 1.5))
        ax.set_title("Manipulability margin at each station\n"
                     "(red = the binding station)", fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    else:
        ax.text(0.5, 0.5, f"infeasible\n({outcome.rejection})", ha="center", va="center",
                transform=ax.transAxes, fontsize=13, color="#d62728")
        ax.set_axis_off()

    fig.tight_layout()
    _save(fig, path)


# ----------------------------------------------------------------------
# what the arm does
# ----------------------------------------------------------------------


def _box_faces(machine) -> List[List[List[float]]]:
    """The six faces of a machine as polygons, top face first.

    Drawing machines as solid boxes rather than wireframes is what makes the
    3-D views readable: a wireframe gives the eye no occlusion cue, so boxes
    at different depths read as though they intersect.
    """
    corners = machine.corners()
    height = machine.height
    top = [[float(x), float(y), height] for x, y in corners]
    bottom = [[float(x), float(y), 0.0] for x, y in corners]
    faces = [top, bottom]
    for k in range(4):
        a, b = corners[k], corners[(k + 1) % 4]
        faces.append(
            [
                [float(a[0]), float(a[1]), 0.0],
                [float(b[0]), float(b[1]), 0.0],
                [float(b[0]), float(b[1]), height],
                [float(a[0]), float(a[1]), height],
            ]
        )
    return faces


def _draw_scene_3d(
    ax,
    machines,
    arm: Optional[Puma560Arm] = None,
    poses: Sequence[np.ndarray] = (),
    machine_alpha: float = 0.95,
    pose_alphas: Optional[Sequence[float]] = None,
) -> None:
    """The machines and the solid robot, as **one** depth-sorted collection.

    Matplotlib sorts polygons within a ``Poly3DCollection`` but not between
    collections, so a robot in its own collection would either always sit in
    front of the boxes or always behind them, whichever was added last.  One
    collection for the whole scene is what makes the arm disappear behind a
    machine it is genuinely behind and stand in front of one it is not.

    ``poses`` is a sequence of ``(4,)`` configurations, drawn oldest first;
    ``pose_alphas`` fades the earlier ones so a sequence reads as a sweep
    through time rather than a pile of robots.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    polys: List[np.ndarray] = []
    facecolours: List[Sequence[float]] = []
    edgecolours: List[Sequence[float]] = []

    for machine in machines:
        for face in _box_faces(machine):
            face = np.asarray(face, dtype=float)
            polys.append(face)
            rgb = shade(_MACHINE_RGB, np.cross(face[1] - face[0], face[2] - face[0]))
            facecolours.append((*rgb, machine_alpha))
            edgecolours.append((0.133, 0.204, 0.310, machine_alpha))

    if arm is not None and len(poses):
        parts = build_parts(arm)
        if pose_alphas is None:
            pose_alphas = [1.0] * len(poses)
        for q, alpha in zip(poses, pose_alphas):
            posed, colours = posed_polygons(arm, q, parts)
            polys.extend(posed)
            facecolours.extend((*rgb, alpha) for rgb in colours)
            # Each facet is edged in its own fill colour.  A contrasting edge
            # would stripe every cylinder into staves; edging in the fill
            # instead only closes the hairline seams antialiasing leaves
            # between neighbouring facets.
            edgecolours.extend((*rgb, alpha) for rgb in colours)

    if not polys:
        return
    collection = Poly3DCollection(
        polys,
        facecolors=np.asarray(facecolours),
        edgecolors=np.asarray(edgecolours),
        linewidths=0.35,
        zsort="average",
        zorder=1,
    )
    # With one collection holding the whole scene there is nothing left for
    # matplotlib's automatic artist ordering to decide, and leaving it on is
    # actively wrong: it ranks a collection by its *nearest* polygon, so the
    # scene -- which now reaches right up to the camera -- would be drawn last
    # and bury the tool path and the port markers.  Explicit zorder instead,
    # solids at the bottom.
    ax.computed_zorder = False
    ax.add_collection3d(collection)


def _draw_machines_3d(ax, machines, alpha: float = 0.92) -> None:
    """The machines alone -- kept for callers that draw no robot."""
    _draw_scene_3d(ax, machines, machine_alpha=alpha)


def _scene_ceiling(arm: Puma560Arm, q: np.ndarray, margin: float = 0.14) -> float:
    """Top of the z axis: high enough that no part of the arm is clipped."""
    highest = max(
        float(arm.joint_positions(row)[:, 2].max()) for row in np.atleast_2d(q)
    )
    return max(0.95, highest + margin)


def figure_trajectory(
    instance: Instance,
    outcome: LayoutOutcome,
    evaluator: CycleTimeEvaluator,
    path: str,
    poses: int = 8,
) -> None:
    """``trajectory.png``: the 3-D path, the joint angles and the rate profiles."""
    if not outcome.feasible or outcome.configurations is None:
        return

    arm = evaluator.arm
    t, q, q_dot = sample_cycle(
        outcome.configurations, evaluator.velocity_limit,
        evaluator.acceleration_limit, evaluator.dwell, samples_per_segment=70,
        dwell_at=outcome.station_index,
    )
    tool = arm.fk(q)[:, :3]

    fig = plt.figure(figsize=(13.5, 11.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.85, 1.0], hspace=0.16, wspace=0.22)

    # -- 3-D tool path with the arm frozen at intervals -----------------
    ax3d = fig.add_subplot(grid[0, :], projection="3d")
    ax3d.plot(tool[:, 0], tool[:, 1], tool[:, 2], color="#d62728", linewidth=1.7,
              label="tool path", zorder=6)

    # The arm is drawn solid **at the stations**, faintest first, because those
    # are the poses the feasibility criteria actually govern: reach, joint
    # travel, sigma_min and the clearance from every machine are all asserted
    # there and nowhere else.  Drawing evenly spaced instants instead showed
    # mid-move poses, which the criteria say nothing about -- see the
    # swept-path note in the README.  The continuous motion is carried by the
    # tool path and the wrist track below.
    stations = outcome.stations
    if len(stations) > poses:
        stations = stations[np.linspace(0, len(stations) - 1, poses).astype(int)]
    # A steep ramp: the stations sit close together in a converged layout, so a
    # flat one stacks six near-identical translucent robots into a smear.  This
    # way the eye follows the service order to the solid final pose.
    alphas = np.linspace(0.10, 1.0, len(stations)) ** 2.4
    _draw_scene_3d(
        ax3d,
        outcome.machines or instance.machines,
        arm=arm,
        poses=list(stations),
        pose_alphas=alphas,
    )

    wrist = np.stack([arm.joint_positions(row)[3] for row in q])
    ax3d.plot(wrist[:, 0], wrist[:, 1], wrist[:, 2], color="#7a5cc4", linewidth=1.0,
              alpha=0.75, zorder=5, label="wrist centre")

    ports = np.stack([m.port() for m in (outcome.machines or instance.machines)])
    ax3d.scatter(ports[:, 0], ports[:, 1], ports[:, 2], color="#d62728", s=22, depthshade=False,
                 zorder=7, label="loading ports")

    min_x, max_x, min_y, max_y = instance.bounds
    ax3d.set_xlim(min_x, max_x)
    ax3d.set_ylim(min_y, max_y)
    # Tall enough for the elbow: a solid arm that leaves the top of the axes
    # is clipped, not just cropped, and the missing links read as a gap.
    ax3d.set_zlim(0.0, _scene_ceiling(arm, q))
    # The cell is wide and shallow; giving z a larger share than its metres
    # deserve is what makes the arm legible rather than a flat scribble.
    ax3d.set_box_aspect((max_x - min_x, max_y - min_y, 1.15))
    ax3d.set_xlabel("x (m)", fontsize=8)
    ax3d.set_ylabel("y (m)", fontsize=8)
    ax3d.set_zlabel("z (m)", fontsize=8)
    ax3d.view_init(elev=22, azim=-58)
    ax3d.tick_params(labelsize=7)
    ax3d.set_title(
        f"Arm trajectory over one cycle -- {outcome.cycle_time:.3f} s, "
        f"posture {outcome.branch_name} (arm shown at each station, in service order, "
        f"faint to solid)",
        fontsize=10,
    )
    ax3d.legend(fontsize=8, loc="upper left")

    # -- joint angles ---------------------------------------------------
    ax_q = fig.add_subplot(grid[1, 0])
    for j in range(4):
        ax_q.plot(t, np.degrees(q[:, j]), color=_JOINT_COLOURS[j], linewidth=1.5,
                  label=_JOINT_NAMES[j])
    _mark_stations(ax_q, instance, outcome, evaluator)
    ax_q.set_xlabel("time (s)")
    ax_q.set_ylabel("joint angle (deg)")
    ax_q.set_title("Joint angles", fontsize=9)
    ax_q.legend(fontsize=7.5, ncol=2, frameon=False)
    ax_q.grid(alpha=0.25, linewidth=0.5)

    # -- joint rates, with the limits drawn on --------------------------
    ax_v = fig.add_subplot(grid[1, 1])
    for j in range(4):
        ax_v.plot(t, q_dot[:, j], color=_JOINT_COLOURS[j], linewidth=1.5, label=_JOINT_NAMES[j])
        limit = evaluator.velocity_limit[j]
        ax_v.axhline(limit, color=_JOINT_COLOURS[j], linestyle=":", linewidth=0.8, alpha=0.55)
        ax_v.axhline(-limit, color=_JOINT_COLOURS[j], linestyle=":", linewidth=0.8, alpha=0.55)
    _mark_stations(ax_v, instance, outcome, evaluator)
    ax_v.set_xlabel("time (s)")
    ax_v.set_ylabel("joint rate (rad/s)")
    ax_v.set_title("Trapezoidal rate profiles (dotted = each joint's limit)", fontsize=9)
    ax_v.grid(alpha=0.25, linewidth=0.5)

    _save(fig, path)


def _mark_stations(ax, instance: Instance, outcome: LayoutOutcome,
                   evaluator: CycleTimeEvaluator) -> None:
    """Vertical rules where the arm arrives at each station.

    With a retract the tour also holds a lift pose either side of every
    station, so the rule has to be drawn at the arrival that is actually a
    station rather than at every waypoint -- otherwise every label lands a
    lift-and-descend early.
    """
    segments = plan_cycle(outcome.configurations, evaluator.velocity_limit,
                          evaluator.acceleration_limit)
    holds = set(int(k) for k in dwell_indices(outcome.configurations,
                                              outcome.station_index))
    clock = 0.0
    names = [instance.machines[i].name for i in instance.sequence]
    labelled = 0
    for index, segment in enumerate(segments):
        clock += segment.duration
        arrival = index + 1
        if arrival not in holds:
            continue
        ax.axvline(clock, color="#888888", linewidth=0.7, linestyle="--", alpha=0.7)
        if labelled < len(names):
            ax.annotate(names[labelled], xy=(clock, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(2, -9), textcoords="offset points", fontsize=6.5, color="#555555")
            labelled += 1
        clock += evaluator.dwell


# ----------------------------------------------------------------------
# convergence
# ----------------------------------------------------------------------


def figure_convergence(results: Sequence[RunResult], path: str, title: str = "") -> None:
    """``convergence.png``: best-so-far and the feasible share of the population."""
    if not results:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))

    width = max(r.generations for r in results)
    curves = np.full((len(results), width), np.nan)
    shares = np.full((len(results), width), np.nan)
    for k, result in enumerate(results):
        curves[k, : result.generations] = result.best_so_far
        curves[k, result.generations :] = result.best_so_far[-1]
        shares[k, : result.generations] = result.feasible_fraction
        shares[k, result.generations :] = result.feasible_fraction[-1]

    x = np.arange(width)
    for data, ax, label in (
        (curves, axes[0], "best cycle time (s)"),
        (shares, axes[1], "feasible fraction of the population"),
    ):
        with np.errstate(invalid="ignore"):
            median = np.nanmedian(data, axis=0)
            low = np.nanpercentile(data, 25, axis=0)
            high = np.nanpercentile(data, 75, axis=0)
        ax.plot(x, median, color="#4c72b0", linewidth=1.9, label="median")
        ax.fill_between(x, low, high, color="#4c72b0", alpha=0.18, linewidth=0, label="IQR")
        ax.set_xlabel("generation")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.legend(fontsize=8, frameon=False)

    axes[0].set_title("Convergence" + (f" -- {title}" if title else ""), fontsize=10)
    axes[1].set_title("How much of the population is feasible", fontsize=10)
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    _save(fig, path)


# ----------------------------------------------------------------------
# animation
# ----------------------------------------------------------------------


def animate_cycle(
    instance: Instance,
    outcome: LayoutOutcome,
    evaluator: CycleTimeEvaluator,
    path: str,
    frames: int = 120,
    fps: int = 20,
    spin: float = 360.0,
    elevation: float = 22.0,
) -> bool:
    """Write ``cycle.gif``.  Returns False if Pillow is unavailable.

    The camera orbits by ``spin`` degrees over the animation.  A GIF cannot be
    interactive, and a single fixed viewpoint is exactly what makes boxes at
    different depths read as though they overlap; turning the camera gives the
    parallax that resolves them.  ``spin=0`` restores a static viewpoint.
    """
    if not outcome.feasible or outcome.configurations is None:
        return False
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:  # pragma: no cover
        return False

    arm = evaluator.arm
    t, q, _ = sample_cycle(outcome.configurations, evaluator.velocity_limit,
                           evaluator.acceleration_limit, evaluator.dwell,
                           samples_per_segment=40, dwell_at=outcome.station_index)
    picks = np.linspace(0, q.shape[0] - 1, frames).astype(int)
    tool = arm.fk(q)[:, :3]

    fig = plt.figure(figsize=(7.4, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    min_x, max_x, min_y, max_y = instance.bounds
    ceiling = _scene_ceiling(arm, q)

    ports = np.stack([m.port() for m in (outcome.machines or instance.machines)])

    def draw(frame_index: int):
        ax.clear()
        index = picks[frame_index]
        # The robot and the boxes go into one collection, so they depth-sort
        # against each other.  The boxes stay slightly translucent because the
        # tool path is a Line3D and still sorts separately from any collection.
        _draw_scene_3d(
            ax,
            outcome.machines or instance.machines,
            arm=arm,
            poses=[q[index]],
            machine_alpha=0.80,
        )
        ax.scatter(ports[:, 0], ports[:, 1], ports[:, 2], color="#d62728", s=16,
                   depthshade=False, zorder=4)
        ax.plot(tool[: index + 1, 0], tool[: index + 1, 1], tool[: index + 1, 2],
                color="#d62728", linewidth=1.5, alpha=0.95, zorder=3)
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_zlim(0.0, ceiling)
        ax.set_box_aspect((max_x - min_x, max_y - min_y, 1.15))
        azimuth = -58.0 + spin * frame_index / max(len(picks) - 1, 1)
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_xlabel("x (m)", fontsize=8)
        ax.set_ylabel("y (m)", fontsize=8)
        ax.set_zlabel("z (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(
            f"{instance.name}  t = {t[index]:6.3f} s / {outcome.cycle_time:.3f} s"
            f"   (view {azimuth:+.0f} deg)",
            fontsize=10,
        )
        return ()

    animation = FuncAnimation(fig, draw, frames=len(picks), blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  wrote {path}")
    return True


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ----------------------------------------------------------------------
# an actually-rotatable viewer
# ----------------------------------------------------------------------


def export_viewer(
    instance: Instance,
    outcome: LayoutOutcome,
    evaluator: CycleTimeEvaluator,
    path: str,
    samples_per_segment: int = 24,
) -> bool:
    """Write ``cycle.html`` -- drag to orbit, scroll to zoom, scrub the cycle.

    The static figures and the GIF are baked at fixed viewpoints; this one is
    not, which is the only real cure for boxes that look like they overlap.
    See :mod:`arm_study.viewer` for how the page draws itself.
    """
    if not outcome.feasible or outcome.configurations is None:
        return False

    from viewer import write_viewer

    arm = evaluator.arm
    t, q, _ = sample_cycle(
        outcome.configurations, evaluator.velocity_limit, evaluator.acceleration_limit,
        evaluator.dwell, samples_per_segment=samples_per_segment,
        dwell_at=outcome.station_index,
    )
    machines = outcome.machines or instance.machines
    scene = {
        "name": instance.name,
        "cycle": round(float(outcome.cycle_time), 4),
        "posture": outcome.branch_name,
        "bounds": [float(b) for b in instance.bounds],
        "keepOut": float(instance.keep_out_radius),
        "base": [float(instance.base_xy[0]), float(instance.base_xy[1])],
        "boxes": [
            {"name": m.name, "faces": _box_faces(m),
             "order": instance.sequence.index(k) + 1}
            for k, m in enumerate(machines)
        ],
        "ports": [[float(v) for v in m.port()] for m in machines],
        "tool": arm.fk(q)[:, :3].round(5).tolist(),
        # The solid model travels once, in link-local coordinates; only the
        # four joint angles are per frame, and the page rebuilds the link
        # transforms itself.  Baking world geometry per frame would be about
        # a hundred times larger and could not re-light as the camera turns.
        "arm": viewer_payload(arm),
        "q": np.asarray(q, dtype=float).round(6).tolist(),
        "t": np.asarray(t, dtype=float).round(4).tolist(),
        "ceiling": round(_scene_ceiling(arm, q), 4),
    }
    write_viewer(scene, path, f"{instance.name} cycle viewer")
    print(f"  wrote {path}")
    return True
