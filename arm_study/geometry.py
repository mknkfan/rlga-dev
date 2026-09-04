"""
Planar geometry of the workcell floor: machines, their loading ports, and the
three purely geometric feasibility tests.

A machine is an upright box standing on the floor.  Its *footprint* is a
rectangle that may be rotated about the vertical axis; its *top surface* is at
height ``height`` and carries the loading port the robot reaches into.  The
port sits at a fixed place in the machine's own frame, so rotating a machine
moves the point the robot must reach -- which is why the rotation gene is not
decorative here.

Everything is metres and degrees-in / radians-out at the boundary: the layout
genes are in degrees because that is what the GA mutates, and the robot code
is handed radians.

Nothing in this module knows about the robot.  Reachability and manipulability
live in :mod:`arm_study.robot`; this file answers only "do the boxes fit".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

#: (min_x, max_x, min_y, max_y) of the cell floor.
Bounds = Tuple[float, float, float, float]


@dataclass
class Machine:
    """One machine: a rotatable box with a loading port on its top face.

    ``width`` and ``depth`` are the footprint extents along the machine's own
    x and y axes; ``height`` is the top surface the robot picks from.
    ``port_local`` is the port's position in that same local frame, so the
    world port moves when the machine is rotated.
    """

    name: str
    width: float
    depth: float
    height: float
    x: float = 0.0
    y: float = 0.0
    rotation: float = 0.0  # degrees, about the vertical axis
    port_local: Tuple[float, float] = (0.0, 0.0)

    # -- derived geometry ----------------------------------------------

    @property
    def footprint_radius(self) -> float:
        """Half-diagonal: the machine fits in this disc whatever its rotation.

        Used to keep the position genes inside the cell without having to
        re-check the corners for every candidate rotation.
        """
        return 0.5 * math.hypot(self.width, self.depth)

    def _rotation_matrix(self) -> np.ndarray:
        theta = math.radians(self.rotation)
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c, -s], [s, c]], dtype=float)

    def corners(self) -> np.ndarray:
        """The four footprint corners in world coordinates, shape ``(4, 2)``."""
        half_w, half_d = self.width / 2.0, self.depth / 2.0
        local = np.array(
            [[-half_w, -half_d], [half_w, -half_d], [half_w, half_d], [-half_w, half_d]],
            dtype=float,
        )
        return local @ self._rotation_matrix().T + np.array([self.x, self.y])

    def port(self) -> np.ndarray:
        """World position ``(x, y, z)`` of the loading port on the top face."""
        local = np.asarray(self.port_local, dtype=float)
        planar = self._rotation_matrix() @ local + np.array([self.x, self.y])
        return np.array([planar[0], planar[1], self.height], dtype=float)

    def copy_at(self, x: float, y: float, rotation: float) -> "Machine":
        """The same machine placed somewhere else -- the decode step's product."""
        return Machine(
            name=self.name,
            width=self.width,
            depth=self.depth,
            height=self.height,
            x=float(x),
            y=float(y),
            rotation=float(rotation),
            port_local=self.port_local,
        )


# ----------------------------------------------------------------------
# the three geometric tests
# ----------------------------------------------------------------------


def _project(corners: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    """Interval covered by ``corners`` when projected onto ``axis``."""
    dots = corners @ axis
    return float(dots.min()), float(dots.max())


def rectangles_overlap(a: Machine, b: Machine, tolerance: float = 0.0) -> bool:
    """Do two (possibly rotated) footprints intersect?

    Separating-axis test.  Two convex polygons are disjoint exactly when some
    axis normal to an edge of one of them separates their projections, so for
    two rectangles four axes settle it.  ``tolerance`` inflates both boxes, so
    a positive value enforces a clearance gap rather than mere non-overlap.
    """
    corners_a, corners_b = a.corners(), b.corners()

    # A cheap disc test first: most random pairs are far apart, and this skips
    # the projections for them.
    centre_gap = math.hypot(a.x - b.x, a.y - b.y)
    if centre_gap > a.footprint_radius + b.footprint_radius + tolerance:
        return False

    for corners in (corners_a, corners_b):
        edges = np.roll(corners, -1, axis=0) - corners
        for edge in edges[:2]:  # a rectangle's two distinct edge directions
            norm = math.hypot(edge[0], edge[1])
            if norm < 1e-12:
                continue
            axis = np.array([-edge[1], edge[0]]) / norm  # outward normal
            lo_a, hi_a = _project(corners_a, axis)
            lo_b, hi_b = _project(corners_b, axis)
            if hi_a + tolerance <= lo_b or hi_b + tolerance <= lo_a:
                return False
    return True


def within_bounds(machine: Machine, bounds: Bounds) -> bool:
    """Does the whole footprint lie inside the cell?"""
    min_x, max_x, min_y, max_y = bounds
    corners = machine.corners()
    return bool(
        (corners[:, 0] >= min_x).all()
        and (corners[:, 0] <= max_x).all()
        and (corners[:, 1] >= min_y).all()
        and (corners[:, 1] <= max_y).all()
    )


def clears_circle(machine: Machine, centre: Sequence[float], radius: float) -> bool:
    """Is the footprint entirely outside a keep-out disc?

    The disc is the robot's own column: the pedestal plus the volume the base
    and shoulder sweep through.  A machine standing inside it would foul the
    robot however good the cycle time looked.
    """
    theta = math.radians(machine.rotation)
    c, s = math.cos(theta), math.sin(theta)
    # Work in the machine's frame, where the box is axis-aligned and the
    # closest-point calculation is a clamp.
    dx = centre[0] - machine.x
    dy = centre[1] - machine.y
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy

    half_w, half_d = machine.width / 2.0, machine.depth / 2.0
    nearest_x = max(-half_w, min(half_w, local_x))
    nearest_y = max(-half_d, min(half_d, local_y))
    return math.hypot(local_x - nearest_x, local_y - nearest_y) >= radius


def box_frames(machines: Sequence[Machine]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The machines as boxes ready for :func:`points_to_boxes_distance`.

    Returns ``(centres, half_extents, rotations)``: the box centre in 3-D (the
    footprint centre, half the height up), the three half-extents in the
    machine's own frame, and ``(cos, sin)`` of each rotation.  Packing them
    once per layout is what lets the clearance test be a single broadcast.
    """
    centres = np.array([[m.x, m.y, m.height / 2.0] for m in machines], dtype=float)
    half = np.array(
        [[m.width / 2.0, m.depth / 2.0, m.height / 2.0] for m in machines], dtype=float
    )
    angles = np.radians([m.rotation for m in machines])
    return centres, half, np.stack([np.cos(angles), np.sin(angles)], axis=-1)


def points_to_boxes_distance(
    points: np.ndarray,
    centres: np.ndarray,
    half_extents: np.ndarray,
    rotations: np.ndarray,
) -> np.ndarray:
    """Signed distance from each point to each machine's solid, in metres.

    ``points`` is ``(..., 3)``; the result is ``(..., n_machines)``.  Positive
    outside the box, negative inside -- the negative branch is the usual
    box signed-distance field, which is what lets a caller report how deep an
    interpenetration is rather than only that there is one.
    """
    points = np.asarray(points, dtype=float)
    delta = points[..., None, :] - centres          # (..., n_machines, 3)
    cos, sin = rotations[:, 0], rotations[:, 1]
    local = np.stack(
        [
            cos * delta[..., 0] + sin * delta[..., 1],
            -sin * delta[..., 0] + cos * delta[..., 1],
            delta[..., 2],
        ],
        axis=-1,
    )
    excess = np.abs(local) - half_extents
    outside = np.linalg.norm(np.maximum(excess, 0.0), axis=-1)
    inside = np.minimum(excess.max(axis=-1), 0.0)
    return outside + inside


def points_over_boxes(
    points: np.ndarray,
    centres: np.ndarray,
    half_extents: np.ndarray,
    rotations: np.ndarray,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Which points sit inside a machine's footprint at or above its top face.

    ``points`` is ``(..., 3)``; the result is ``(..., n_machines)`` of bools.
    This is the robot's approach corridor: a tool descending vertically onto a
    top face is above that machine, not inside it.
    """
    points = np.asarray(points, dtype=float)
    delta = points[..., None, :] - centres
    cos, sin = rotations[:, 0], rotations[:, 1]
    local_x = cos * delta[..., 0] + sin * delta[..., 1]
    local_y = -sin * delta[..., 0] + cos * delta[..., 1]
    return (
        (np.abs(local_x) <= half_extents[:, 0] + tolerance)
        & (np.abs(local_y) <= half_extents[:, 1] + tolerance)
        & (delta[..., 2] >= half_extents[:, 2] - tolerance)
    )


def layout_is_placeable(
    machines: Sequence[Machine],
    bounds: Bounds,
    base_xy: Sequence[float],
    keep_out_radius: float,
    clearance: float = 0.0,
) -> bool:
    """All three geometric tests, in increasing order of cost.

    Returns ``True`` only if every machine is inside the cell, clear of the
    robot's column, and disjoint from every other machine.
    """
    for machine in machines:
        if not within_bounds(machine, bounds):
            return False
        if not clears_circle(machine, base_xy, keep_out_radius):
            return False

    for i in range(len(machines)):
        for j in range(i + 1, len(machines)):
            if rectangles_overlap(machines[i], machines[j], clearance):
                return False
    return True


# ----------------------------------------------------------------------
# projecting an unplaceable layout back onto the placeable set
# ----------------------------------------------------------------------


def _separating_axes(a: Machine, b: Machine) -> np.ndarray:
    """The four axes that settle :func:`rectangles_overlap` for two boxes.

    A rectangle's two distinct edge normals, for each of the pair.  Returned
    as unit vectors, shape ``(4, 2)``.
    """
    axes = []
    for corners in (a.corners(), b.corners()):
        edges = np.roll(corners, -1, axis=0) - corners
        for edge in edges[:2]:
            norm = math.hypot(edge[0], edge[1])
            if norm < 1e-12:
                continue
            axes.append(np.array([-edge[1], edge[0]]) / norm)
    return np.asarray(axes, dtype=float)


def overlap_depth(a: Machine, b: Machine, tolerance: float = 0.0) -> Tuple[float, np.ndarray]:
    """How far apart to push two footprints, and along which axis.

    The minimum translation that makes :func:`rectangles_overlap` return
    ``False``: the separating axis needing the smallest push, and how much.
    A non-positive depth means the boxes already clear each other by
    ``tolerance``, so no push is needed.

    Returned as ``(depth, axis)`` with ``axis`` oriented so that moving ``b``
    along ``+axis`` (and ``a`` along ``-axis``) separates them.
    """
    corners_a, corners_b = a.corners(), b.corners()
    best_depth = math.inf
    best_axis = np.array([1.0, 0.0])
    for axis in _separating_axes(a, b):
        lo_a, hi_a = _project(corners_a, axis)
        lo_b, hi_b = _project(corners_b, axis)
        # Push b positive along the axis, or push b negative -- whichever is
        # the shorter move on this axis.
        push_b_positive = hi_a + tolerance - lo_b
        push_b_negative = hi_b + tolerance - lo_a
        if push_b_positive <= push_b_negative:
            depth, oriented = push_b_positive, axis
        else:
            depth, oriented = push_b_negative, -axis
        if depth <= 0.0:
            # This axis already separates them: the pair is disjoint.
            return 0.0, oriented
        if depth < best_depth:
            best_depth, best_axis = depth, oriented
    return best_depth, best_axis


def keep_out_shortfall(
    machine: Machine, centre: Sequence[float], radius: float
) -> float:
    """How far short of the keep-out disc a footprint falls, in metres.

    Zero when the machine already clears the disc.  The same nearest-point
    calculation :func:`clears_circle` makes, reported rather than thresholded.
    """
    theta = math.radians(machine.rotation)
    c, s = math.cos(theta), math.sin(theta)
    dx = centre[0] - machine.x
    dy = centre[1] - machine.y
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy

    half_w, half_d = machine.width / 2.0, machine.depth / 2.0
    nearest_x = max(-half_w, min(half_w, local_x))
    nearest_y = max(-half_d, min(half_d, local_y))
    gap = math.hypot(local_x - nearest_x, local_y - nearest_y)
    return max(0.0, radius - gap)


def repair_placement(
    machines: Sequence[Machine],
    bounds: Bounds,
    base_xy: Sequence[float],
    keep_out_radius: float,
    clearance: float = 0.0,
    margin: float = 0.005,
    max_iters: int = 12,
) -> List[Machine]:
    """Project an unplaceable layout back onto the placeable set.

    Translation only: rotations are left exactly as the GA drew them, so the
    repair moves machines without touching the gene that decides where the
    loading port ends up.  Three moves, applied in a loop because they
    interact -- pushing a machine off the robot's column can push it into a
    neighbour, and clipping it back into the cell can undo either:

    1. clip each centre into the cell, inset by the footprint half-diagonal
       as :meth:`Instance.gene_bounds` does, and by ``margin`` again;
    2. push any machine fouling the robot's column radially outward until it
       clears, by the shortfall :func:`keep_out_shortfall` reports;
    3. separate each overlapping pair along the axis needing the smallest
       move -- the minimum translation of :func:`overlap_depth`, which is the
       exact inverse of the test :func:`rectangles_overlap` applies.

    ``margin`` is projected *past* each constraint rather than onto it.  A
    layout pushed exactly onto the boundary is placeable only by measure zero
    and is brittle downstream -- the arm still has to reach into it without
    fouling anything -- so the slack is what makes a repaired layout usable
    rather than merely legal.

    Identity on the placeable set: a layout that already passes
    :func:`layout_is_placeable` is returned untouched, margin or no margin, so
    this is a projection rather than a deformation of the whole space.  It is
    also objective-blind -- no cycle time is ever evaluated -- so it repairs
    the representation without doing local search on the thing being
    optimised.

    The loop is bounded.  A crowded instance can be unrepairable in this many
    passes and the caller gets back whatever the last pass produced; it is not
    promised to be placeable, only closer, and the feasibility test still
    decides.
    """
    placed = [m.copy_at(m.x, m.y, m.rotation) for m in machines]
    if layout_is_placeable(placed, bounds, base_xy, keep_out_radius, clearance):
        return placed

    n = len(placed)
    min_x, max_x, min_y, max_y = bounds
    base = np.asarray(base_xy, dtype=float)
    tolerance = clearance + margin

    # Rotation is fixed for the whole repair, so a machine's corner offsets
    # from its own centre -- and the axes its edges define -- are constants.
    # Projecting the offsets onto every axis once turns each pair test in the
    # loop below into a handful of adds.
    offsets = [m.corners() - np.array([m.x, m.y]) for m in placed]
    angles = np.radians([m.rotation for m in placed])
    axes = np.empty((2 * n, 2), dtype=float)
    axes[0::2, 0], axes[0::2, 1] = np.cos(angles), np.sin(angles)
    axes[1::2, 0], axes[1::2, 1] = -np.sin(angles), np.cos(angles)

    spans = np.stack([offset @ axes.T for offset in offsets])   # (n, 4, 2n)
    extent_lo, extent_hi = spans.min(axis=1), spans.max(axis=1)  # (n, 2n)

    centres = np.array([[m.x, m.y] for m in placed], dtype=float)
    radii = np.array([m.footprint_radius for m in placed])
    # The half-diagonal inset of Instance.gene_bounds, plus the same slack the
    # other two constraints are given -- a centre clipped exactly onto the
    # inset leaves the corner on the wall, where within_bounds is deciding on
    # rounding error in the rotation.  A machine too big for the cell has no
    # inset to clip to, and np.clip with lower > upper would silently return
    # the upper; parking it in the middle is the honest degenerate answer, and
    # the feasibility test still rejects the layout.
    inset = radii + margin
    lower = np.stack([min_x + inset, min_y + inset], axis=1)
    upper = np.stack([max_x - inset, max_y - inset], axis=1)
    midpoint = np.array([0.5 * (min_x + max_x), 0.5 * (min_y + max_y)])
    too_big = lower > upper
    lower = np.where(too_big, midpoint, lower)
    upper = np.where(too_big, midpoint, upper)

    half = np.array([[m.width / 2.0, m.depth / 2.0] for m in placed])
    cos_t, sin_t = np.cos(angles), np.sin(angles)

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    reach = radii[:, None] + radii[None, :] + tolerance

    for _ in range(max_iters):
        moved = False
        np.clip(centres, lower, upper, out=centres)

        for i in range(n):
            dx, dy = base - centres[i]
            local_x = cos_t[i] * dx + sin_t[i] * dy
            local_y = -sin_t[i] * dx + cos_t[i] * dy
            nearest_x = min(max(local_x, -half[i, 0]), half[i, 0])
            nearest_y = min(max(local_y, -half[i, 1]), half[i, 1])
            shortfall = keep_out_radius + margin - math.hypot(
                local_x - nearest_x, local_y - nearest_y
            )
            if shortfall > 0.0:
                away = centres[i] - base
                norm = math.hypot(away[0], away[1])
                # A machine sitting dead on the column has no outward
                # direction of its own; any ray will do.
                direction = away / norm if norm > 1e-12 else np.array([1.0, 0.0])
                centres[i] += direction * shortfall
                moved = True

        projection = centres @ axes.T                            # (n, 2n)
        for i, j in pairs:
            if math.hypot(*(centres[i] - centres[j])) > reach[i, j]:
                continue
            k = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
            lo_a, hi_a = extent_lo[i, k] + projection[i, k], extent_hi[i, k] + projection[i, k]
            lo_b, hi_b = extent_lo[j, k] + projection[j, k], extent_hi[j, k] + projection[j, k]

            # On each axis, the shorter of the two ways to pull them apart.
            forward, backward = hi_a + tolerance - lo_b, hi_b + tolerance - lo_a
            depths = np.minimum(forward, backward)
            if depths.min() <= 0.0:
                continue  # this axis already separates them
            best = int(np.argmin(depths))
            axis = axes[k[best]] if forward[best] <= backward[best] else -axes[k[best]]

            shift = 0.5 * depths[best] * axis
            centres[i] -= shift
            centres[j] += shift
            projection[i] = centres[i] @ axes.T
            projection[j] = centres[j] @ axes.T
            moved = True

        if not moved:
            break

    # A final clip, so the last separation pass cannot leave a machine
    # hanging over the cell wall.
    np.clip(centres, lower, upper, out=centres)
    for machine, centre in zip(placed, centres):
        machine.x, machine.y = float(centre[0]), float(centre[1])
    return placed
