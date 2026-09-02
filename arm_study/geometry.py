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
from typing import Sequence, Tuple

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
