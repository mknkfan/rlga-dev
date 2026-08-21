"""
Fast, semantics-preserving evaluator for robotic-workcell layouts.

Chromosome rotations are snapped to a multiple of 90 degrees by the decoder,
so the rotated local corner sets for the five possible angles are precomputed
once per machine and a decode reduces to a translation.

``LayoutEvaluator.fitness`` also counts objective-function evaluations
(``n_evals``), including calls made from inside local search, so runs can be
compared at an equal evaluation budget.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.datastruct import Machine, Point

# The decoder snaps rotations to round(raw / 90) * 90 for raw in [0, 360),
# which can only produce these five values.  360 is kept distinct from 0
# because cos/sin(2*pi) are not bit-identical to cos/sin(0).
SNAPPED_ROTATIONS: Tuple[float, ...] = (0.0, 90.0, 180.0, 270.0, 360.0)

XY = Tuple[float, float]


def local_corners(machine: Machine) -> List[XY]:
    """Machine corners in body coordinates, in the same order as ``Machine.get_corners``."""
    w, h = machine.width, machine.height
    if machine.shape == "rectangle":
        return [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    # L-shape
    cw, ch = machine.l_cutout_width, machine.l_cutout_height
    return [
        (-w / 2, -h / 2),
        (w / 2, -h / 2),
        (w / 2, h / 2 - ch),
        (w / 2 - cw, h / 2 - ch),
        (w / 2 - cw, h / 2),
        (-w / 2, h / 2),
    ]


def _rotate(corners: Sequence[XY], degrees: float) -> List[XY]:
    """Rotate body-frame corners, reproducing ``Machine.get_corners`` arithmetic."""
    rad = np.radians(degrees)
    cos_r, sin_r = float(np.cos(rad)), float(np.sin(rad))
    return [(x * cos_r - y * sin_r, x * sin_r + y * cos_r) for x, y in corners]


def _point_in_polygon(px: float, py: float, poly: Sequence[XY]) -> bool:
    """Ray-casting test mirroring ``CollisionDetector.point_in_polygon``."""
    inside = False
    n = len(poly)
    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if py > min(p1y, p2y) and py <= max(p1y, p2y) and px <= max(p1x, p2x):
            # p1y == p2y is impossible here: it would force py > py.
            xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or px <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def _orientation(px: float, py: float, qx: float, qy: float, rx: float, ry: float) -> int:
    val = (qy - py) * (rx - qx) - (qx - px) * (ry - qy)
    if val == 0:
        return 0
    return 1 if val > 0 else 2


def _on_segment(px: float, py: float, qx: float, qy: float, rx: float, ry: float) -> bool:
    return (
        qx <= max(px, rx)
        and qx >= min(px, rx)
        and qy <= max(py, ry)
        and qy >= min(py, ry)
    )


def _segments_intersect(
    p1: XY, q1: XY, p2: XY, q2: XY
) -> bool:
    """Mirrors ``CollisionDetector.line_segments_intersect``."""
    o1 = _orientation(p1[0], p1[1], q1[0], q1[1], p2[0], p2[1])
    o2 = _orientation(p1[0], p1[1], q1[0], q1[1], q2[0], q2[1])
    o3 = _orientation(p2[0], p2[1], q2[0], q2[1], p1[0], p1[1])
    o4 = _orientation(p2[0], p2[1], q2[0], q2[1], q1[0], q1[1])

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and _on_segment(p1[0], p1[1], p2[0], p2[1], q1[0], q1[1]):
        return True
    if o2 == 0 and _on_segment(p1[0], p1[1], q2[0], q2[1], q1[0], q1[1]):
        return True
    if o3 == 0 and _on_segment(p2[0], p2[1], p1[0], p1[1], q2[0], q2[1]):
        return True
    if o4 == 0 and _on_segment(p2[0], p2[1], q1[0], q1[1], q2[0], q2[1]):
        return True
    return False


def _polygons_intersect(poly1: Sequence[XY], poly2: Sequence[XY]) -> bool:
    """Mirrors ``CollisionDetector.polygons_intersect`` (narrow phase)."""
    for px, py in poly1:
        if _point_in_polygon(px, py, poly2):
            return True
    for px, py in poly2:
        if _point_in_polygon(px, py, poly1):
            return True
    n1, n2 = len(poly1), len(poly2)
    for i in range(n1):
        a1 = poly1[i]
        a2 = poly1[(i + 1) % n1]
        for j in range(n2):
            if _segments_intersect(a1, a2, poly2[j], poly2[(j + 1) % n2]):
                return True
    return False


class LayoutEvaluator:
    """Evaluates a layout chromosome ``[x1, y1, rot1, x2, y2, rot2, ...]``.

    Parameters
    ----------
    machines, sequence, robot_position, workspace_bounds
        Same problem description as the GA classes.
    compactness_weight, accessibility_weight
        Weights alpha/beta of the two auxiliary fitness terms, exposed here
        so they can be swept by the weight-sensitivity study.
    """

    def __init__(
        self,
        machines: List[Machine],
        sequence: List[int],
        robot_position: Point,
        workspace_bounds: Tuple[float, float, float, float],
        compactness_weight: float = 0.1,
        accessibility_weight: float = 0.05,
    ):
        self.machines = machines
        self.sequence = list(sequence)
        self.robot_position = robot_position
        self.workspace_bounds = workspace_bounds
        self.compactness_weight = compactness_weight
        self.accessibility_weight = accessibility_weight

        self.n_machines = len(machines)
        self.robot_xy: XY = (float(robot_position.x), float(robot_position.y))
        self.min_x, self.max_x, self.min_y, self.max_y = (float(v) for v in workspace_bounds)

        # Precompute rotated corner sets: [machine][rotation_index] -> corners
        self._corners_by_rot: List[List[List[XY]]] = []
        self._aabb_by_rot: List[List[Tuple[float, float, float, float]]] = []
        self._access_by_rot: List[List[XY]] = []
        for machine in machines:
            base = local_corners(machine)
            per_rot_corners, per_rot_aabb, per_rot_access = [], [], []
            for degrees in SNAPPED_ROTATIONS:
                rotated = _rotate(base, degrees)
                per_rot_corners.append(rotated)
                xs = [c[0] for c in rotated]
                ys = [c[1] for c in rotated]
                per_rot_aabb.append((min(xs), max(xs), min(ys), max(ys)))
                rad = np.radians(degrees)
                cos_r, sin_r = float(np.cos(rad)), float(np.sin(rad))
                ax, ay = float(machine.access_point.x), float(machine.access_point.y)
                per_rot_access.append((ax * cos_r - ay * sin_r, ax * sin_r + ay * cos_r))
            self._corners_by_rot.append(per_rot_corners)
            self._aabb_by_rot.append(per_rot_aabb)
            self._access_by_rot.append(per_rot_access)

        # Machine ids in visiting order -> index into the machine list.
        id_to_index: Dict[int, int] = {}
        for idx, machine in enumerate(machines):
            id_to_index.setdefault(machine.id, idx)
        self._sequence_indices = [id_to_index[mid] for mid in self.sequence]

        # The clearance term excludes the machine itself *by id*, as in the
        self._other_indices: List[List[int]] = [
            [j for j in range(self.n_machines) if machines[j].id != machines[i].id]
            for i in range(self.n_machines)
        ]

        # Objective-function evaluation counter.
        self.n_evals = 0

    # ------------------------------------------------------------------
    # decoding
    # ------------------------------------------------------------------

    def rotation_indices(self, chromosome: np.ndarray) -> List[int]:
        """Snap each gene triple's angle to one of the five allowed rotations."""
        return [
            int(round(float(chromosome[3 * i + 2]) % 360.0 / 90.0))
            for i in range(self.n_machines)
        ]

    def decode_to_machines(self, chromosome: np.ndarray) -> List[Machine]:
        """Decode into ``Machine`` objects (for plotting / reporting only)."""
        import copy

        machines = copy.deepcopy(self.machines)
        for i, machine in enumerate(machines):
            idx = i * 3
            machine.position.x = chromosome[idx]
            machine.position.y = chromosome[idx + 1]
            raw_rot = chromosome[idx + 2] % 360
            machine.rotation = round(raw_rot / 90) * 90
        return machines

    # ------------------------------------------------------------------
    # fitness
    # ------------------------------------------------------------------

    def fitness(self, chromosome: np.ndarray) -> float:
        """Total fitness (lower is better); ``inf`` for infeasible layouts.

        Increments the evaluation counter.  Feasibility rules, in order:
        no machine may cover the robot origin, no two machines may overlap,
        and every machine must lie inside the workspace.
        """
        self.n_evals += 1

        n = self.n_machines
        centres: List[XY] = []
        corners: List[List[XY]] = []
        aabbs: List[Tuple[float, float, float, float]] = []
        access: List[XY] = []

        for i in range(n):
            base = 3 * i
            cx = float(chromosome[base])
            cy = float(chromosome[base + 1])
            r = int(round(float(chromosome[base + 2]) % 360.0 / 90.0))
            centres.append((cx, cy))
            corners.append([(x + cx, y + cy) for x, y in self._corners_by_rot[i][r]])
            lo_x, hi_x, lo_y, hi_y = self._aabb_by_rot[i][r]
            aabbs.append((lo_x + cx, hi_x + cx, lo_y + cy, hi_y + cy))
            ax, ay = self._access_by_rot[i][r]
            access.append((ax + cx, ay + cy))

        # --- constraint 1: no machine may cover the robot origin (0, 0) -----
        for i in range(n):
            lo_x, hi_x, lo_y, hi_y = aabbs[i]
            if lo_x <= 0.0 <= hi_x and lo_y <= 0.0 <= hi_y:
                if _point_in_polygon(0.0, 0.0, corners[i]):
                    return float("inf")

        # --- constraint 2: pairwise collisions (AABB broad phase first) -----
        for i in range(n):
            ilo_x, ihi_x, ilo_y, ihi_y = aabbs[i]
            for j in range(i + 1, n):
                jlo_x, jhi_x, jlo_y, jhi_y = aabbs[j]
                if ihi_x < jlo_x or jhi_x < ilo_x or ihi_y < jlo_y or jhi_y < ilo_y:
                    continue  # bounding boxes disjoint => polygons cannot touch
                if _polygons_intersect(corners[i], corners[j]):
                    return float("inf")

        # --- constraint 3: workspace bounds ---------------------------------
        for lo_x, hi_x, lo_y, hi_y in aabbs:
            if not (
                self.min_x <= lo_x
                and hi_x <= self.max_x
                and self.min_y <= lo_y
                and hi_y <= self.max_y
            ):
                return float("inf")

        return (
            self.total_distance_from(access)
            + self.compactness_weight * self._span(centres)
            - self.accessibility_weight * self._clearance(access, corners)
        )

    def fitness_components(self, chromosome: np.ndarray) -> Optional[Dict[str, float]]:
        """Individual fitness terms for a feasible layout, else ``None``.

        Used by the weight-sensitivity study: the three terms are computed
        once and recombined for many (alpha, beta) pairs.
        """
        if not np.isfinite(self.fitness(chromosome)):
            return None
        n = self.n_machines
        centres, corners, access = [], [], []
        for i in range(n):
            base = 3 * i
            cx = float(chromosome[base])
            cy = float(chromosome[base + 1])
            r = int(round(float(chromosome[base + 2]) % 360.0 / 90.0))
            centres.append((cx, cy))
            corners.append([(x + cx, y + cy) for x, y in self._corners_by_rot[i][r]])
            ax, ay = self._access_by_rot[i][r]
            access.append((ax + cx, ay + cy))
        return {
            "distance": self.total_distance_from(access),
            "span": self._span(centres),
            "clearance": self._clearance(access, corners),
        }

    # ------------------------------------------------------------------
    # fitness components (also exposed individually for the sensitivity study)
    # ------------------------------------------------------------------

    def total_distance_from(self, access: List[XY]) -> float:
        """Closed robot tour: origin -> access points in sequence -> origin."""
        path = [self.robot_xy]
        for idx in self._sequence_indices:
            path.append(access[idx])
        path.append(self.robot_xy)

        total = 0.0
        for k in range(len(path) - 1):
            dx = path[k][0] - path[k + 1][0]
            dy = path[k][1] - path[k + 1][1]
            total += math.sqrt(dx * dx + dy * dy)
        return total

    @staticmethod
    def _span(centres: List[XY]) -> float:
        """Bounding-box span of machine centres (the compactness penalty term)."""
        if not centres:
            return 0.0
        xs = [c[0] for c in centres]
        ys = [c[1] for c in centres]
        return (max(xs) - min(xs)) + (max(ys) - min(ys))

    def _clearance(self, access: List[XY], corners: List[List[XY]]) -> float:
        """Sum over machines of the distance to the nearest *other* machine corner."""
        total = 0.0
        n = len(access)
        for i in range(n):
            ax, ay = access[i]
            best = float("inf")
            for j in self._other_indices[i]:
                for cx, cy in corners[j]:
                    dx = ax - cx
                    dy = ay - cy
                    d = math.sqrt(dx * dx + dy * dy)
                    if d < best:
                        best = d
            total += best if best != float("inf") else 10.0
        return total

    def total_distance(self, machines: List[Machine]) -> float:
        """Robot tour length for an already-decoded machine list."""
        access_by_index = [m.get_access_point_world() for m in machines]
        access = [(float(p.x), float(p.y)) for p in access_by_index]
        return self.total_distance_from(access)

    # ------------------------------------------------------------------

    def reset_counter(self) -> None:
        self.n_evals = 0
