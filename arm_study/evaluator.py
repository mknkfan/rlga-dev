"""
The objective: cycle time of a layout, and the tests that decide whether a
layout has a cycle time at all.

Feasibility
-----------
A layout is feasible only if all five hold.  The first two are geometric and
cheap; the rest need the robot and are only reached once those pass, which is
what keeps 60 000 evaluations per run affordable.

1. **Placement** -- every footprint inside the cell, clear of the robot's
   column by ``keep_out_radius``, and disjoint from every other footprint with
   at least ``clearance`` between them.
2. **Reachability** -- the arm can put its tool on every loading port with a
   vertical approach, inside every joint's travel, and can retract from it.
3. **Conditioning** -- at every port, ``sigma_min`` of the position Jacobian
   is at least ``sigma_min_threshold``.  This is the anti-singularity test:
   it rejects poses where the elbow is locked out or folded, near which the
   joint rates needed for a given tool motion blow up.
4. **Clearance** -- no part of the robot comes within ``arm_clearance`` of any
   machine, at every station and (with ``check_path``) at samples along every
   move.  The arm is bounded by the chain of spheres
   :func:`armmesh.collision_spheres` measures off the solid model that gets
   drawn, so a layout that passes cannot show the robot inside a box.
5. **One posture for the whole cycle** -- a single IK branch must satisfy 2 to
   4 at *every* station.  A real cell does not reconfigure the arm mid-cycle;
   allowing a different posture at each station would let the optimiser buy
   cycle time with elbow flips that no controller would execute.

The motion
----------
Each station is bracketed by a **lift pose**: the same tool position raised to
``retract`` metres above the tallest machine in the cell.  The cycle is
therefore descend / dwell / retract / traverse, which is what a real cell does.
It is not decoration: with direct point-to-point moves the tool interpolates in
joint space between two ports and drags the forearm straight through whatever
machine stands between them, and in a cell this tightly packed criterion 4 over
the whole cycle is then not merely strict but *unsatisfiable* -- the GA finds
no feasible layout at all.  With the retract it is satisfied by 99% of the
layouts that clear the stations.  ``retract=None`` restores the direct moves,
for comparison against the earlier form of this study.

Anything that fails scores ``+inf``.  That is a hard wall rather than a
penalty on purpose: a penalty would need a scale relating metres of overlap to
seconds of cycle time, and any choice of that scale is an unstated preference
that shows up in the answer.

The objective
-------------
Among the branches that pass, the layout is scored by the fastest one -- the
cell would be commissioned in its best working posture.  Cycle time itself is
the trapezoidal total from :mod:`arm_study.trajectory`: home, every station in
process order with its lift poses either side, back home, plus a dwell at each
station.

Counting
--------
``n_evals`` counts calls to :meth:`fitness`, so the GA's evaluation budget is
measured in the same currency regardless of how much work each call did.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from armmesh import collision_spheres
from geometry import (
    Machine,
    box_frames,
    layout_is_placeable,
    points_over_boxes,
    points_to_boxes_distance,
)
from problems import Instance
from robot import BRANCHES, BRANCH_NAMES, Puma560Arm
from trajectory import cycle_time, plan_cycle

#: Why a layout was rejected -- reported in the results so a run's failures
#: can be attributed rather than guessed at.
REJECTIONS: Tuple[str, ...] = (
    "placement", "reach", "limits", "sigma_min", "clearance", "path", "none",
)


@dataclass
class LayoutOutcome:
    """Everything the evaluator learned about one layout.

    :meth:`CycleTimeEvaluator.fitness` throws all but the cycle time away; the
    reporting and plotting code asks for this instead.
    """

    feasible: bool
    cycle_time: float
    rejection: str
    branch: Optional[Tuple[int, int]] = None
    configurations: Optional[np.ndarray] = None
    sigma_min: Optional[np.ndarray] = None
    waypoints: Optional[np.ndarray] = None
    machines: Optional[List[Machine]] = None
    #: Indices into ``configurations`` that are stations -- where the arm
    #: dwells.  With a retract the tour also holds a lift pose either side of
    #: each station, and those are passed through, not held at.
    station_index: Optional[np.ndarray] = None

    @property
    def branch_name(self) -> str:
        return BRANCH_NAMES.get(self.branch, "-") if self.branch else "-"

    @property
    def stations(self) -> Optional[np.ndarray]:
        """The station configurations alone, without home or the lift poses."""
        if self.configurations is None:
            return None
        if self.station_index is None:
            return self.configurations[1:-1]
        return self.configurations[self.station_index]


class CycleTimeEvaluator:
    """Chromosome -> cycle time, with the feasibility wall in front of it."""

    def __init__(
        self,
        instance: Instance,
        arm: Optional[Puma560Arm] = None,
        sigma_min_threshold: float = 0.14,
        dwell: float = 0.30,
        check_path: bool = True,
        path_samples: int = 12,
        check_arm: bool = True,
        arm_clearance: float = 0.005,
        retract: float = 0.06,
    ):
        self.instance = instance
        self.arm = arm or Puma560Arm(base=instance.base_xy)
        self.sigma_min_threshold = float(sigma_min_threshold)
        self.dwell = float(dwell)
        self.check_path = bool(check_path)
        self.path_samples = int(path_samples)
        self.check_arm = bool(check_arm)
        self.arm_clearance = float(arm_clearance)
        self.n_evals = 0

        #: Spheres bounding the drawn solid, for the clearance test.
        self._spheres = collision_spheres(self.arm)

        # -- the retract height -----------------------------------------
        # Every move goes up to this height before it traverses and comes
        # back down onto the port, which is what a real cell does and what
        # keeps the arm out of the machines *between* stations.  Only the
        # machines' heights enter it, and those are fixed by the instance, so
        # it is one number per instance rather than per layout.
        #: Metres above the tallest machine that the tool retracts to.  Zero
        #: (or ``None``) restores the direct point-to-point moves.
        self.retract = None if retract is None else float(retract)
        self.retract_height: Optional[float] = None
        if self.retract:
            self.retract_height = (
                max(m.height for m in instance.machines) + self.retract
            )

        _, _, self.velocity_limit, self.acceleration_limit = self.arm.limits.as_arrays()

        # The home pose is a fixed task point, so its IK depends on the branch
        # but not on the layout: solve it once per branch here rather than
        # 60 000 times inside the run.
        home = np.asarray(instance.home_point, dtype=float)[None, :]
        self._home: Dict[Tuple[int, int], Optional[np.ndarray]] = {}
        for branch in BRANCHES:
            q, ok = self.arm.ik_vertical(home, *branch)
            usable = bool(ok[0]) and bool(self.arm.within_limits(q)[0])
            usable = usable and float(self.arm.sigma_min(q)[0]) >= self.sigma_min_threshold
            self._home[branch] = q[0] if usable else None
        if all(value is None for value in self._home.values()):
            raise ValueError(
                f"home point {instance.home_point} is not reachable in any posture; "
                "the instance is misconfigured"
            )

    # -- decoding ------------------------------------------------------

    def decode(self, chromosome: np.ndarray) -> List[Machine]:
        """``[x, y, rotation] * n`` -> placed machines."""
        genes = np.asarray(chromosome, dtype=float).reshape(-1, 3)
        return [
            machine.copy_at(row[0], row[1], row[2])
            for machine, row in zip(self.instance.machines, genes)
        ]

    def reset_counter(self) -> None:
        self.n_evals = 0

    # -- the objective -------------------------------------------------

    def fitness(self, chromosome: np.ndarray) -> float:
        """Cycle time in seconds, or ``+inf`` for an infeasible layout."""
        self.n_evals += 1
        return self.evaluate(self.decode(chromosome)).cycle_time

    def evaluate(self, machines: Sequence[Machine]) -> LayoutOutcome:
        """The full outcome for a placed layout, tests in cost order."""
        instance = self.instance

        if not layout_is_placeable(
            machines,
            instance.bounds,
            instance.base_xy,
            instance.keep_out_radius,
            instance.clearance,
        ):
            return LayoutOutcome(False, math.inf, "placement", machines=list(machines))

        # home -> every port in process order -> home
        ports = np.stack([machines[i].port() for i in instance.sequence])

        best: Optional[LayoutOutcome] = None
        # When every branch fails, report the stage the *most successful*
        # branch reached, not whichever happened to be tried last -- that is
        # the one that says what actually stood in the way.
        furthest = 0
        for branch in BRANCHES:
            home_q = self._home[branch]
            if home_q is None:
                continue

            q, reachable = self.arm.ik_vertical(ports, *branch)
            if not reachable.all():
                continue
            if not self.arm.within_limits(q).all():
                furthest = max(furthest, 1)
                continue
            sigma = self.arm.sigma_min(q)
            if float(sigma.min()) < self.sigma_min_threshold:
                furthest = max(furthest, 2)
                continue

            built = self._build_tour(home_q, q, ports, branch)
            if built is None:
                # No lift pose over some port: the arm cannot get above it and
                # still be inside its travel.  That is a reach failure, and it
                # is the retract that causes it, so it is reported as one.
                continue
            tour, station_index = built

            boxes = box_frames(machines)
            if self.check_arm and self._arm_hits_a_machine(q, boxes):
                furthest = max(furthest, 3)
                continue
            if self.check_path and self._path_hits_a_machine(tour, station_index, boxes):
                furthest = max(furthest, 4)
                continue
            total = cycle_time(tour, self.velocity_limit, self.acceleration_limit,
                               self.dwell, station_index)
            if best is None or total < best.cycle_time:
                best = LayoutOutcome(
                    feasible=True,
                    cycle_time=total,
                    rejection="none",
                    branch=branch,
                    configurations=tour,
                    sigma_min=sigma,
                    waypoints=ports,
                    machines=list(machines),
                    station_index=station_index,
                )

        if best is None:
            rejection = ("reach", "limits", "sigma_min", "clearance", "path")[furthest]
            return LayoutOutcome(
                False, math.inf, rejection, waypoints=ports, machines=list(machines)
            )
        return best

    # -- the tour ------------------------------------------------------

    def _build_tour(
        self,
        home_q: np.ndarray,
        stations: np.ndarray,
        ports: np.ndarray,
        branch: Tuple[int, int],
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """The whole cycle as a list of configurations, plus which are stations.

        Without a retract this is just ``home -> station -> ... -> home``, and
        the tool takes whatever curve joint-space interpolation gives it
        between two ports -- which, in a cell this tightly packed, drags the
        forearm straight through the machines in between.

        With one, each station is bracketed by a **lift pose**: the same tool
        position raised to :attr:`retract_height`, one number above the tallest
        machine in the cell.  The cycle becomes descend / dwell / retract /
        traverse, which is what a real cell does, and the traverse now happens
        with the whole arm above everything it could hit.

        Returns ``None`` when a lift pose is unreachable or outside the joint
        travel -- lifting off a port at the edge of the envelope can put the
        wrist beyond it, and a layout the robot cannot retract from is not a
        layout it can run.
        """
        n = stations.shape[0]
        if not self.retract_height:
            tour = np.vstack([home_q[None, :], stations, home_q[None, :]])
            return tour, np.arange(1, n + 1)

        above = ports.copy()
        above[:, 2] = self.retract_height
        lifts, reachable = self.arm.ik_vertical(above, *branch)
        if not reachable.all() or not self.arm.within_limits(lifts).all():
            return None
        # sigma_min is deliberately *not* required at a lift pose: it is a via
        # point the arm passes through, not one it has to control fine motion
        # at, and imposing it there would reject layouts for a condition the
        # task never encounters.

        rows = [home_q]
        station_index = []
        for k in range(n):
            rows.append(lifts[k])
            station_index.append(len(rows))
            rows.append(stations[k])
            rows.append(lifts[k])
        rows.append(home_q)
        return np.stack(rows), np.asarray(station_index, dtype=int)

    # -- clearance between the robot and the machines --------------------

    def arm_clearances(
        self, configurations: np.ndarray, machines: Sequence[Machine]
    ) -> np.ndarray:
        """Gap between the robot and every machine, at every configuration.

        ``configurations`` is ``(n, 4)``; the result is ``(n, n_machines)`` in
        metres, negative where the arm is *inside* that machine.  The reporting
        code uses it to say how much margin a layout actually has; the
        feasibility test below is the same computation with an early exit.
        """
        return self._clearances(np.atleast_2d(configurations), box_frames(machines))

    def _clearances(
        self, configurations: np.ndarray, boxes: Tuple[np.ndarray, ...]
    ) -> np.ndarray:
        """The worst gap between the arm and each machine, per configuration.

        The arm is bounded by the chain of spheres :func:`armmesh.collision_spheres`
        measures off the drawn solid, so a layout that clears here cannot show
        the robot inside a box in any figure.

        One exemption, and it is a real one rather than a fudge: the spheres
        bounding the wrist and gripper are ignored for a machine the tool is
        *directly above* -- inside its footprint, at or above its top face.
        That is the vertical approach corridor.  The tool tip sits exactly on
        the top surface at a station, so its bounding sphere necessarily dips
        below that surface; counting it would make every legal pick a
        collision.  Nothing else is exempt: the same gripper still has to clear
        every other machine, and the forearm and upper arm have to clear the
        served machine too -- an arm that reaches over a tall machine and
        clips its own station's top edge is a crash like any other.
        """
        centres, half, rotations = boxes
        points = self.arm.points_along_arm(configurations, self._spheres.coefficients)
        distance = points_to_boxes_distance(points, centres, half, rotations)
        gap = distance - self._spheres.radii[None, :, None]

        tips = self.arm.fk(configurations)[:, :3]
        # (n_config, n_machines) -> broadcast over the spheres of each config.
        approaching = points_over_boxes(tips, centres, half, rotations)
        exempt = approaching[:, None, :] & self._spheres.is_tool[None, :, None]
        gap = np.where(exempt, np.inf, gap)
        return gap.min(axis=1)

    def _arm_hits_a_machine(
        self, configurations: np.ndarray, boxes: Tuple[np.ndarray, ...]
    ) -> bool:
        """Is any part of the robot inside a machine, or too close to one?"""
        return bool(
            (self._clearances(configurations, boxes) < self.arm_clearance).any()
        )

    def _path_hits_a_machine(
        self, tour: np.ndarray, station_index: np.ndarray,
        boxes: Tuple[np.ndarray, ...],
    ) -> bool:
        """Does the robot sweep through a machine on its way between waypoints?

        Every move interpolates in *joint* space, so the tool follows a curve
        rather than a straight line and the forearm swings above it; the
        clearance at the endpoints says nothing about the middle.  This samples
        every segment and applies the same whole-arm test the stations get.

        The endpoints themselves are excluded -- they are the waypoints, and
        the stations among them are covered by criterion 5.  With a retract in
        place the descent and the lift are part of the approach corridor, which
        :meth:`_clearances` already exempts for the machine being served.
        """
        fractions = np.linspace(0.0, 1.0, self.path_samples + 2)[1:-1]
        for k in range(tour.shape[0] - 1):
            q0, q1 = tour[k], tour[k + 1]
            samples = q0[None, :] + fractions[:, None] * (q1 - q0)[None, :]
            if self._arm_hits_a_machine(samples, boxes):
                return True
        return False

    # -- reporting helpers ---------------------------------------------

    def segment_times(self, outcome: LayoutOutcome) -> np.ndarray:
        """Duration of each move in a feasible cycle.

        One entry per segment of the tour, so with a retract there are three
        per station (descend, lift, traverse) rather than one.
        """
        if not outcome.feasible or outcome.configurations is None:
            return np.array([])
        segments = plan_cycle(
            outcome.configurations, self.velocity_limit, self.acceleration_limit
        )
        return np.array([segment.duration for segment in segments])


def feasibility_rate(
    instance: Instance,
    samples: int = 2000,
    seed: int = 7,
    sigma_min_threshold: float = 0.14,
    check_path: bool = False,
    check_arm: bool = True,
) -> Dict[str, float]:
    """How hard the instance is: what fraction of random layouts survive each test.

    A direct measure of how much the robot narrows the problem beyond the
    packing constraints, and the number that says whether a GA run is doing
    anything at all.
    """
    from problems import random_layout

    evaluator = CycleTimeEvaluator(
        instance, sigma_min_threshold=sigma_min_threshold, check_path=check_path,
        check_arm=check_arm,
    )
    rng = np.random.default_rng(seed)
    counts = {name: 0 for name in REJECTIONS}
    for _ in range(samples):
        outcome = evaluator.evaluate(evaluator.decode(random_layout(instance, rng)))
        counts[outcome.rejection] += 1
    return {
        "feasible_rate": counts["none"] / samples,
        "rejected_placement": counts["placement"] / samples,
        "rejected_reach": counts["reach"] / samples,
        "rejected_limits": counts["limits"] / samples,
        "rejected_sigma_min": counts["sigma_min"] / samples,
        "rejected_clearance": counts["clearance"] / samples,
        "rejected_path": counts["path"] / samples,
    }
