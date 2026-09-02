"""
The instance family: workcells sized to a Puma 560.

Scale
-----
This is the one place the study departs from a purely notional layout problem.
A Puma 560 reaches 0.864 m from the shoulder, so the cell has to be a cell a
Puma could actually serve: the floor is 1.4 m square, machines are benchtop
units 0.10-0.18 m across and 0.10-0.40 m tall, and the arm stands in the
middle on a 0.67 m pedestal.  Sizing the problem to the robot is what makes
reachability a real constraint instead of one that is either always or never
satisfied.  The cell's corners sit at radius 0.99 m, well outside the arm's
envelope, so the corners are genuinely unusable rather than decorative.

The consequence is worth stating plainly: only an annulus of the floor is
usable at all.  Measured on this arm, a top-down pick is possible for port
radii of roughly 0.16-0.71 m at the lowest machine height and 0.21-0.85 m at
the highest; requiring ``sigma_min >= 0.14`` then removes a further 19 % of
those poses -- the outermost band, where the elbow approaches lock-out.  The
middle is occupied by the robot's own column.  A layout that ignores any of
this scores ``+inf``.

What varies between instances
-----------------------------
Machine footprints, heights, the port position on each machine, and the
process order.  What does not vary: the cell size, the robot, and the number
of machines, so cycle times are comparable across instances of a split.

Splits are disjoint by construction -- they are drawn from disjoint seed
ranges -- so an instance used to develop the operators is never one that is
reported on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from geometry import Bounds, Machine, layout_is_placeable

#: Seed ranges, disjoint so no instance appears in two splits.
_SPLIT_SEEDS: Dict[str, Tuple[int, int]] = {
    "test": (1000, 1010),
    "train": (2000, 2015),
    "generalization": (4000, 4006),
}

ALL_SPLITS: Tuple[str, ...] = tuple(_SPLIT_SEEDS)


@dataclass
class Instance:
    """One workcell: the machines, the order they are served, and the cell."""

    name: str
    split: str
    index: int
    machines: List[Machine]
    sequence: List[int]
    bounds: Bounds
    base_xy: Tuple[float, float] = (0.0, 0.0)
    keep_out_radius: float = 0.18
    clearance: float = 0.02
    #: Tool-tip position the cycle starts and ends at.
    home_point: Tuple[float, float, float] = (0.40, 0.0, 0.45)

    @property
    def n_machines(self) -> int:
        return len(self.machines)

    def gene_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Per-gene box for ``[x, y, rotation]`` of every machine.

        Positions are inset by each machine's half-diagonal, so any rotation
        of a machine placed inside the box still has its footprint inside the
        cell and the position and rotation genes stay independent.
        """
        min_x, max_x, min_y, max_y = self.bounds
        lower: List[float] = []
        upper: List[float] = []
        for machine in self.machines:
            radius = machine.footprint_radius
            lower.extend([min_x + radius, min_y + radius, 0.0])
            upper.extend([max_x - radius, max_y - radius, 360.0])
        return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def build_instance(seed: int, name: str, split: str, index: int, n_machines: int = 6) -> Instance:
    """One reproducible instance from a seed."""
    rng = np.random.default_rng(seed)

    machines: List[Machine] = []
    for k in range(n_machines):
        width = float(rng.uniform(0.10, 0.18))
        depth = float(rng.uniform(0.10, 0.18))
        height = float(rng.uniform(0.10, 0.40))
        # The port sits off-centre on the top face, so a machine's rotation
        # changes where the robot must reach and the rotation gene matters.
        port = (0.30 * width, 0.0)
        machines.append(
            Machine(
                name=f"M{k + 1}",
                width=width,
                depth=depth,
                height=height,
                port_local=port,
            )
        )

    sequence = list(rng.permutation(n_machines))
    return Instance(
        name=name,
        split=split,
        index=index,
        machines=machines,
        sequence=[int(s) for s in sequence],
        bounds=(-0.70, 0.70, -0.70, 0.70),
    )


def build_instances(split: str = "test") -> List[Instance]:
    """Every instance of a split, in a fixed order."""
    if split not in _SPLIT_SEEDS:
        raise ValueError(f"unknown split {split!r} (expected one of {ALL_SPLITS})")
    first, stop = _SPLIT_SEEDS[split]
    return [
        build_instance(seed, f"{split}_{k + 1}", split, k, n_machines=6)
        for k, seed in enumerate(range(first, stop))
    ]


def random_layout(instance: Instance, rng: np.random.Generator) -> np.ndarray:
    """A uniformly random chromosome inside the gene box."""
    lower, upper = instance.gene_bounds()
    return lower + rng.random(lower.size) * (upper - lower)


def placement_rate(instance: Instance, samples: int = 2000, seed: int = 7) -> float:
    """Fraction of random layouts that pass the *geometric* tests alone.

    Reported next to the full feasibility rate so the two constraint families
    can be told apart: this one measures how crowded the cell is, and the gap
    to the full rate measures how much the robot itself rules out.
    """
    rng = np.random.default_rng(seed)
    passed = 0
    for _ in range(samples):
        genes = random_layout(instance, rng).reshape(-1, 3)
        placed = [
            machine.copy_at(row[0], row[1], row[2])
            for machine, row in zip(instance.machines, genes)
        ]
        if layout_is_placeable(
            placed, instance.bounds, instance.base_xy,
            instance.keep_out_radius, instance.clearance,
        ):
            passed += 1
    return passed / samples


def describe(instance: Instance) -> Dict[str, float]:
    """A few scalars per instance, for the results metadata."""
    footprint = sum(m.width * m.depth for m in instance.machines)
    min_x, max_x, min_y, max_y = instance.bounds
    floor = (max_x - min_x) * (max_y - min_y)
    heights = [m.height for m in instance.machines]
    return {
        "n_machines": float(instance.n_machines),
        "footprint_area": footprint,
        "floor_area": floor,
        "occupancy": footprint / floor,
        "min_height": min(heights),
        "max_height": max(heights),
        "mean_height": float(np.mean(heights)),
    }
