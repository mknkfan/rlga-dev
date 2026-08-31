"""
Benchmark instance generation for the robot-cell layout problem, with a fixed
train / test / generalization split so no instance is ever reused across
purposes.

An instance is a tuple (machines, visiting sequence, robot position,
workspace bounds). Machines alternate rectangle and L-shape, are sized,
placed and rotated (snapped to 90-degree steps) from a per-instance seed, so
every instance is reproducible from that seed alone.

Splits (disjoint seed ranges):

=================  =====  ==========  ==========================  ============
split              n      seeds       geometry                    used for
=================  =====  ==========  ==========================  ============
``train``          15     2000-2014   8 machines, H = 15          RL training
``test``           10     1000-1009   8 machines, H = 15          all reported comparisons
``generalization``  6     4000-4005   6-12 machines, H = 10-20    out-of-distribution
=================  =====  ==========  ==========================  ============

The ``test`` split reproduces the original submission's generator
bit-for-bit; ``tests/test_problem_parity.py`` checks this against
``tests/reference_fyp/original_problems.py``.

Also provides :func:`instance_features` (per-instance descriptive stats -
occupancy, feasibility rate, size dispersion, etc. - used by the analysis
scripts) and :func:`dump_instance_catalogue` (writes the full instance set to
JSON).
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.datastruct import Machine, Point

ALL_SPLITS: Tuple[str, ...] = ("train", "test", "generalization")

Bounds = Tuple[float, float, float, float]


@dataclass(frozen=True)
class SizeClass:
    """Machine geometry: side ranges, the access-point square and the
    L-shape cut-out floor.  ``InstanceSpec.geometry_for`` builds one from the
    flat per-instance fields for every generated instance.
    """

    width_range: Tuple[float, float]
    height_range: Tuple[float, float]
    access_range: float
    cutout_min: float


#: How the shapes are distributed over the machine indices.
SHAPE_PATTERNS: Tuple[str, ...] = ("alternating", "all_rectangle", "all_l_shape")


@dataclass
class InstanceSpec:
    """Generation parameters for one instance (everything needed to rebuild it)."""

    seed: int
    n_machines: int = 8
    half_extent: float = 15.0
    width_range: Tuple[float, float] = (3.0, 6.0)
    height_range: Tuple[float, float] = (2.5, 5.0)
    access_range: float = 1.5
    cutout_min: float = 1.0
    shape_pattern: str = "alternating"
    rect_size: Optional[SizeClass] = None
    l_size: Optional[SizeClass] = None

    def shape_at(self, index: int) -> str:
        """Which shape machine ``index`` has."""
        if self.shape_pattern == "all_rectangle":
            return "rectangle"
        if self.shape_pattern == "all_l_shape":
            return "l_shape"
        if self.shape_pattern == "alternating":
            return "l_shape" if index % 2 == 0 else "rectangle"
        raise ValueError(
            f"unknown shape_pattern {self.shape_pattern!r} "
            f"(expected one of {SHAPE_PATTERNS})"
        )

    def geometry_for(self, shape: str) -> SizeClass:
        """The size class machine of ``shape`` is drawn from."""
        override = self.l_size if shape == "l_shape" else self.rect_size
        if override is not None:
            return override
        return SizeClass(
            width_range=self.width_range,
            height_range=self.height_range,
            access_range=self.access_range,
            cutout_min=self.cutout_min,
        )


@dataclass
class Instance:
    """A fully materialised benchmark instance."""

    name: str
    split: str
    index: int
    spec: InstanceSpec
    machines: List[Machine]
    sequence: List[int]
    robot_position: Point
    workspace_bounds: Bounds

    @property
    def n_machines(self) -> int:
        return len(self.machines)


# ----------------------------------------------------------------------
# generation
# ----------------------------------------------------------------------


def build_instance(spec: InstanceSpec, name: str, split: str, index: int) -> Instance:
    """Materialise an instance from its specification (deterministic in ``spec.seed``)."""
    rng = random.Random(spec.seed)

    h = spec.half_extent
    workspace_bounds: Bounds = (-h, h, -h, h)
    robot_position = Point(0, 0)

    machines: List[Machine] = []
    for i in range(spec.n_machines):
        machine_id = i + 1
        shape = spec.shape_at(i)
        geometry = spec.geometry_for(shape)

        width = rng.uniform(*geometry.width_range)
        height = rng.uniform(*geometry.height_range)
        access_x = rng.uniform(-geometry.access_range, geometry.access_range)
        access_y = rng.uniform(-geometry.access_range, geometry.access_range)

        if shape == "l_shape":
            # min(...) keeps the cut-out at most half the side for machines
            # smaller than the baseline; at the baseline geometry every side
            # exceeds 2 units, so this is U(1, side/2) unchanged.
            cutout_w = rng.uniform(min(geometry.cutout_min, width / 2.0), width / 2.0)
            cutout_h = rng.uniform(min(geometry.cutout_min, height / 2.0), height / 2.0)
            machines.append(
                Machine(
                    id=machine_id,
                    shape="l_shape",
                    width=width,
                    height=height,
                    access_point=Point(access_x, access_y),
                    l_cutout_width=cutout_w,
                    l_cutout_height=cutout_h,
                )
            )
        else:
            machines.append(
                Machine(
                    id=machine_id,
                    shape="rectangle",
                    width=width,
                    height=height,
                    access_point=Point(access_x, access_y),
                )
            )

    sequence = list(range(1, spec.n_machines + 1))
    return Instance(
        name=name,
        split=split,
        index=index,
        spec=spec,
        machines=machines,
        sequence=sequence,
        robot_position=robot_position,
        workspace_bounds=workspace_bounds,
    )


DEFAULT_SPLIT_SIZES: Dict[str, int] = {"train": 15, "test": 10}


def split_size(split: str) -> int:
    """Number of instances in a split, honouring the environment override."""
    default = DEFAULT_SPLIT_SIZES.get(split, 0)
    raw = os.environ.get(f"FYP_{split.upper()}_INSTANCES")
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"FYP_{split.upper()}_INSTANCES must be an integer, got {raw!r}"
        ) from exc
    if value < 1:
        raise ValueError(f"FYP_{split.upper()}_INSTANCES must be >= 1, got {value}")
    return value


def split_specs(split: str) -> List[InstanceSpec]:
    """The generation parameters of every instance in a split."""
    if split == "test":
        return [InstanceSpec(seed=1000 + k) for k in range(split_size("test"))]
    if split == "train":
        return [InstanceSpec(seed=2000 + k) for k in range(split_size("train"))]
    if split == "generalization":
        # Out-of-distribution: machine counts and workspace sizes never seen in
        # training, spanning low to high constraint density.  Machine size
        # ranges are held fixed so that density varies with n and H alone.
        return [
            InstanceSpec(seed=4000, n_machines=6, half_extent=15.0),
            InstanceSpec(seed=4001, n_machines=10, half_extent=15.0),
            InstanceSpec(seed=4002, n_machines=12, half_extent=18.0),
            InstanceSpec(seed=4003, n_machines=8, half_extent=11.0),
            InstanceSpec(seed=4004, n_machines=10, half_extent=20.0),
            InstanceSpec(seed=4005, n_machines=12, half_extent=14.0),
        ]
    raise ValueError(f"unknown split {split!r} (expected one of {ALL_SPLITS})")


def build_instances(split: str) -> List[Instance]:
    """All instances of a split, in a fixed order."""
    specs = split_specs(split)
    return [
        build_instance(spec, name=f"{split}_{k + 1}", split=split, index=k)
        for k, spec in enumerate(specs)
    ]


# ----------------------------------------------------------------------
# instance descriptors
# ----------------------------------------------------------------------


def machine_area(machine: Machine) -> float:
    """Footprint area, accounting for the L-shaped cut-out."""
    area = machine.width * machine.height
    if machine.shape == "l_shape":
        area -= machine.l_cutout_width * machine.l_cutout_height
    return area


def feasibility_rate(instance: Instance, n_samples: int = 2000, seed: int = 7) -> float:
    """Fraction of uniformly random layouts that satisfy all hard constraints.

    A direct measure of how constrained an instance is: the smaller this is,
    the harder it is for the GA to find any feasible chromosome at all.
    """
    from feasibility.evaluator import LayoutEvaluator

    ev = LayoutEvaluator(
        instance.machines, instance.sequence, instance.robot_position, instance.workspace_bounds
    )
    rng = np.random.default_rng(seed)
    min_x, max_x, min_y, max_y = instance.workspace_bounds

    feasible = 0
    for _ in range(n_samples):
        genes = []
        for machine in instance.machines:
            genes.extend(
                [
                    rng.uniform(min_x + machine.width / 2, max_x - machine.width / 2),
                    rng.uniform(min_y + machine.height / 2, max_y - machine.height / 2),
                    rng.uniform(0.0, 360.0),
                ]
            )
        if np.isfinite(ev.fitness(np.array(genes))):
            feasible += 1
    return feasible / n_samples


def instance_features(instance: Instance, with_feasibility: bool = True) -> Dict[str, float]:
    """Measurable instance characteristics used to explain per-instance results."""
    min_x, max_x, min_y, max_y = instance.workspace_bounds
    workspace_area = (max_x - min_x) * (max_y - min_y)

    areas = np.array([machine_area(m) for m in instance.machines], dtype=float)
    widths = np.array([m.width for m in instance.machines], dtype=float)
    heights = np.array([m.height for m in instance.machines], dtype=float)
    aspect = np.maximum(widths, heights) / np.minimum(widths, heights)
    access_offset = np.array(
        [np.hypot(m.access_point.x, m.access_point.y) for m in instance.machines], dtype=float
    )
    diagonal = np.hypot(widths, heights)

    features: Dict[str, float] = {
        "n_machines": float(len(instance.machines)),
        "workspace_half_extent": float(max_x),
        "workspace_area": float(workspace_area),
        "total_machine_area": float(areas.sum()),
        "occupancy": float(areas.sum() / workspace_area),
        "free_space_ratio": float(1.0 - areas.sum() / workspace_area),
        "mean_machine_area": float(areas.mean()),
        "machine_area_cv": float(areas.std() / areas.mean()),
        "mean_aspect_ratio": float(aspect.mean()),
        "max_aspect_ratio": float(aspect.max()),
        "l_shape_fraction": float(
            np.mean([1.0 if m.shape == "l_shape" else 0.0 for m in instance.machines])
        ),
        "mean_access_offset": float(access_offset.mean()),
        # How far the largest machine footprint reaches relative to the
        # workspace: a proxy for how much room is left to manoeuvre.
        "max_diagonal_over_extent": float(diagonal.max() / (max_x - min_x)),
    }
    if with_feasibility:
        features["random_feasibility_rate"] = feasibility_rate(instance)
    return features


def dump_instance_catalogue(
    path: str, splits: Sequence[str] = ALL_SPLITS, with_feasibility: bool = True
) -> Dict[str, List[Dict]]:
    """Write a machine-readable description of every instance."""
    catalogue: Dict[str, List[Dict]] = {}
    for split in splits:
        entries = []
        for instance in build_instances(split):
            entries.append(
                {
                    "name": instance.name,
                    "split": instance.split,
                    "index": instance.index,
                    "seed": instance.spec.seed,
                    "n_machines": instance.n_machines,
                    "workspace_bounds": list(instance.workspace_bounds),
                    "robot_position": [instance.robot_position.x, instance.robot_position.y],
                    "sequence": instance.sequence,
                    "machines": [
                        {
                            "id": m.id,
                            "shape": m.shape,
                            "width": m.width,
                            "height": m.height,
                            "access_point": [m.access_point.x, m.access_point.y],
                            "l_cutout_width": m.l_cutout_width,
                            "l_cutout_height": m.l_cutout_height,
                        }
                        for m in instance.machines
                    ],
                    "features": instance_features(instance, with_feasibility=with_feasibility),
                }
            )
        catalogue[split] = entries

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(catalogue, fh, indent=2)
    return catalogue


if __name__ == "__main__":
    from run.paths import DEFAULT_RESULTS_DIR

    out = os.path.join(DEFAULT_RESULTS_DIR, "instances.json")
    catalogue = dump_instance_catalogue(out)
    for split, entries in catalogue.items():
        print(f"\n=== {split} ({len(entries)} instances) ===")
        for entry in entries:
            f = entry["features"]
            print(
                f"  {entry['name']:<20} seed={entry['seed']:<5} n={int(f['n_machines']):<3}"
                f" H={f['workspace_half_extent']:<5.1f}"
                f" occupancy={f['occupancy']:.3f}"
                f" feasible={f.get('random_feasibility_rate', float('nan')):.4f}"
            )
    print(f"\nWrote {out}")
