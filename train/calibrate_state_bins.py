"""
Calibrate the RL state discretisation on the *training* split only.

1. Static GA baselines (no RL, no learning) are run on the 15 **training**
   instances only, for both local-search targets and several seeds.
2. The observed population-diversity values are pooled and their tertiles
   become the two diversity cut points; the median of the strictly positive
   relative best-fitness improvements becomes the improvement cut point.
3. The thresholds, the number of samples behind them and the split they came
   from are written to ``state_bins.json``, which the training and evaluation
   scripts load.  No test-split data is touched at any point.

Run:  python -m train.calibrate_state_bins
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

import numpy as np

from core.problems import build_instances
from optimization.ga_core import ConfigurableGA, GAConfig, make_initial_population
from optimization.rl_control import StateBins
from paths import DEFAULT_BINS_PATH

DEFAULT_OUTPUT = DEFAULT_BINS_PATH


def _job(args: Tuple[str, int, int, str, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Run one static-GA probe and return its diversity / improvement samples."""
    split, index, seed, ls_target, generations, population_size = args

    instance = build_instances(split)[index]
    config = GAConfig(
        label=f"calibration-{ls_target}",
        ls_target=ls_target,
        ls_controller="static",
        ls_rate=0.1,
        generations=generations,
        population_size=population_size,
    )
    ga = ConfigurableGA(
        instance.machines,
        instance.sequence,
        instance.robot_position,
        instance.workspace_bounds,
        config=config,
        seed=seed,
        instance_name=instance.name,
    )
    population = make_initial_population(
        instance.machines, instance.workspace_bounds, population_size, seed=seed
    )
    result = ga.optimize(initial_population=population)

    diversity = result.diversity[np.isfinite(result.diversity)]

    # Relative improvement of the generation best, the quantity the RL state
    # is actually built from.
    gen_best = result.gen_best
    improvements: List[float] = []
    for g in range(1, len(gen_best)):
        prev, curr = gen_best[g - 1], gen_best[g]
        if np.isfinite(prev) and np.isfinite(curr) and prev != 0:
            improvements.append((prev - curr) / abs(prev))
    return diversity, np.array(improvements, dtype=float)


def calibrate(
    split: str = "train",
    seeds: Tuple[int, ...] = (0, 1),
    generations: int = 300,
    population_size: int = 200,
    workers: int = 0,
    output: str = DEFAULT_OUTPUT,
) -> StateBins:
    instances = build_instances(split)
    jobs = [
        (split, index, seed, ls_target, generations, population_size)
        for index in range(len(instances))
        for seed in seeds
        for ls_target in ("offspring", "elites")
    ]
    workers = workers or min(len(jobs), max(1, (os.cpu_count() or 2) - 2))

    print(
        f"Calibrating state bins on the '{split}' split: {len(instances)} instances x "
        f"{len(seeds)} seeds x 2 local-search targets = {len(jobs)} probe runs "
        f"({workers} workers)"
    )

    diversity_samples: List[np.ndarray] = []
    improvement_samples: List[np.ndarray] = []

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for k, (diversity, improvements) in enumerate(pool.map(_job, jobs), start=1):
                diversity_samples.append(diversity)
                improvement_samples.append(improvements)
                print(f"  probe {k}/{len(jobs)} done", end="\r")
    else:
        for k, job in enumerate(jobs, start=1):
            diversity, improvements = _job(job)
            diversity_samples.append(diversity)
            improvement_samples.append(improvements)
            print(f"  probe {k}/{len(jobs)} done", end="\r")

    diversity = np.concatenate(diversity_samples)
    improvements = np.concatenate(improvement_samples)
    positive = improvements[improvements > 0]

    lo, hi = np.percentile(diversity, [33.3333, 66.6667])
    improvement_edge = float(np.median(positive)) if positive.size else 0.01

    bins = StateBins(
        diversity_edges=(float(lo), float(hi)),
        improvement_edge=improvement_edge,
        source=(
            f"tertiles of population diversity and median positive relative improvement "
            f"observed on the '{split}' split "
            f"({len(instances)} instances x {len(seeds)} seeds x 2 LS targets, "
            f"{generations} generations, pop {population_size}); "
            f"{diversity.size} diversity samples, {positive.size} positive-improvement samples. "
            f"No test-split data was used."
        ),
    )
    bins.save(output)

    print("\nCalibrated state bins")
    print(f"  diversity tertiles : {lo:.2f}, {hi:.2f}")
    print(f"     (min {diversity.min():.2f}, median {np.median(diversity):.2f}, max {diversity.max():.2f})")
    print(f"  improvement edge   : {improvement_edge:.6f}")
    print(f"     (positive improvements: {positive.size} of {improvements.size} generations)")
    print(f"  written to         : {output}")

    # Keep the raw samples so the choice can be re-derived or plotted later.
    raw_path = os.path.splitext(output)[0] + "_samples.npz"
    np.savez_compressed(raw_path, diversity=diversity, improvements=improvements)
    print(f"  raw samples        : {raw_path}")

    return bins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--population-size", type=int, default=200)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    calibrate(
        split=args.split,
        seeds=tuple(args.seeds),
        generations=args.generations,
        population_size=args.population_size,
        workers=args.workers,
        output=args.output,
    )


if __name__ == "__main__":
    main()
