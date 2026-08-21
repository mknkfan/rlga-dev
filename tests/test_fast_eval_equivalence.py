"""
Assert that ``feasibility.evaluator.LayoutEvaluator`` reproduces the original
object-based fitness function exactly.

Run:  python -m tests.test_fast_eval_equivalence
"""

from __future__ import annotations

import random
import sys
import time

import numpy as np

from tests.reference_fyp.original_ga import GeneticAlgorithm
from core.problems import ALL_SPLITS, build_instances
from feasibility.evaluator import LayoutEvaluator


def random_chromosome(rng: random.Random, machines, bounds, tight: bool) -> np.ndarray:
    """A random layout; ``tight`` clusters machines so collisions are common."""
    min_x, max_x, min_y, max_y = bounds
    if tight:
        min_x, max_x, min_y, max_y = (v * 0.25 for v in (min_x, max_x, min_y, max_y))
    genes = []
    for _ in machines:
        genes.extend(
            [
                rng.uniform(min_x, max_x),
                rng.uniform(min_y, max_y),
                rng.uniform(-720, 720),  # exercise the modulo / snapping path
            ]
        )
    return np.array(genes)


def main(n_per_instance: int = 400) -> int:
    rng = random.Random(20250801)
    checked = 0
    infeasible = 0
    max_rel_err = 0.0
    t_old = 0.0
    t_new = 0.0

    for split in ALL_SPLITS:
        for inst in build_instances(split):
            ga = GeneticAlgorithm(
                inst.machines, inst.sequence, inst.robot_position, inst.workspace_bounds
            )
            ev = LayoutEvaluator(
                inst.machines, inst.sequence, inst.robot_position, inst.workspace_bounds
            )

            random.seed(rng.randrange(1 << 30))
            chromosomes = ga.create_initial_population()[: n_per_instance // 2]
            chromosomes += [
                random_chromosome(rng, inst.machines, inst.workspace_bounds, tight=(k % 2 == 0))
                for k in range(n_per_instance // 2)
            ]

            for chrom in chromosomes:
                t0 = time.perf_counter()
                expected = ga.fitness_function(chrom)
                t1 = time.perf_counter()
                actual = ev.fitness(chrom)
                t2 = time.perf_counter()
                t_old += t1 - t0
                t_new += t2 - t1

                checked += 1
                if np.isinf(expected) or np.isinf(actual):
                    if not (np.isinf(expected) and np.isinf(actual)):
                        print(
                            f"FAIL [{split} instance {inst.name}]: feasibility mismatch "
                            f"expected={expected} actual={actual}"
                        )
                        return 1
                    infeasible += 1
                    continue

                if expected != actual:
                    rel = abs(expected - actual) / max(abs(expected), 1e-12)
                    max_rel_err = max(max_rel_err, rel)
                    if rel > 1e-12:
                        print(
                            f"FAIL [{split} instance {inst.name}]: "
                            f"expected={expected!r} actual={actual!r} rel={rel:.3e}"
                        )
                        return 1

    print(
        f"OK: {checked} chromosomes match "
        f"({infeasible} infeasible, {checked - infeasible} feasible); "
        f"max relative error {max_rel_err:.3e}"
    )
    print(f"    original: {t_old:.2f}s   fast: {t_new:.2f}s   speed-up: {t_old / max(t_new, 1e-9):.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
