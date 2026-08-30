"""
Assert the properties the GA-versus-metaheuristics comparison rests on.

The claim the comparison makes is that the GA, DE, PSO, ABC and CSS differ
only in their search strategy: same problem, same objective, same starting
point, same budget.  This script checks each half of that claim rather than
trusting it --

1. *same starting point*  - on one (instance, seed) pair every method's first
   recorded iteration describes the identical initial population (best, mean,
   diversity and feasible fraction all bit-identical to a direct evaluation of
   the shared population);
2. *same budget*          - no method exceeds the evaluation cap, and the ones
   that spend one population per iteration hit it exactly;
3. *same box*             - every reported layout respects the bounds the GA's
   ``apply_bounds`` enforces;
4. *reproducible*         - one seed, one result, twice;
5. *the search works*     - on a smooth surrogate objective (a sphere in the
   normalised variables) each algorithm reduces the objective by orders of
   magnitude, which the constrained layout objective, where most candidates
   score ``+inf``, cannot show on its own;
6. *one grid*             - a baseline run goes through the same job runner as
   a GA run and produces the same CSV record.

Run:  python -m tests.test_metaheuristics
"""

from __future__ import annotations

import math
import sys

import numpy as np

from core.problems import build_instances
from optimization.ga_core import ConfigurableGA, GAConfig, make_initial_population
from optimization.metaheuristics import (
    ALGORITHMS,
    MetaConfig,
    PopulationOptimizer,
    build_optimizer,
)
from optimization.run_experiments import CSV_COLUMNS, run_seed_for

#: Small budget: these checks are about mechanics, not about who wins.
POPULATION = 40
ITERATIONS = 30


def check(name: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{(' - ' + detail) if detail else ''}")
    return bool(condition)


def make_optimizer(algorithm: str, instance, seed: int, **overrides) -> PopulationOptimizer:
    settings = {"population_size": POPULATION, "generations": ITERATIONS}
    settings.update(overrides)
    config = MetaConfig(algorithm=algorithm, **settings)
    return build_optimizer(
        algorithm,
        instance.machines,
        instance.sequence,
        instance.robot_position,
        instance.workspace_bounds,
        config=config,
        seed=seed,
        instance_name=instance.name,
    )


def sphere_objective(optimizer: PopulationOptimizer):
    """A smooth stand-in for the layout objective, in normalised variables.

    Wraps the real evaluator so the evaluation counter - and therefore the
    budget accounting under test - keeps working.
    """
    target = 0.5 * (optimizer.lower + optimizer.upper)
    span = optimizer.upper - optimizer.lower

    def objective(chromosome: np.ndarray) -> float:
        optimizer.evaluator.n_evals += 1
        return float(np.sum(((np.asarray(chromosome, dtype=float) - target) / span) ** 2))

    return objective


def main() -> int:
    ok = True
    instance = build_instances("test")[0]
    seed = run_seed_for(0, 0)
    population = make_initial_population(
        instance.machines, instance.workspace_bounds, POPULATION, seed=seed
    )

    # ---- the shared starting point, measured directly ------------------
    reference = ConfigurableGA(
        instance.machines,
        instance.sequence,
        instance.robot_position,
        instance.workspace_bounds,
        config=GAConfig(
            label="GA no LS",
            ls_target="none",
            ls_controller="static",
            ls_rate=0.0,
            population_size=POPULATION,
            generations=ITERATIONS,
        ),
        seed=seed,
        instance_name=instance.name,
    )
    initial_fitness = np.array([reference.fitness_function(c) for c in population])
    finite = initial_fitness[np.isfinite(initial_fitness)]
    expected = {
        "gen_best": float(initial_fitness.min()),
        "gen_avg": float(finite.mean()),
        "feasible_fraction": float(finite.size) / float(initial_fitness.size),
        "diversity": reference.calculate_diversity(population),
    }
    ga_result = reference.optimize(initial_population=population)

    print(f"Instance {instance.name}, seed {seed}, population {POPULATION}, "
          f"{ITERATIONS} iterations.")
    print(f"Shared initial population: best {expected['gen_best']:.4f}, "
          f"feasible {expected['feasible_fraction']:.1%}\n")

    results = {"ga_ls00": ga_result}
    for algorithm in ALGORITHMS:
        results[algorithm] = make_optimizer(algorithm, instance, seed).optimize(
            initial_population=population
        )

    print("1. Every method starts from the same initial population:")
    for key, result in results.items():
        same = all(
            float(result.trace_dict()[trace][0]) == value for trace, value in expected.items()
        )
        ok &= check(
            f"{key} first iteration matches the shared population",
            same,
            f"best {result.gen_best[0]:.4f}, feasible {result.feasible_fraction[0]:.1%}, "
            f"diversity {result.diversity[0]:.4f}",
        )

    print("\n2. Every method respects the same evaluation budget:")
    budget = POPULATION * ITERATIONS
    for key, result in results.items():
        ok &= check(
            f"{key} spends at most {budget} evaluations",
            result.total_evaluations <= budget,
            f"{result.total_evaluations} in {result.generations} iterations",
        )
    for key in ("ga_ls00", "de", "pso", "css"):
        ok &= check(
            f"{key} spends one population per iteration, so exactly {budget}",
            results[key].total_evaluations == budget,
            f"{results[key].total_evaluations}",
        )
    ok &= check(
        "abc spends two populations per cycle, so about half the iterations",
        results["abc"].generations <= ITERATIONS // 2 + 1,
        f"{results['abc'].generations} cycles",
    )

    print("\n3. Every reported layout stays inside the GA's bounds:")
    probe = make_optimizer("de", instance, seed)
    for key, result in results.items():
        chromosome = np.asarray(result.best_chromosome, dtype=float)
        positions = ~probe.is_rotation
        inside = bool(
            np.all(chromosome[positions] >= probe.lower[positions] - 1e-9)
            and np.all(chromosome[positions] <= probe.upper[positions] + 1e-9)
        )
        rotations = chromosome[probe.is_rotation]
        wrapped = bool(np.all(rotations >= 0.0) and np.all(rotations < 360.0))
        ok &= check(f"{key} best layout is inside the box", inside and wrapped)

    print("\n4. A seed determines the run:")
    for algorithm in ALGORITHMS:
        repeat = make_optimizer(algorithm, instance, seed).optimize(initial_population=population)
        identical = (
            repeat.best_fitness == results[algorithm].best_fitness
            and np.array_equal(repeat.best_chromosome, results[algorithm].best_chromosome)
            and np.array_equal(
                repeat.best_so_far, results[algorithm].best_so_far, equal_nan=True
            )
        )
        ok &= check(f"{algorithm} repeats exactly", identical, f"best {repeat.best_fitness:.6f}")

    print("\n5. Each algorithm actually searches (sphere surrogate):")
    for algorithm in ALGORITHMS:
        optimizer = make_optimizer(algorithm, instance, seed, generations=120)
        optimizer.fitness_function = sphere_objective(optimizer)
        result = optimizer.optimize(initial_population=population)
        start = float(result.gen_best[0])
        reduction = start / max(result.best_fitness, 1e-12)
        ok &= check(
            f"{algorithm} reduces the surrogate objective at least 5-fold",
            reduction >= 5.0,
            f"{start:.4f} -> {result.best_fitness:.6f} ({reduction:,.0f}x)",
        )

    print("\n6. A baseline run goes through the same grid as a GA run:")
    from run_static_ga import StaticJob, execute

    job = StaticJob(
        variant_key="pso",
        config=MetaConfig(algorithm="pso", population_size=POPULATION, generations=5),
        split="test",
        instance_index=0,
        run=0,
        output_dir="",
        store_trace=False,
    )
    row = execute(job)
    ok &= check(
        "the run record has exactly the archived CSV columns",
        set(row) == set(CSV_COLUMNS),
        f"{len(row)} columns",
    )
    ok &= check(
        "the record carries the run seed the GA rows use",
        row["run_seed"] == run_seed_for(0, 0) and row["variant"] == "pso",
        f"run_seed {row['run_seed']}",
    )
    ok &= check(
        "the record reports a finite fitness and its layout distance",
        math.isfinite(row["best_fitness"]) and math.isfinite(row["total_distance"]),
        f"fitness {row['best_fitness']:.3f}",
    )

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
