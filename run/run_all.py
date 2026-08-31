"""
Stages, in order (each can be run on its own; see ``--only``):

1. ``instances``   - write the instance catalogue with per-instance features.
2. ``calibrate``   - derive the RL state bins from the training split.
3. ``train``       - train the agents, several independent seeds each.
4. ``main``        - the main grid on the test split.
5. ``timing``      - a sequential replication used for the wall-clock table
                     : running the main grid in parallel makes
                     wall-clock time noisy, so the timing numbers come from a
                     pass in which exactly one run occupies the machine.
6. ``analyze``     - statistics and report.
7. ``figures``     - all plots.

Run everything:      python -m run.run_all
Run a single stage:  python -m run.run_all --only analyze
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Sequence

from run.paths import DEFAULT_RESULTS_DIR

STAGES: Sequence[str] = (
    "instances",
    "calibrate",
    "train",
    "main",
    "timing",
    "analyze",
    "figures",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=list(STAGES), default=None)
    parser.add_argument("--skip", nargs="+", choices=list(STAGES), default=())
    parser.add_argument("--runs", type=int, default=10, help="seeds per instance in the main grid")
    parser.add_argument("--timing-runs", type=int, default=3, help="seeds per instance in the sequential timing pass")
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--population-size", type=int, default=200)
    parser.add_argument("--runs-per-training-instance", type=int, default=10)
    parser.add_argument("--agent-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--control-interval",
        type=int,
        default=None,
        help="generations between RL control decisions (default: GAConfig's value)",
    )
    parser.add_argument(
        "--rl-penalty",
        type=float,
        default=None,
        help="reward for a non-improving generation, e.g. -0.01 (default: -0.1). "
        "Applies to training and evaluation alike; exported as "
        "FYP_RL_STALL_PENALTY so the worker processes see it too",
    )
    parser.add_argument("--agent-dir", default=None, help="where trained agents live")
    parser.add_argument("--bins", default=None, help="state-bin file")
    parser.add_argument(
        "--train-ablation-agents",
        action="store_true",
        help="also train the point-4b per-parameter ablation agents (lsonly/mutonly/"
        "xoveronly/noLS). Off by default: nothing in the main/timing/analyze/figures "
        "stages evaluates them yet -- pass this only when you are about to run the "
        "still-unwired ablation analysis described in the report's future-work list.",
    )
    args = parser.parse_args()

    # Before anything imports ga_core: GAConfig reads this when a configuration
    # is built, in this process and in every spawned worker.
    if args.rl_penalty is not None:
        os.environ["FYP_RL_STALL_PENALTY"] = repr(args.rl_penalty)

    from optimization.variants import DEFAULT_CONTROL_INTERVAL as _DEFAULT_INTERVAL

    interval = args.control_interval if args.control_interval is not None else _DEFAULT_INTERVAL
    agent_dir = args.agent_dir or os.path.join(args.results_dir, "agents")
    bins_path = args.bins or os.path.join(agent_dir, "state_bins.json")

    from optimization.ga_core import GAConfig

    stages = [s for s in (args.only or STAGES) if s not in args.skip]
    print(f"Pipeline stages: {', '.join(stages)}")
    print(f"RL stall penalty: {GAConfig().rl_stall_penalty}\n")
    started = time.perf_counter()

    for stage in stages:
        print(f"\n{'=' * 78}\n== {stage}\n{'=' * 78}")
        stage_started = time.perf_counter()

        if stage == "instances":
            from core.problems import dump_instance_catalogue

            dump_instance_catalogue(os.path.join(args.results_dir, "instances.json"))

        elif stage == "calibrate":
            from train.calibrate_state_bins import calibrate

            calibrate(
                generations=args.generations,
                population_size=args.population_size,
                workers=args.workers,
                output=bins_path,
            )

        elif stage == "train":
            from train.train_agents import DEFAULT_VARIANTS, train_all
            from optimization.variants import ablation_variants

            # Every RL variant needs its own agent: the action sets differ, so
            # a Q-table trained for one cannot be reused by another.
            variant_keys: List[str] = list(DEFAULT_VARIANTS)
            if args.train_ablation_agents:
                variant_keys += list(ablation_variants())

            train_all(
                variant_keys=variant_keys,
                agent_seeds=args.agent_seeds,
                runs_per_instance=args.runs_per_training_instance,
                workers=args.workers,
                directory=agent_dir,
                bins_path=bins_path,
                generations=args.generations,
                population_size=args.population_size,
                control_interval=interval,
            )

        elif stage == "main":
            from optimization.run_experiments import run_grid
            from optimization.variants import main_variants

            run_grid(
                variant_keys=list(main_variants()),
                split="test",
                runs=args.runs,
                generations=args.generations,
                population_size=args.population_size,
                workers=args.workers,
                control_interval=interval,
                output_dir=args.results_dir,
                agent_seeds=args.agent_seeds,
                agent_dir=agent_dir,
                bins_path=bins_path,
            )

        elif stage == "timing":
            from optimization.run_experiments import run_grid
            from optimization.variants import main_variants

            print(
                "Sequential timing replication (one run at a time) so the "
                "wall-clock comparison is not distorted by parallel load."
            )
            run_grid(
                variant_keys=list(main_variants()),
                split="test",
                runs=args.timing_runs,
                generations=args.generations,
                population_size=args.population_size,
                workers=1,
                control_interval=interval,
                output_dir=args.results_dir,
                agent_seeds=args.agent_seeds,
                agent_dir=agent_dir,
                bins_path=bins_path,
                csv_name="runs_timing.csv",
                store_trace=False,
            )

        elif stage == "analyze":
            from analysis.analyze import analyse

            analyse(results_dir=args.results_dir, csv_name="runs.csv", split="test")

        elif stage == "figures":
            from visualization.plots import make_all_figures

            make_all_figures(results_dir=args.results_dir, csv_name="runs.csv", split="test")

        print(f"-- {stage} finished in {(time.perf_counter() - stage_started) / 60:.1f} min")

    print(f"\nPipeline finished in {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
