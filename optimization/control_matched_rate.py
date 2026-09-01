"""Controls for the offspring-target result: does adaptation add anything beyond its mean rate?

`rlga_offspring` uses a time-averaged local-search rate of 0.426 and therefore
~129k evaluations, against 141k for the static 0.5 baseline.  Section 6.8 of the
walkthrough reports that as the study's one positive result, but the saving is
fully explained by the mean rate (cost is linear in it to within 0.05 %).  Two
controls separate the possible explanations:

  A. ``static_ls43``  -- a static GA at ls_rate = 0.426, i.e. the RL variant's own
     average.  Same expected budget, no adaptation.  If the RL variant does not
     beat this, adaptation contributes nothing beyond where the rate happened to
     settle.
  B. ``random_ctrl``  -- the RL control loop with a zero Q-table and epsilon = 1,
     so actions are drawn uniformly and nothing is learned.  Isolates the
     clipped-accumulator dynamics from the learned policy.

Both are run on exactly the (instance, seed) pairs used by the main grid, with
the same seed rule and the same initial populations, so every comparison is
paired against the stored runs.

Outputs go to ``results/control_matched/``.

Run:  python -m optimization.control_matched_rate [--workers N]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.problems import build_instances  # noqa: E402
from optimization.ga_core import (  # noqa: E402
    ConfigurableGA,
    GAConfig,
    make_initial_population,
)
from optimization.rl_control import QLearningAgent, StateBins  # noqa: E402
from run.paths import DEFAULT_BINS_PATH, DEFAULT_RESULTS_DIR  # noqa: E402

OUT_DIR = os.path.join(DEFAULT_RESULTS_DIR, "control_matched")
TRACE_DIR = os.path.join(OUT_DIR, "control_traces")
BINS = DEFAULT_BINS_PATH

#: The RL variant's own time-averaged rate, measured from its stored traces.
MATCHED_RATE = 0.426


def configs() -> dict:
    shared = dict(generations=300, population_size=200, ls_target="offspring")
    return {
        "static_ls43": GAConfig(
            label=f"GA {MATCHED_RATE:.0%} LS [LS on offspring]",
            ls_controller="static", ls_rate=MATCHED_RATE, **shared),
        "random_ctrl": GAConfig(
            label="Random control [LS on offspring]",
            ls_controller="rl", ls_rate=0.1, control_interval=1,
            rl_learn_online=False, **shared),
    }


def run_one(job: tuple) -> dict:
    key, instance_index, run = job
    instance = build_instances("test")[instance_index]
    seed = 42 + 100 * instance_index + run
    config = configs()[key]

    agent = bins = None
    if config.ls_controller == "rl":
        bins = StateBins.load(BINS)
        # Zero Q-table + epsilon 1.0 => uniform random action, and learning is off,
        # so the table stays zero for the whole run.
        agent = QLearningAgent(n_states=bins.n_states, n_actions=7, epsilon=1.0, seed=seed)

    ga = ConfigurableGA(instance.machines, instance.sequence, instance.robot_position,
                        instance.workspace_bounds, config=config, rl_agent=agent,
                        state_bins=bins, seed=seed, instance_name=instance.name)
    population = make_initial_population(instance.machines, instance.workspace_bounds,
                                         config.population_size, seed=seed)
    result = ga.optimize(initial_population=population)

    os.makedirs(os.path.join(TRACE_DIR, key), exist_ok=True)
    np.savez_compressed(os.path.join(TRACE_DIR, key, f"{instance.name}_run{run}.npz"),
                        best_so_far=result.best_so_far, evaluations=result.evaluations,
                        elapsed=result.elapsed, ls_rate=result.ls_rate)
    return {
        "variant": key, "label": config.label, "instance": instance.name,
        "instance_index": instance_index, "seed": run, "run_seed": seed,
        "best_fitness": result.best_fitness, "execution_time": result.execution_time,
        "cpu_time": result.cpu_time, "evaluations": result.total_evaluations,
        "mean_ls_rate": float(np.nanmean(result.ls_rate)), "feasible": int(result.feasible),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = parser.parse_args()

    jobs = [(key, index, run) for key in configs() for index in range(10) for run in range(10)]
    print(f"{len(jobs)} runs ({len(configs())} controls x 10 instances x 10 seeds), "
          f"{args.workers} workers")
    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, row in enumerate(pool.map(run_one, jobs, chunksize=1), start=1):
            rows.append(row)
            if k % 25 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} done ({time.perf_counter() - started:.0f}s)", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "control_runs.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
