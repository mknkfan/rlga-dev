"""
Main experiment driver: runs the full grid of variants (see
:mod:`optimization.variants`) over every instance and seed.

All variants sharing an (instance, seed) pair start from the same initial
population and RNG seed (common random numbers), so comparisons across
variants are paired. RL runs rotate through the independently trained
agents. Each run stores its full anytime trace (best-so-far fitness against
generations, cumulative evaluations and cumulative time) plus the run
configuration, seeds and environment, written next to the summary CSV.

Run:  python -m optimization.run_experiments --runs 10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from core.problems import build_instances, split_size
from optimization.ga_core import ConfigurableGA, make_initial_population
from optimization.rl_control import QLearningAgent, StateBins
from optimization.variants import DEFAULT_CONTROL_INTERVAL, main_variants, resolve
from run.paths import DEFAULT_RESULTS_DIR
from train.calibrate_state_bins import DEFAULT_OUTPUT as DEFAULT_BINS_PATH
from train.train_agents import AGENT_DIR, DEFAULT_AGENT_SEEDS, agent_path

RESULTS_DIR = DEFAULT_RESULTS_DIR

CSV_COLUMNS = (
    "variant",
    "label",
    "split",
    "instance",
    "instance_index",
    "seed",
    "run_seed",
    "agent_seed",
    "best_fitness",
    "total_distance",
    "execution_time",
    "cpu_time",
    "evaluations",
    "generations",
    "feasible",
    "mean_ls_rate",
    "final_mutation_rate",
    "final_crossover_rate",
)


def run_seed_for(instance_index: int, run: int) -> int:
    """Seed shared by every variant on a given (instance, run) pair."""
    return 42 + instance_index * 100 + run


@dataclass
class Job:
    variant_key: str
    split: str
    instance_index: int
    run: int
    generations: int
    population_size: int
    agent_seeds: Tuple[int, ...]
    agent_dir: str
    bins_path: str
    output_dir: str
    control_interval: int = DEFAULT_CONTROL_INTERVAL
    store_trace: bool = True


def _execute(job: Job) -> Dict:
    """Run one (variant, instance, seed) combination.  Executed in a worker."""
    instance = build_instances(job.split)[job.instance_index]
    config = resolve(
        [job.variant_key],
        generations=job.generations,
        population_size=job.population_size,
        control_interval=job.control_interval,
    )[job.variant_key]

    seed = run_seed_for(job.instance_index, job.run)

    agent: Optional[QLearningAgent] = None
    bins: Optional[StateBins] = None
    agent_seed = -1
    if config.ls_controller == "rl":
        bins = StateBins.load(job.bins_path)
        # Rotate through the independently trained agents so the reported
        # numbers are not those of one lucky training repetition.
        agent_seed = job.agent_seeds[job.run % len(job.agent_seeds)]
        # A variant may deliberately re-use another variant's trained agent
        # (see GAConfig.agent_key), e.g. to evaluate the same policy under a
        # different control interval.
        path = agent_path(config.agent_key or job.variant_key, agent_seed, job.agent_dir)
        agent = QLearningAgent.load(path, epsilon=0.05, seed=seed)
        agent.lr = 0.05

    ga = ConfigurableGA(
        instance.machines,
        instance.sequence,
        instance.robot_position,
        instance.workspace_bounds,
        config=config,
        rl_agent=agent,
        state_bins=bins,
        seed=seed,
        instance_name=instance.name,
    )
    population = make_initial_population(
        instance.machines, instance.workspace_bounds, job.population_size, seed=seed
    )
    result = ga.optimize(initial_population=population)

    if job.store_trace:
        trace_dir = os.path.join(job.output_dir, "raw", job.variant_key)
        os.makedirs(trace_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(trace_dir, f"{instance.name}_run{job.run}.npz"),
            **result.trace_dict(),
            best_fitness=result.best_fitness,
            total_distance=result.total_distance,
            execution_time=result.execution_time,
            cpu_time=result.cpu_time,
            total_evaluations=result.total_evaluations,
            q_table=result.q_table if result.q_table is not None else np.zeros(0),
        )

    return {
        "variant": job.variant_key,
        "label": config.label,
        "split": job.split,
        "instance": instance.name,
        "instance_index": job.instance_index,
        "seed": job.run,
        "run_seed": seed,
        "agent_seed": agent_seed,
        "best_fitness": result.best_fitness,
        "total_distance": result.total_distance,
        "execution_time": result.execution_time,
        "cpu_time": result.cpu_time,
        "evaluations": result.total_evaluations,
        "generations": result.generations,
        "feasible": int(result.feasible),
        "mean_ls_rate": float(np.nanmean(result.ls_rate)),
        "final_mutation_rate": float(result.mutation_rate[-1]),
        "final_crossover_rate": float(result.crossover_rate[-1]),
    }


def environment_info(workers: int) -> Dict:
    """Software / hardware description for the reproducibility statement (9b)."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "parallel_workers": workers,
        "timing_note": (
            "Wall-clock time was measured with time.perf_counter and CPU time with "
            "time.process_time. When parallel_workers > 1 several runs share the machine, "
            "which inflates and adds noise to wall-clock time; CPU time is the more "
            "reliable effort measure in that case, and the evaluation count is "
            "hardware-independent."
        ),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def run_grid(
    variant_keys: Sequence[str],
    split: str = "test",
    runs: int = 10,
    generations: int = 300,
    population_size: int = 200,
    workers: int = 1,
    control_interval: int = DEFAULT_CONTROL_INTERVAL,
    output_dir: str = RESULTS_DIR,
    agent_seeds: Sequence[int] = DEFAULT_AGENT_SEEDS,
    agent_dir: str = AGENT_DIR,
    bins_path: str = DEFAULT_BINS_PATH,
    csv_name: str = "runs.csv",
    store_trace: bool = True,
    agent_dirs: Optional[Mapping[str, str]] = None,
) -> List[Dict]:
    """Run every (variant, instance, seed) combination and write the results.

    ``agent_dirs`` overrides ``agent_dir`` per variant key.  It is what lets a
    single grid compare several independently trained agent *sets* against one
    shared set of static baselines: the baselines load no agent at all, so
    running them once rather than once per set is not an approximation.
    """
    instances = build_instances(split)
    jobs = [
        Job(
            variant_key=variant_key,
            split=split,
            instance_index=index,
            run=run,
            generations=generations,
            population_size=population_size,
            agent_seeds=tuple(agent_seeds),
            agent_dir=(agent_dirs or {}).get(variant_key, agent_dir),
            bins_path=bins_path,
            output_dir=output_dir,
            control_interval=control_interval,
            store_trace=store_trace,
        )
        for index in range(len(instances))
        for run in range(runs)
        for variant_key in variant_keys
    ]

    os.makedirs(output_dir, exist_ok=True)
    print(
        f"Running {len(variant_keys)} variants x {len(instances)} '{split}' instances x "
        f"{runs} seeds = {len(jobs)} runs "
        f"({'sequential' if workers <= 1 else f'{workers} workers'})"
    )

    started = time.perf_counter()
    rows: List[Dict] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for k, row in enumerate(pool.map(_execute, jobs, chunksize=1), start=1):
                rows.append(row)
                _progress(k, len(jobs), started)
    else:
        for k, job in enumerate(jobs, start=1):
            rows.append(_execute(job))
            _progress(k, len(jobs), started)

    print()
    csv_path = os.path.join(output_dir, csv_name)
    write_rows(csv_path, rows)
    print(f"Wrote {len(rows)} run records to {csv_path} ({time.perf_counter() - started:.0f}s)")

    configs = resolve(
        variant_keys,
        generations=generations,
        population_size=population_size,
        control_interval=control_interval,
    )
    metadata = {
        "split": split,
        "runs_per_instance": runs,
        "generations": generations,
        "population_size": population_size,
        "control_interval": control_interval,
        "agent_seeds": list(agent_seeds),
        "agent_dir": agent_dir,
        "agent_dirs": dict(agent_dirs) if agent_dirs else None,
        "state_bins": json.load(open(bins_path, encoding="utf-8"))
        if os.path.exists(bins_path)
        else None,
        "variants": {key: config.to_dict() for key, config in configs.items()},
        "seed_rule": "run_seed = 42 + 100 * instance_index + run; shared by all variants",
        "split_sizes": {
            name: split_size(name) for name in ("train", "test")
        },
        "environment": environment_info(workers),
        "n_runs": len(rows),
    }
    meta_path = os.path.join(output_dir, os.path.splitext(csv_name)[0] + "_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)
    print(f"Wrote configuration metadata to {meta_path}")

    return rows


def _progress(done: int, total: int, started: float, every: int = 25) -> None:
    """Progress line, printed sparsely so redirected logs stay readable."""
    if done != total and done % every:
        return
    elapsed = time.perf_counter() - started
    rate = done / max(elapsed, 1e-9)
    remaining = (total - done) / max(rate, 1e-9)
    print(
        f"  {done}/{total} runs done | elapsed {elapsed / 60:.1f} min | "
        f"eta {remaining / 60:.1f} min",
        flush=True,
    )


def write_rows(path: str, rows: Sequence[Dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in CSV_COLUMNS})


def read_rows(path: str) -> List[Dict]:
    """Read a runs CSV back, converting numeric columns."""
    numeric_int = {"instance_index", "seed", "run_seed", "agent_seed", "evaluations", "generations", "feasible"}
    numeric_float = {
        "best_fitness",
        "total_distance",
        "execution_time",
        "cpu_time",
        "mean_ls_rate",
        "final_mutation_rate",
        "final_crossover_rate",
    }
    rows: List[Dict] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            parsed = dict(row)
            for key in numeric_int:
                if key in parsed:
                    parsed[key] = int(parsed[key])
            for key in numeric_float:
                if key in parsed:
                    parsed[key] = float(parsed[key])
            rows.append(parsed)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=None, help="default: all main variants")
    parser.add_argument("--split", default="test")
    parser.add_argument("--runs", type=int, default=10, help="independent seeds per instance")
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--population-size", type=int, default=200)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="1 (default) keeps wall-clock timings clean; >1 is much faster but "
        "makes wall-clock time noisier (CPU time and evaluation counts stay valid)",
    )
    parser.add_argument(
        "--control-interval",
        type=int,
        default=DEFAULT_CONTROL_INTERVAL,
        help="generations between RL control decisions",
    )
    parser.add_argument("--output-dir", default=RESULTS_DIR)
    parser.add_argument("--csv-name", default="runs.csv")
    parser.add_argument("--agent-seeds", type=int, nargs="+", default=list(DEFAULT_AGENT_SEEDS))
    parser.add_argument("--agent-dir", default=AGENT_DIR)
    parser.add_argument("--bins", default=DEFAULT_BINS_PATH)
    parser.add_argument("--no-trace", action="store_true", help="skip storing per-generation traces")
    args = parser.parse_args()

    variant_keys = args.variants or list(
        main_variants(
            generations=args.generations,
            population_size=args.population_size,
            control_interval=args.control_interval,
        )
    )

    rows = run_grid(
        variant_keys=variant_keys,
        split=args.split,
        runs=args.runs,
        generations=args.generations,
        population_size=args.population_size,
        workers=args.workers,
        control_interval=args.control_interval,
        output_dir=args.output_dir,
        agent_seeds=tuple(args.agent_seeds),
        agent_dir=args.agent_dir,
        bins_path=args.bins,
        csv_name=args.csv_name,
        store_trace=not args.no_trace,
    )


if __name__ == "__main__":
    main()
