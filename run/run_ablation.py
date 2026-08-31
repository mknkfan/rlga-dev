"""
Crossover x mutation ablation for the static GA.

A full-factorial grid over the two operator rates, on the same instances,
seeds and initial populations the main static study uses --

* crossover rate  : 0.1, 0.3, 0.5, 0.7, 0.9
* mutation rate   : 0.1, 0.01, 0.001, 0.0001

-- which is 20 combinations.  Every combination is run ``--runs`` times on
every selected instance (default: 20 runs x 10 test instances = 200 runs per
combination, 4000 runs in total).

Local search is off by default (the ``ga_ls00`` setting), so a difference
between two cells is a difference in crossover and mutation alone.
``--ls-rate`` / ``--ls-target`` put the grid on a memetic variant instead.

What the two rates mean in :class:`~optimization.ga_core.ConfigurableGA`: the
crossover rate is the probability that a selected *pair* undergoes uniform
crossover, and the mutation rate is the probability that any one *machine*
(a 3-gene block: x, y, rotation) receives Gaussian jitter and a random
re-orientation.  So on an 8-machine instance a mutation rate of 0.0001 leaves
roughly 999 offspring in 1000 completely unmutated -- the low end of this
sweep is close to a crossover-only GA, which is the point of including it.

Pairing
-------
The run seed is ``optimization.run_experiments.run_seed_for`` -- the same rule
the main study uses -- so on a given (instance, run) pair all 20 combinations
start from the *same* initial population.  The cells are therefore paired,
which is what makes the per-seed ranking in the summary meaningful.

What gets written (under ``--results-dir``, default ``results_ablation/``)
-------------------------------------------------------------------------
``raw/<combo>/<instance>_run<k>.json``  one JSON per run: the operator
                                        settings, the scalars and (by default)
                                        the convergence traces.
``runs.json`` / ``runs.csv``            every run record, scalars only.
``summary.json`` / ``summary.csv``      per combination: mean and median of
                                        fitness, evaluations and compute time.
``summary_per_instance.csv``            the same, split by instance.
``summary.md``                          the readable report.
``metadata.json``                       grid, operators, environment.

Fitness is *minimised*, and an infeasible run has fitness ``inf``.  JSON has
no ``inf`` literal, so such a run is written as ``best_fitness: null`` with
``feasible: false``; the fitness statistics are taken over feasible runs only
and every table carries the feasible count.

Examples
--------
The default grid, 8 workers::

    python -m run.run_ablation --workers 8

A quick smoke test first (seconds)::

    python -m run.run_ablation --instances 0 --runs 2 \
        --generations 20 --population-size 30 \
        --results-dir results_ablation/smoke

Resume an interrupted run (skips every (combination, instance, seed) already
on disk), then rebuild the tables from whatever is there::

    python -m run.run_ablation --workers 8 --resume
    python -m run.run_ablation --only aggregate
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from paths import DEFAULT_ABLATION_RESULTS_DIR
from core.problems import build_instances
from optimization.ga_core import ConfigurableGA, GAConfig, make_initial_population
from optimization.run_experiments import environment_info, run_seed_for

#: The grid this study asks for: 5 crossover rates x 4 mutation rates.
DEFAULT_CROSSOVER_RATES: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
DEFAULT_MUTATION_RATES: Tuple[float, ...] = (0.1, 0.01, 0.001, 0.0001)

#: Repetitions per (combination, instance).
DEFAULT_RUNS: int = 20

#: Its own directory under ``results/``; the main study is never written to
#: from here.
DEFAULT_RESULTS_DIR: str = DEFAULT_ABLATION_RESULTS_DIR

#: Rough sequential cost of one no-LS run at 300 generations x population 200,
#: taken from the archived study's timing pass.  Used only for the estimate
#: printed before a run starts.
CPU_SECONDS_PER_RUN: float = 6.7

STAGES: Tuple[str, ...] = ("run", "aggregate")

#: Traces kept in each run's JSON under ``--traces compact``: what a
#: convergence plot needs, without the RL columns a static GA never fills.
COMPACT_TRACES: Tuple[str, ...] = ("best_so_far", "gen_best", "gen_avg", "evaluations", "elapsed")

#: Scalar columns of ``runs.csv``, in order.
RUN_FIELDS: Tuple[str, ...] = (
    "combo",
    "crossover_rate",
    "mutation_rate",
    "instance",
    "instance_index",
    "run",
    "run_seed",
    "best_fitness",
    "total_distance",
    "evaluations",
    "generations",
    "execution_time",
    "cpu_time",
    "feasible",
)


# ----------------------------------------------------------------------
# the grid
# ----------------------------------------------------------------------


def _rate(value: float) -> str:
    """Compact, round-trippable rate text: 0.1 -> '0.1', 0.0001 -> '0.0001'."""
    return f"{value:g}"


def combo_key(crossover: float, mutation: float) -> str:
    """Directory / row name of one cell, e.g. ``cx0.9_mut0.001``."""
    return f"cx{_rate(crossover)}_mut{_rate(mutation)}"


def build_configs(
    base: GAConfig,
    crossover_rates: Sequence[float],
    mutation_rates: Sequence[float],
) -> Dict[str, GAConfig]:
    """``{combo key: GAConfig}`` for the full-factorial grid.

    ``base`` carries everything the cells share -- population size, generation
    budget, elitism, tournament size and the local-search setting -- so two
    cells differ in the crossover and mutation rates and nothing else.
    """
    configs: Dict[str, GAConfig] = {}
    for crossover in crossover_rates:
        for mutation in mutation_rates:
            key = combo_key(crossover, mutation)
            configs[key] = replace(
                base,
                label=f"GA cx={_rate(crossover)} mut={_rate(mutation)}",
                crossover_rate=float(crossover),
                mutation_rate=float(mutation),
            )
    return configs


# ----------------------------------------------------------------------
# running
# ----------------------------------------------------------------------


@dataclass
class AblationJob:
    """One (combination, instance, seed) run.

    Picklable, and it carries its own configuration, so a worker needs no
    variant registry and no agent.
    """

    combo: str
    config: GAConfig
    split: str
    instance_index: int
    run: int
    output_dir: str
    traces: str = "compact"


def run_json_path(output_dir: str, combo: str, instance_name: str, run: int) -> str:
    return os.path.join(output_dir, "raw", combo, f"{instance_name}_run{run}.json")


def _jsonable(value: float) -> Optional[float]:
    """``inf`` / ``nan`` -> ``None``; JSON has a literal for neither."""
    value = float(value)
    return value if math.isfinite(value) else None


def _trace_list(array: np.ndarray, decimals: int = 6) -> List[Optional[float]]:
    return [_jsonable(round(float(x), decimals)) for x in np.asarray(array).ravel()]


def execute(job: AblationJob) -> Dict:
    """Run one job and write its JSON.  Executed in a worker process."""
    instance = build_instances(job.split)[job.instance_index]
    seed = run_seed_for(job.instance_index, job.run)

    ga = ConfigurableGA(
        instance.machines,
        instance.sequence,
        instance.robot_position,
        instance.workspace_bounds,
        config=job.config,
        seed=seed,
        instance_name=instance.name,
    )
    # The same seed gives every cell of the grid the same initial population,
    # which is what pairs the combinations against each other.
    population = make_initial_population(
        instance.machines, instance.workspace_bounds, job.config.population_size, seed=seed
    )
    result = ga.optimize(initial_population=population)

    record: Dict = {
        "combo": job.combo,
        "crossover_rate": job.config.crossover_rate,
        "mutation_rate": job.config.mutation_rate,
        "instance": instance.name,
        "instance_index": job.instance_index,
        "run": job.run,
        "run_seed": seed,
        "best_fitness": _jsonable(result.best_fitness),
        "total_distance": _jsonable(result.total_distance),
        "evaluations": int(result.total_evaluations),
        "generations": int(result.generations),
        "execution_time": float(result.execution_time),
        "cpu_time": float(result.cpu_time),
        "feasible": bool(result.feasible),
    }

    payload = dict(record)
    payload["split"] = job.split
    payload["config"] = job.config.to_dict()
    if job.traces != "none":
        traces = result.trace_dict()
        wanted = list(traces) if job.traces == "full" else list(COMPACT_TRACES)
        payload["traces"] = {name: _trace_list(traces[name]) for name in wanted if name in traces}

    path = run_json_path(job.output_dir, job.combo, instance.name, job.run)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)

    return record


def _progress(done: int, total: int, started: float, every: int = 25) -> None:
    """Sparse progress line, so a redirected log stays readable."""
    if done != total and done % every:
        return
    elapsed = time.perf_counter() - started
    rate = done / max(elapsed, 1e-9)
    remaining = (total - done) / rate if rate > 0 else float("nan")
    print(
        f"\r  {done}/{total} runs  {elapsed / 60:.1f} min elapsed, "
        f"~{remaining / 60:.1f} min left   ",
        end="",
        flush=True,
    )


def load_record(path: str) -> Dict:
    """Read back the scalar half of a run JSON (the traces are ignored)."""
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return {field: payload[field] for field in RUN_FIELDS}


def run_grid(
    configs: Dict[str, GAConfig],
    split: str,
    instance_indices: Sequence[int],
    runs: int,
    workers: int,
    output_dir: str,
    traces: str,
    resume: bool,
) -> List[Dict]:
    """Run every (combination, instance, seed) and return the run records."""
    instances = build_instances(split)

    jobs: List[AblationJob] = []
    done: List[Dict] = []
    for index in instance_indices:
        for run in range(runs):
            for combo, config in configs.items():
                path = run_json_path(output_dir, combo, instances[index].name, run)
                if resume and os.path.exists(path):
                    done.append(load_record(path))
                    continue
                jobs.append(
                    AblationJob(
                        combo=combo,
                        config=config,
                        split=split,
                        instance_index=index,
                        run=run,
                        output_dir=output_dir,
                        traces=traces,
                    )
                )

    os.makedirs(output_dir, exist_ok=True)
    if done:
        print(f"Resuming: {len(done)} runs already on disk, {len(jobs)} left to do.")
    print(
        f"Running {len(configs)} combinations x {len(instance_indices)} '{split}' instances x "
        f"{runs} seeds = {len(jobs)} runs "
        f"({'sequential' if workers <= 1 else f'{workers} workers'})"
    )

    started = time.perf_counter()
    rows: List[Dict] = list(done)
    if jobs and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for k, row in enumerate(pool.map(execute, jobs, chunksize=1), start=1):
                rows.append(row)
                _progress(k, len(jobs), started)
    else:
        for k, job in enumerate(jobs, start=1):
            rows.append(execute(job))
            _progress(k, len(jobs), started)
    print()
    print(f"Finished {len(jobs)} runs in {(time.perf_counter() - started) / 60:.1f} min")
    return rows


def load_metadata(output_dir: str) -> Optional[Dict]:
    """The ``metadata.json`` an earlier ``run`` stage left, if there is one.

    ``base_config`` comes back ready to hand to :class:`GAConfig`: JSON has no
    tuples, so the two fields the dataclass declares as tuples are restored.
    """
    path = os.path.join(output_dir, "metadata.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        metadata = json.load(fh)
    config = metadata.get("base_config")
    if isinstance(config, dict):
        config["ls_schedule"] = tuple(tuple(pair) for pair in config.get("ls_schedule", ()))
        actions = config.get("allowed_actions")
        config["allowed_actions"] = tuple(actions) if actions else None
    return metadata


def collect_records(output_dir: str) -> List[Dict]:
    """Every run record under ``raw/``, for an aggregate-only invocation."""
    raw_root = os.path.join(output_dir, "raw")
    if not os.path.isdir(raw_root):
        raise SystemExit(f"no raw runs under {raw_root}. Run the 'run' stage before aggregating.")
    rows: List[Dict] = []
    for combo in sorted(os.listdir(raw_root)):
        combo_dir = os.path.join(raw_root, combo)
        if not os.path.isdir(combo_dir):
            continue
        for name in sorted(os.listdir(combo_dir)):
            if name.endswith(".json"):
                rows.append(load_record(os.path.join(combo_dir, name)))
    if not rows:
        raise SystemExit(f"no run JSON files found under {raw_root}.")
    return rows


# ----------------------------------------------------------------------
# summarising
# ----------------------------------------------------------------------


def _stats(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    """Mean / median / sd / min / max over the finite values only."""
    finite = np.array(
        [float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=float
    )
    if finite.size == 0:
        return {"mean": None, "median": None, "sd": None, "min": None, "max": None}
    return {
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "sd": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def mean_ranks(rows: Sequence[Dict]) -> Dict[str, Optional[float]]:
    """Average rank of each combination across the paired (instance, seed) sets.

    Within one (instance, seed) every combination started from the same
    initial population, so the cells can be ranked directly: 1 is the best
    (lowest) fitness, tied runs share the average rank of their block, and an
    infeasible run sorts last.  Averaging those ranks compares the cells
    without pooling fitness values across instances, whose scales differ.
    """
    groups: Dict[Tuple[int, int], List[Tuple[str, float]]] = {}
    for row in rows:
        fitness = row["best_fitness"]
        value = float(fitness) if fitness is not None else float("inf")
        groups.setdefault((row["instance_index"], row["run"]), []).append((row["combo"], value))

    totals: Dict[str, List[float]] = {}
    for members in groups.values():
        order = sorted(members, key=lambda item: item[1])
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and order[j + 1][1] == order[i][1]:
                j += 1
            shared = (i + j) / 2.0 + 1.0  # average rank of the tied block
            for combo, _ in order[i : j + 1]:
                totals.setdefault(combo, []).append(shared)
            i = j + 1
    return {combo: float(np.mean(ranks)) if ranks else None for combo, ranks in totals.items()}


def summarise(rows: Sequence[Dict], by_instance: bool = False) -> List[Dict]:
    """Per-combination (optionally per-instance) mean / median tables."""
    ranks = {} if by_instance else mean_ranks(rows)

    groups: Dict[Tuple, List[Dict]] = {}
    for row in rows:
        key = (row["combo"], row["instance"]) if by_instance else (row["combo"],)
        groups.setdefault(key, []).append(row)

    table: List[Dict] = []
    for members in groups.values():
        first = members[0]
        feasible = [r for r in members if r["feasible"]]
        entry: Dict = {
            "combo": first["combo"],
            "crossover_rate": first["crossover_rate"],
            "mutation_rate": first["mutation_rate"],
        }
        if by_instance:
            entry["instance"] = first["instance"]
        entry.update(
            {
                "n_runs": len(members),
                "n_feasible": len(feasible),
                "feasible_rate": len(feasible) / len(members),
            }
        )
        for name, field in (
            ("fitness", "best_fitness"),
            ("evaluations", "evaluations"),
            ("execution_time", "execution_time"),
            ("cpu_time", "cpu_time"),
        ):
            # Quality is only defined on feasible runs; effort and time are
            # measured on every run that happened.
            source = feasible if field == "best_fitness" else members
            for stat, value in _stats(r[field] for r in source).items():
                entry[f"{name}_{stat}"] = value
        if not by_instance:
            entry["mean_rank"] = ranks.get(first["combo"])
        table.append(entry)

    table.sort(key=lambda e: (-e["crossover_rate"], -e["mutation_rate"], e.get("instance", "")))
    return table


def write_csv(path: str, rows: Sequence[Dict], fields: Optional[Sequence[str]] = None) -> None:
    fields = list(fields or (rows[0].keys() if rows else []))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _cell(value: Optional[float], spec: str = ".4f") -> str:
    return "-" if value is None else format(value, spec)


def write_report(
    path: str,
    summary: Sequence[Dict],
    rows: Sequence[Dict],
    crossover_rates: Sequence[float],
    mutation_rates: Sequence[float],
    base: GAConfig,
    split: str,
    runs: int,
    n_instances: int,
) -> None:
    """The readable report: the main table, then a grid per operator pair."""
    indexed = {entry["combo"]: entry for entry in summary}
    best = min(
        (e for e in summary if e["fitness_mean"] is not None),
        key=lambda e: e["fitness_mean"],
        default=None,
    )
    ls_text = (
        "off" if base.ls_target == "none" else f"{base.ls_rate:.0%} on {base.ls_target}"
    )

    lines: List[str] = []
    lines.append("# Crossover x mutation ablation - static GA\n")
    lines.append(
        f"Grid: {len(crossover_rates)} crossover rates x {len(mutation_rates)} mutation rates "
        f"= {len(crossover_rates) * len(mutation_rates)} combinations, "
        f"{runs} runs on each of {n_instances} '{split}' instances "
        f"({len(rows)} runs recorded).\n"
    )
    lines.append(
        f"Fixed operators: population {base.population_size}, {base.generations} generations, "
        f"elitism {base.elitism_rate}, tournament k={base.tournament_k}, local search {ls_text}"
        f"{'' if base.ls_target == 'none' else f', {base.ls_iterations} iterations'}.\n"
    )
    lines.append(
        "**Fitness is minimised** (lower is better), and infeasible runs are excluded from the "
        "fitness statistics; `n_feasible` reports how many runs in each cell were feasible. "
        "Evaluation counts and times cover every run.\n"
    )
    lines.append(
        "The crossover rate is the probability that a selected pair undergoes uniform crossover; "
        "the mutation rate is the probability that any one machine (x, y, rotation) receives "
        "Gaussian jitter and a random re-orientation.\n"
    )
    lines.append(
        "`mean_rank` ranks the combinations against each other within every (instance, seed) "
        "pair - all of which start from the same initial population - and averages those ranks. "
        "It is the scale-free comparison; the pooled fitness columns mix instances whose "
        "fitness scales differ.\n"
    )
    if best is not None:
        lines.append(
            f"Lowest mean fitness: **crossover {_rate(best['crossover_rate'])}, "
            f"mutation {_rate(best['mutation_rate'])}** "
            f"(mean {best['fitness_mean']:.4f}, median {_cell(best['fitness_median'])}).\n"
        )

    lines.append("\n## Per combination\n")
    lines.append(
        "| crossover | mutation | n | feasible | fitness mean | fitness median | "
        "evals mean | evals median | wall mean (s) | wall median (s) | "
        "CPU mean (s) | CPU median (s) | mean rank |"
    )
    lines.append("|" + "---|" * 13)
    for entry in summary:
        lines.append(
            "| "
            + " | ".join(
                (
                    _rate(entry["crossover_rate"]),
                    _rate(entry["mutation_rate"]),
                    str(entry["n_runs"]),
                    f"{entry['n_feasible']}/{entry['n_runs']}",
                    _cell(entry["fitness_mean"]),
                    _cell(entry["fitness_median"]),
                    _cell(entry["evaluations_mean"], ".0f"),
                    _cell(entry["evaluations_median"], ".0f"),
                    _cell(entry["execution_time_mean"], ".2f"),
                    _cell(entry["execution_time_median"], ".2f"),
                    _cell(entry["cpu_time_mean"], ".2f"),
                    _cell(entry["cpu_time_median"], ".2f"),
                    _cell(entry["mean_rank"], ".2f"),
                )
            )
            + " |"
        )

    for title, field, spec in (
        ("Mean fitness", "fitness_mean", ".4f"),
        ("Median fitness", "fitness_median", ".4f"),
        ("Mean rank", "mean_rank", ".2f"),
    ):
        lines.append(f"\n## {title} - crossover (rows) x mutation (columns)\n")
        lines.append(
            "| crossover \\ mutation | " + " | ".join(_rate(m) for m in mutation_rates) + " |"
        )
        lines.append("|" + "---|" * (len(mutation_rates) + 1))
        for crossover in crossover_rates:
            cells = [
                _cell(indexed.get(combo_key(crossover, m), {}).get(field), spec)
                for m in mutation_rates
            ]
            lines.append(f"| **{_rate(crossover)}** | " + " | ".join(cells) + " |")

    lines.append(
        "\n---\n\nWall-clock time is inflated and noisy whenever the grid was run with "
        "`--workers > 1`, because parallel runs share the machine; CPU time and the evaluation "
        "count are the reliable effort measures in that case.\n"
    )

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def write_outputs(
    output_dir: str,
    rows: Sequence[Dict],
    crossover_rates: Sequence[float],
    mutation_rates: Sequence[float],
    base: GAConfig,
    split: str,
    runs: int,
    n_instances: int,
) -> List[Dict]:
    """``runs.*``, ``summary.*`` and the report; returns the pooled summary."""
    rows = sorted(rows, key=lambda r: (r["combo"], r["instance_index"], r["run"]))
    summary = summarise(rows)
    per_instance = summarise(rows, by_instance=True)

    with open(os.path.join(output_dir, "runs.json"), "w", encoding="utf-8") as fh:
        json.dump(list(rows), fh, indent=1)
    write_csv(os.path.join(output_dir, "runs.csv"), rows, RUN_FIELDS)
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    write_csv(os.path.join(output_dir, "summary.csv"), summary)
    write_csv(os.path.join(output_dir, "summary_per_instance.csv"), per_instance)
    write_report(
        os.path.join(output_dir, "summary.md"),
        summary,
        rows,
        crossover_rates,
        mutation_rates,
        base,
        split,
        runs,
        n_instances,
    )

    print(f"Wrote {len(rows)} run records to {os.path.join(output_dir, 'runs.json')} (and runs.csv)")
    print(f"Wrote {len(summary)} combination summaries to {os.path.join(output_dir, 'summary.csv')}")
    print(f"Wrote the report to {os.path.join(output_dir, 'summary.md')}")
    return summary


# ----------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------


def _probability(name: str):
    """Parser for a rate given on the command line."""

    def parse(text: str) -> float:
        try:
            value = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{name} expects a number, got {text!r}")
        if not 0.0 <= value <= 1.0:
            raise argparse.ArgumentTypeError(f"{name} must lie in [0, 1], got {value}")
        return value

    return parse


def _positive_int(name: str):
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{name} expects an integer, got {text!r}")
        if value < 1:
            raise argparse.ArgumentTypeError(f"{name} must be >= 1, got {value}")
        return value

    return parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m run.run_ablation",
        description="Full-factorial crossover x mutation ablation for the static GA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    grid = parser.add_argument_group("the grid")
    grid.add_argument(
        "--crossover-rates",
        type=_probability("--crossover-rates"),
        nargs="+",
        default=list(DEFAULT_CROSSOVER_RATES),
        help="crossover rates to sweep",
    )
    grid.add_argument(
        "--mutation-rates",
        type=_probability("--mutation-rates"),
        nargs="+",
        default=list(DEFAULT_MUTATION_RATES),
        help="mutation rates to sweep",
    )
    grid.add_argument(
        "--runs",
        type=_positive_int("--runs"),
        default=DEFAULT_RUNS,
        help="repetitions of each combination on each instance",
    )
    grid.add_argument("--split", default="test", choices=("test", "train", "generalization"))
    grid.add_argument(
        "--instances",
        type=int,
        nargs="+",
        default=None,
        help="instance indices to use (default: the whole split)",
    )

    operators = parser.add_argument_group("fixed GA operators")
    operators.add_argument("--population-size", type=_positive_int("--population-size"), default=200)
    operators.add_argument("--generations", type=_positive_int("--generations"), default=300)
    operators.add_argument("--elitism-rate", type=_probability("--elitism-rate"), default=0.1)
    operators.add_argument("--tournament-k", type=_positive_int("--tournament-k"), default=3)
    operators.add_argument(
        "--ls-rate",
        type=_probability("--ls-rate"),
        default=0.0,
        help="local-search rate; 0 keeps the grid on the no-LS variant",
    )
    operators.add_argument("--ls-target", default="none", choices=("none", "offspring", "elites"))
    operators.add_argument("--ls-iterations", type=_positive_int("--ls-iterations"), default=5)

    execution = parser.add_argument_group("execution and output")
    execution.add_argument(
        "--workers",
        type=_positive_int("--workers"),
        default=1,
        help="parallel worker processes (wall-clock times get noisy above 1)",
    )
    execution.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    execution.add_argument(
        "--traces",
        choices=("none", "compact", "full"),
        default="compact",
        help="how much per-generation history each run's JSON keeps",
    )
    execution.add_argument(
        "--resume",
        action="store_true",
        help="skip any (combination, instance, seed) whose JSON already exists",
    )
    execution.add_argument(
        "--only",
        nargs="+",
        choices=STAGES,
        default=None,
        help="run a subset of the stages; 'aggregate' rebuilds the tables from raw/",
    )
    execution.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and the cost estimate, then stop",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Caught here because otherwise it surfaces generations deep as an opaque
    # "Sample larger than population" from random.sample inside selection.
    if args.tournament_k > args.population_size:
        raise SystemExit(
            f"--tournament-k ({args.tournament_k}) cannot exceed --population-size "
            f"({args.population_size}): selection draws that many distinct individuals."
        )
    if args.ls_rate > 0.0 and args.ls_target == "none":
        raise SystemExit("--ls-rate > 0 needs --ls-target offspring or elites.")

    n_available = len(build_instances(args.split))
    instance_indices = list(range(n_available)) if args.instances is None else list(args.instances)
    out_of_range = [i for i in instance_indices if not 0 <= i < n_available]
    if out_of_range:
        raise SystemExit(
            f"instance index/indices {out_of_range} out of range for split '{args.split}' "
            f"(0..{n_available - 1})"
        )

    base = GAConfig(
        population_size=args.population_size,
        generations=args.generations,
        elitism_rate=args.elitism_rate,
        tournament_k=args.tournament_k,
        ls_iterations=args.ls_iterations,
        ls_target=args.ls_target if args.ls_rate > 0.0 else "none",
        ls_controller="static",
        ls_rate=args.ls_rate,
    )
    configs = build_configs(base, args.crossover_rates, args.mutation_rates)
    stages = list(args.only or STAGES)

    total_runs = len(configs) * len(instance_indices) * args.runs
    scale = (args.generations * args.population_size) / (300 * 200)
    estimate = total_runs * CPU_SECONDS_PER_RUN * scale / 3600.0
    ls_text = "off" if base.ls_target == "none" else f"{base.ls_rate:.0%} on {base.ls_target}"

    print("Crossover x mutation ablation")
    print(f"Results directory : {args.results_dir}")
    print(f"Stages            : {', '.join(stages)}")
    print(f"Crossover rates   : {', '.join(_rate(r) for r in args.crossover_rates)}")
    print(f"Mutation rates    : {', '.join(_rate(r) for r in args.mutation_rates)}")
    print(
        f"Grid              : {len(configs)} combinations x {len(instance_indices)} "
        f"'{args.split}' instances x {args.runs} runs = {total_runs} runs"
    )
    print(
        f"Fixed operators   : pop={args.population_size} gens={args.generations} "
        f"elitism={args.elitism_rate} tournament_k={args.tournament_k} ls={ls_text}"
    )
    print(
        f"Cost estimate     : ~{estimate:.1f} CPU-hours "
        f"(~{estimate / args.workers:.1f} h wall-clock on {args.workers} worker(s))"
    )
    print()

    if args.dry_run:
        print("--dry-run: nothing was executed.")
        return

    os.makedirs(args.results_dir, exist_ok=True)
    started = time.perf_counter()

    rows: List[Dict] = []
    if "run" in stages:
        rows = run_grid(
            configs=configs,
            split=args.split,
            instance_indices=instance_indices,
            runs=args.runs,
            workers=args.workers,
            output_dir=args.results_dir,
            traces=args.traces,
            resume=args.resume,
        )
        metadata = {
            "crossover_rates": list(args.crossover_rates),
            "mutation_rates": list(args.mutation_rates),
            "combinations": sorted(configs),
            "split": args.split,
            "instance_indices": instance_indices,
            "runs_per_instance": args.runs,
            "n_runs": len(rows),
            "traces": args.traces,
            "seed_rule": "run_seed = 42 + 100 * instance_index + run; shared by every combination",
            "fitness_direction": (
                "minimised; an infeasible run is written as null with feasible=false"
            ),
            "base_config": base.to_dict(),
            "configs": {key: config.to_dict() for key, config in configs.items()},
            "environment": environment_info(args.workers),
        }
        meta_path = os.path.join(args.results_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, default=str)
        print(f"Wrote configuration metadata to {meta_path}")

    if "aggregate" in stages:
        crossover_rates, mutation_rates = args.crossover_rates, args.mutation_rates
        split, runs, n_instances = args.split, args.runs, len(instance_indices)
        # An aggregate-only invocation reads the runs back off disk, so it
        # covers whatever is there -- including earlier, interrupted passes.
        # The report must then describe the settings those runs actually used,
        # not this invocation's defaults, so they come from the metadata.
        if not rows:
            rows = collect_records(args.results_dir)
            recorded = load_metadata(args.results_dir)
            if recorded is not None:
                base = GAConfig(**recorded["base_config"])
                crossover_rates = recorded["crossover_rates"]
                mutation_rates = recorded["mutation_rates"]
                split = recorded["split"]
                runs = recorded["runs_per_instance"]
                n_instances = len(recorded["instance_indices"])
            else:
                print(
                    "No metadata.json in the results directory: the report will describe this "
                    "invocation's operator settings, which may not be the ones the runs used."
                )
        write_outputs(
            output_dir=args.results_dir,
            rows=rows,
            crossover_rates=crossover_rates,
            mutation_rates=mutation_rates,
            base=base,
            split=split,
            runs=runs,
            n_instances=n_instances,
        )

    print(f"\nDone in {(time.perf_counter() - started) / 60:.1f} min.")


if __name__ == "__main__":
    main()
