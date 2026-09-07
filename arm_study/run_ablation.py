"""
Crossover x mutation ablation for the static GA on the cycle-time objective.

A full-factorial grid over the two operator rates, on the same instances,
seeds and initial populations ``run_ga.py`` uses --

* crossover rate : 0.1, 0.3, 0.5, 0.7, 0.9
* mutation rate  : 0.3, 0.2, 0.1, 0.05, 0.01, 0.001

-- which is 30 combinations.  Every combination is run ``--runs`` times on
every selected instance (default: 20 runs x 10 'test' instances = 200 runs per
combination, 6000 runs in total).

    python -m arm_study.run_ablation --workers 12
    python -m arm_study.run_ablation --dry-run          # plan and cost only
    python -m arm_study.run_ablation --workers 12 --resume
    python -m arm_study.run_ablation --only aggregate   # rebuild the tables
    python -m arm_study.run_ablation --repair           # sweep with the repair on

What the two rates mean in :class:`ga.ConfigurableGA`: the crossover rate is
the probability that a selected *pair* undergoes uniform crossover, and the
mutation rate is the probability that any one *machine* (a 3-gene block: x, y,
rotation) receives Gaussian jitter and a re-orientation.  So on a 6-machine
instance a mutation rate of 0.001 leaves roughly 994 offspring in 1000
untouched -- the low end of the sweep is close to a crossover-only GA, which is
why it is in the grid.

Pairing
-------
The run seed is ``run_ga.run_seed_for`` -- ``42 + 100 * instance_index + run``
-- so on a given (instance, run) pair all 30 combinations start from the *same*
initial population.  The cells are therefore paired, which is what makes the
per-seed ranking in the summary meaningful.

Everything is written under ``--results-dir`` (default
``arm_study/results_ablation/``)::

    raw/<combo>/<instance>_run<k>.json  one JSON per run: operator settings,
                                        scalars, the winning chromosome and
                                        (by default) compact traces
    runs.csv / runs.json                every run, scalars only
    summary.csv / summary.json          per combination
    summary_per_instance.csv            per combination x instance
    summary.md                          the readable report
    metadata.json                       grid, operators, instances, environment

Cycle time is **minimised** (seconds) and an infeasible run scores ``+inf``.
JSON has no literal for that, so such a run is written as ``cycle_time: null``
with ``feasible: false``; the cycle-time statistics are taken over feasible
runs only and every table carries the feasible count.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Run as a script or as a module: either way arm_study/ must be importable by
# its own modules, which import each other by bare name.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluator import CycleTimeEvaluator  # noqa: E402
from ga import ConfigurableGA, GAConfig, make_initial_population  # noqa: E402
from problems import ALL_SPLITS, build_instances, describe  # noqa: E402
from robot import (  # noqa: E402
    PUMA_JOINT_INDEX, PUMA_TRAVEL, ArmLimits, Puma560Arm,
)
# The main study owns the seed rule and the table helpers; sharing them keeps
# the two studies' pairing and their CSVs identical.
from run_ga import (  # noqa: E402
    _cell,
    _jsonable,
    _positive_int,
    _stats,
    run_seed_for,
    write_csv,
)

#: The grid this study asks for: 5 crossover rates x 6 mutation rates.
DEFAULT_CROSSOVER_RATES: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
DEFAULT_MUTATION_RATES: Tuple[float, ...] = (0.3, 0.2, 0.1, 0.05, 0.01, 0.001)

#: Repetitions per (combination, instance).
DEFAULT_RUNS: int = 20

#: Its own directory; the main study's results are never written to from here.
DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results_ablation"
)

STAGES: Tuple[str, ...] = ("run", "aggregate")

#: Traces kept under ``--traces compact``: what a convergence plot needs.
COMPACT_TRACES: Tuple[str, ...] = (
    "best_so_far", "gen_best", "gen_avg", "evaluations", "elapsed", "feasible_fraction",
)

#: Measured cost of one evaluation of the cycle-time objective, in seconds of
#: CPU.  Used only for the estimate printed before a run starts.
CPU_SECONDS_PER_EVALUATION: float = 3.6e-4

#: Scalar columns of ``runs.csv``, in order: the main study's run fields with
#: the cell's identity in front.
RUN_FIELDS: Tuple[str, ...] = (
    "combo",
    "crossover_rate",
    "mutation_rate",
    "instance",
    "instance_index",
    "run",
    "run_seed",
    "cycle_time",
    "feasible",
    "branch",
    "min_sigma",
    "generations",
    "evaluations",
    "execution_time",
    "cpu_time",
    "seeded_fraction",
)


# ----------------------------------------------------------------------
# the grid
# ----------------------------------------------------------------------


def _rate(value: float) -> str:
    """Compact, round-trippable rate text: 0.1 -> '0.1', 0.001 -> '0.001'."""
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
    budget, elitism, tournament size, mutation width -- so two cells differ in
    the crossover and mutation rates and in nothing else.
    """
    configs: Dict[str, GAConfig] = {}
    for crossover in crossover_rates:
        for mutation in mutation_rates:
            configs[combo_key(crossover, mutation)] = replace(
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

    Picklable, and it carries its own configuration, so a worker needs nothing
    beyond the job itself.
    """

    combo: str
    config: GAConfig
    split: str
    instance_index: int
    run: int
    sigma_min_threshold: float
    dwell: float
    output_dir: str
    traces: str = "compact"
    check_path: bool = True
    retract: float = 0.06
    check_arm: bool = True
    arm_clearance: float = 0.005


def run_json_path(output_dir: str, combo: str, instance_name: str, run: int) -> str:
    return os.path.join(output_dir, "raw", combo, f"{instance_name}_run{run}.json")


def _trace_list(values, decimals: int = 6) -> List[Optional[float]]:
    return [_jsonable(round(float(v), decimals)) for v in np.asarray(values).ravel()]


def execute(job: AblationJob) -> Dict:
    """Run one job and write its JSON.  Executed in a worker process."""
    instance = build_instances(job.split)[job.instance_index]
    seed = run_seed_for(job.instance_index, job.run)

    evaluator = CycleTimeEvaluator(
        instance, sigma_min_threshold=job.sigma_min_threshold, dwell=job.dwell,
        check_path=job.check_path, check_arm=job.check_arm,
        arm_clearance=job.arm_clearance, retract=job.retract,
    )
    # The seed depends on (instance, run) alone, so every cell of the grid
    # starts this run from the same initial population.
    population, seeded = make_initial_population(
        instance, job.config.population_size, seed=seed
    )
    ga = ConfigurableGA(instance, evaluator=evaluator, config=job.config, seed=seed)
    result = ga.optimize(initial_population=population, seeded_fraction=seeded)

    outcome = evaluator.evaluate(evaluator.decode(result.best_chromosome))
    record: Dict = {
        "combo": job.combo,
        "crossover_rate": job.config.crossover_rate,
        "mutation_rate": job.config.mutation_rate,
        "instance": instance.name,
        "instance_index": job.instance_index,
        "run": job.run,
        "run_seed": seed,
        "cycle_time": _jsonable(result.best_fitness),
        "feasible": bool(result.feasible),
        "branch": outcome.branch_name,
        "min_sigma": _jsonable(
            float(outcome.sigma_min.min()) if outcome.sigma_min is not None else math.nan
        ),
        "generations": int(result.generations),
        "evaluations": int(result.total_evaluations),
        "execution_time": float(result.execution_time),
        "cpu_time": float(result.cpu_time),
        "seeded_fraction": float(result.seeded_fraction),
    }

    payload = dict(record)
    payload["split"] = job.split
    payload["chromosome"] = [float(g) for g in result.best_chromosome]
    payload["config"] = job.config.to_dict()
    payload["sigma_min_threshold"] = job.sigma_min_threshold
    payload["dwell"] = job.dwell
    payload["check_path"] = job.check_path
    payload["check_arm"] = job.check_arm
    payload["arm_clearance"] = job.arm_clearance
    payload["retract"] = job.retract
    if outcome.feasible and outcome.configurations is not None:
        payload["joint_solution_deg"] = np.degrees(outcome.configurations).round(4).tolist()
        payload["sigma_min_per_station"] = np.asarray(outcome.sigma_min).round(6).tolist()
    if job.traces != "none":
        traces = result.trace_dict()
        wanted = list(traces) if job.traces == "full" else list(COMPACT_TRACES)
        payload["traces"] = {
            name: _trace_list(traces[name]) for name in wanted if name in traces
        }

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
    print(f"\r  {done}/{total} runs  {elapsed / 60:.1f} min elapsed, "
          f"~{remaining / 60:.1f} min left   ", end="", flush=True)


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
    sigma_min_threshold: float,
    dwell: float,
    traces: str,
    resume: bool,
    check_path: bool,
    check_arm: bool,
    arm_clearance: float,
    retract: float,
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
                jobs.append(AblationJob(
                    combo=combo, config=config, split=split, instance_index=index,
                    run=run, sigma_min_threshold=sigma_min_threshold, dwell=dwell,
                    output_dir=output_dir, traces=traces, check_path=check_path,
                    check_arm=check_arm, arm_clearance=arm_clearance,
                    retract=retract,
                ))

    os.makedirs(output_dir, exist_ok=True)
    if done:
        print(f"Resuming: {len(done)} runs already on disk, {len(jobs)} left to do.")
    print(f"Running {len(configs)} combinations x {len(instance_indices)} '{split}' "
          f"instances x {runs} seeds = {len(jobs)} runs "
          f"({'sequential' if workers <= 1 else f'{workers} workers'})")

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
    """The ``metadata.json`` an earlier ``run`` stage left, if there is one."""
    path = os.path.join(output_dir, "metadata.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def collect_records(output_dir: str) -> List[Dict]:
    """Every run record under ``raw/``, for an aggregate-only invocation."""
    raw_root = os.path.join(output_dir, "raw")
    if not os.path.isdir(raw_root):
        raise SystemExit(f"no raw runs under {raw_root}. Run the 'run' stage first.")
    rows: List[Dict] = []
    for combo in sorted(os.listdir(raw_root)):
        combo_dir = os.path.join(raw_root, combo)
        if not os.path.isdir(combo_dir):
            continue
        for name in sorted(os.listdir(combo_dir)):
            if name.endswith(".json"):
                rows.append(load_record(os.path.join(combo_dir, name)))
    if not rows:
        raise SystemExit(f"no run JSON files under {raw_root}.")
    return rows


# ----------------------------------------------------------------------
# summarising
# ----------------------------------------------------------------------


def mean_ranks(rows: Sequence[Dict]) -> Dict[str, Optional[float]]:
    """Average rank of each combination across the paired (instance, seed) sets.

    Within one (instance, seed) every combination started from the same initial
    population, so the cells can be ranked directly: 1 is the best (lowest)
    cycle time, tied runs share the average rank of their block, and an
    infeasible run sorts last.  Averaging those ranks compares the cells without
    pooling cycle times across instances, whose scales differ.
    """
    groups: Dict[Tuple[int, int], List[Tuple[str, float]]] = {}
    for row in rows:
        value = row["cycle_time"]
        groups.setdefault((row["instance_index"], row["run"]), []).append(
            (row["combo"], float(value) if value is not None else float("inf"))
        )

    totals: Dict[str, List[float]] = {}
    for members in groups.values():
        order = sorted(members, key=lambda item: item[1])
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and order[j + 1][1] == order[i][1]:
                j += 1
            shared = (i + j) / 2.0 + 1.0  # average rank of the tied block
            for combo, _ in order[i:j + 1]:
                totals.setdefault(combo, []).append(shared)
            i = j + 1
    return {combo: float(np.mean(r)) if r else None for combo, r in totals.items()}


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
            entry["instance_index"] = first["instance_index"]
        entry.update({
            "n_runs": len(members),
            "n_feasible": len(feasible),
            "feasible_rate": len(feasible) / len(members),
        })
        for name, field in (("cycle_time", "cycle_time"), ("min_sigma", "min_sigma"),
                            ("evaluations", "evaluations"),
                            ("execution_time", "execution_time"), ("cpu_time", "cpu_time")):
            # Quality is only defined on feasible runs; effort and time are
            # measured on every run that happened.
            source = feasible if field in ("cycle_time", "min_sigma") else members
            for stat, value in _stats(r[field] for r in source).items():
                entry[f"{name}_{stat}"] = value
        if not by_instance:
            entry["mean_rank"] = ranks.get(first["combo"])
        table.append(entry)

    table.sort(key=lambda e: (-e["crossover_rate"], -e["mutation_rate"],
                              e.get("instance_index", 0)))
    return table


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
    sigma_threshold: float,
    dwell: float,
) -> None:
    """The readable report: the main table, then a grid per operator pair."""
    indexed = {entry["combo"]: entry for entry in summary}
    best = min((e for e in summary if e["cycle_time_mean"] is not None),
               key=lambda e: e["cycle_time_mean"], default=None)
    best_ranked = min((e for e in summary if e["mean_rank"] is not None),
                      key=lambda e: e["mean_rank"], default=None)

    lines: List[str] = []
    lines.append("# Crossover x mutation ablation -- static GA on robot cycle time\n")
    lines.append(
        f"Grid: {len(crossover_rates)} crossover rates x {len(mutation_rates)} mutation rates "
        f"= {len(crossover_rates) * len(mutation_rates)} combinations, {runs} runs on each of "
        f"{n_instances} '{split}' instances ({len(rows)} runs recorded).\n"
    )
    lines.append(
        f"Fixed operators: population {base.population_size}, {base.generations} generations "
        f"({base.evaluation_budget()} evaluations), elitism {base.elitism_rate:.0%}, tournament "
        f"k={base.tournament_k}, mutation width {base.mutation_sigma:g} of each gene's range. "
        f"**No local search, no learned control.**\n"
    )
    lines.append(
        f"Objective: the time for one full cycle (home -> every loading port in process order "
        f"-> home) with a {dwell:g} s dwell per station, under per-joint rate and acceleration "
        f"limits, subject to `sigma_min` >= {sigma_threshold:g} m/rad at every port.\n"
    )
    lines.append(
        "**Cycle time is minimised** (seconds, lower is better), and infeasible runs are "
        "excluded from the cycle-time statistics; `n_feasible` reports how many runs in each "
        "cell produced a layout at all. Evaluation counts and times cover every run.\n"
    )
    lines.append(
        "The crossover rate is the probability that a selected pair undergoes uniform "
        "crossover; the mutation rate is the probability that any one machine (x, y, rotation) "
        "receives Gaussian jitter and a re-orientation.\n"
    )
    lines.append(
        "`mean_rank` ranks the combinations against each other within every (instance, seed) "
        "pair -- all of which start from the same initial population -- and averages those "
        "ranks. It is the scale-free comparison; the pooled cycle-time columns mix instances "
        "whose scales differ.\n"
    )
    if best is not None:
        lines.append(
            f"Lowest mean cycle time: **crossover {_rate(best['crossover_rate'])}, mutation "
            f"{_rate(best['mutation_rate'])}** (mean {best['cycle_time_mean']:.3f} s, median "
            f"{_cell(best['cycle_time_median'], '.3f')} s, feasible "
            f"{best['n_feasible']}/{best['n_runs']}).\n"
        )
    if best_ranked is not None:
        lines.append(
            f"Best mean rank: **crossover {_rate(best_ranked['crossover_rate'])}, mutation "
            f"{_rate(best_ranked['mutation_rate'])}** ({best_ranked['mean_rank']:.2f} of "
            f"{len(summary)} cells).\n"
        )

    lines.append("\n## Per combination\n")
    lines.append(
        "| crossover | mutation | n | feasible | cycle mean (s) | sd | median | best | "
        "min sigma mean | evals mean | wall mean (s) | CPU mean (s) | mean rank |"
    )
    lines.append("|" + "---|" * 13)
    for entry in summary:
        lines.append("| " + " | ".join((
            _rate(entry["crossover_rate"]),
            _rate(entry["mutation_rate"]),
            str(entry["n_runs"]),
            f"{entry['n_feasible']}/{entry['n_runs']}",
            _cell(entry["cycle_time_mean"], ".3f"),
            _cell(entry["cycle_time_sd"], ".3f"),
            _cell(entry["cycle_time_median"], ".3f"),
            _cell(entry["cycle_time_min"], ".3f"),
            _cell(entry["min_sigma_mean"], ".4f"),
            _cell(entry["evaluations_mean"], ".0f"),
            _cell(entry["execution_time_mean"], ".1f"),
            _cell(entry["cpu_time_mean"], ".1f"),
            _cell(entry["mean_rank"], ".2f"),
        )) + " |")

    for title, field, spec in (
        ("Mean cycle time (s)", "cycle_time_mean", ".3f"),
        ("Median cycle time (s)", "cycle_time_median", ".3f"),
        ("Mean rank", "mean_rank", ".2f"),
        ("Feasible fraction", "feasible_rate", ".2f"),
    ):
        lines.append(f"\n## {title} -- crossover (rows) x mutation (columns)\n")
        lines.append("| crossover \\ mutation | "
                     + " | ".join(_rate(m) for m in mutation_rates) + " |")
        lines.append("|" + "---|" * (len(mutation_rates) + 1))
        for crossover in crossover_rates:
            cells = [_cell(indexed.get(combo_key(crossover, m), {}).get(field), spec)
                     for m in mutation_rates]
            lines.append(f"| **{_rate(crossover)}** | " + " | ".join(cells) + " |")

    lines.append(
        "\n---\n\nWall-clock time is inflated and noisy whenever the grid was run with "
        "`--workers > 1`, because parallel runs share the machine; CPU time and the evaluation "
        "count are the reliable effort measures in that case. The results themselves do not "
        "depend on the worker count: a run's seed is fixed by its (instance, run) pair alone.\n"
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
    sigma_threshold: float,
    dwell: float,
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
    write_report(os.path.join(output_dir, "summary.md"), summary, rows,
                 crossover_rates, mutation_rates, base, split, runs, n_instances,
                 sigma_threshold, dwell)

    print(f"Wrote {len(rows)} run records to runs.json and runs.csv")
    print(f"Wrote {len(summary)} combination summaries to summary.csv "
          f"(and summary_per_instance.csv)")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m arm_study.run_ablation",
        description="Full-factorial crossover x mutation ablation for the static GA "
                    "on robot cycle time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    grid = parser.add_argument_group("the grid")
    grid.add_argument("--crossover-rates", type=_probability("--crossover-rates"), nargs="+",
                      default=list(DEFAULT_CROSSOVER_RATES), help="crossover rates to sweep")
    grid.add_argument("--mutation-rates", type=_probability("--mutation-rates"), nargs="+",
                      default=list(DEFAULT_MUTATION_RATES), help="mutation rates to sweep")
    grid.add_argument("--runs", type=_positive_int("--runs"), default=DEFAULT_RUNS,
                      help="repetitions of each combination on each instance")
    grid.add_argument("--split", default="test", choices=list(ALL_SPLITS))
    grid.add_argument("--instances", type=int, nargs="+", default=None,
                      help="instance indices (default: the whole split)")

    operators = parser.add_argument_group("fixed GA operators")
    operators.add_argument("--population-size", type=_positive_int("--population-size"),
                           default=200)
    operators.add_argument("--generations", type=_positive_int("--generations"), default=300)
    operators.add_argument("--elitism-rate", type=_probability("--elitism-rate"), default=0.1)
    operators.add_argument("--tournament-k", type=_positive_int("--tournament-k"), default=3)
    operators.add_argument("--mutation-sigma", type=float, default=0.10,
                           help="Gaussian jitter width, as a fraction of each gene's range")
    operators.add_argument("--repair", action="store_true",
                           help="project an unplaceable child back onto the placeable "
                                "set after mutation. Held fixed across the grid, so the "
                                "sweep still varies the two rates and nothing else. Off "
                                "by default: the archived results_ablation/ runs were "
                                "produced without it, and the rates are being measured "
                                "on a different operator with it on.")
    operators.add_argument("--repair-margin", type=float, default=0.005,
                           help="metres of slack projected past each placement "
                                "constraint by --repair; ignored without it")

    model = parser.add_argument_group("robot and objective")
    model.add_argument("--sigma-min", type=float, default=0.14,
                       help="manipulability threshold, m/rad")
    model.add_argument("--dwell", type=float, default=0.30,
                       help="seconds held at each station")
    model.add_argument("--no-arm-check", action="store_true",
                       help="drop the arm/machine clearance criterion at the stations "
                            "(only for reproducing pre-clearance results)")
    model.add_argument("--arm-clearance", type=float, default=0.005,
                       help="metres the robot must keep from every machine it is not "
                            "reaching into (default 0.005)")
    model.add_argument("--retract", type=float, default=0.06,
                       help="metres above the tallest machine that the tool lifts to "
                            "between stations (default 0.06)")
    model.add_argument("--no-retract", action="store_true",
                       help="direct point-to-point moves; pair with --no-path-check")
    model.add_argument("--no-path-check", action="store_true",
                       help="only test clearance at the stations, not along the moves")

    execution = parser.add_argument_group("execution and output")
    execution.add_argument("--workers", type=_positive_int("--workers"), default=1,
                           help="parallel worker processes (wall-clock times get noisy above "
                                "1; the results themselves do not change)")
    execution.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    execution.add_argument("--traces", choices=("none", "compact", "full"), default="compact",
                           help="how much per-generation history each run's JSON keeps")
    execution.add_argument("--resume", action="store_true",
                           help="skip any (combination, instance, seed) already on disk")
    execution.add_argument("--only", nargs="+", choices=list(STAGES), default=None,
                           help="run a subset of the stages; 'aggregate' rebuilds the tables "
                                "from raw/")
    execution.add_argument("--dry-run", action="store_true",
                           help="print the plan and the cost estimate, then stop")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Caught here because otherwise it surfaces generations deep as an opaque
    # "Sample larger than population" from the tournament's random.sample.
    if args.tournament_k > args.population_size:
        raise SystemExit(
            f"--tournament-k ({args.tournament_k}) cannot exceed --population-size "
            f"({args.population_size}): selection draws that many distinct individuals."
        )

    instances = build_instances(args.split)
    indices = list(range(len(instances))) if args.instances is None else list(args.instances)
    bad = [i for i in indices if not 0 <= i < len(instances)]
    if bad:
        raise SystemExit(f"instance index/indices {bad} out of range for split "
                         f"'{args.split}' (0..{len(instances) - 1})")

    base = GAConfig(
        population_size=args.population_size,
        generations=args.generations,
        elitism_rate=args.elitism_rate,
        tournament_k=args.tournament_k,
        mutation_sigma=args.mutation_sigma,
        repair=args.repair,
        repair_margin=args.repair_margin,
    )
    configs = build_configs(base, args.crossover_rates, args.mutation_rates)
    stages = list(args.only or STAGES)

    total_runs = len(configs) * len(indices) * args.runs
    estimate = total_runs * base.evaluation_budget() * CPU_SECONDS_PER_EVALUATION / 3600.0

    print("Crossover x mutation ablation -- static GA on robot cycle time")
    print(f"Results directory : {args.results_dir}")
    print(f"Stages            : {', '.join(stages)}")
    print(f"Crossover rates   : {', '.join(_rate(r) for r in args.crossover_rates)}")
    print(f"Mutation rates    : {', '.join(_rate(r) for r in args.mutation_rates)}")
    print(f"Grid              : {len(configs)} combinations x {len(indices)} "
          f"'{args.split}' instances x {args.runs} seeds = {total_runs} runs")
    print(f"Fixed operators   : pop={base.population_size} gens={base.generations} "
          f"({base.evaluation_budget()} evals/run) elitism={base.elitism_rate:g} "
          f"k={base.tournament_k} (no local search)"
          + (f", placement repair on (margin {base.repair_margin:g} m)"
             if base.repair else ", no placement repair"))
    print(f"Objective         : cycle time, sigma_min >= {args.sigma_min:g} m/rad, "
          f"dwell {args.dwell:g} s"
          + ("" if args.no_path_check else ", clearance checked along every move")
          + (", direct moves (no retract)" if args.no_retract
             else f", retract {args.retract:g} m above the tallest machine")
          + (", ARM CLEARANCE OFF" if args.no_arm_check
             else f", arm clearance {args.arm_clearance:g} m"))
    print(f"Cost estimate     : ~{estimate:.1f} CPU-hours "
          f"(~{estimate / args.workers:.1f} h wall-clock on {args.workers} worker(s))")
    print()

    if args.dry_run:
        print("--dry-run: nothing was executed.")
        return

    os.makedirs(args.results_dir, exist_ok=True)
    started = time.perf_counter()

    rows: List[Dict] = []
    if "run" in stages:
        rows = run_grid(
            configs=configs, split=args.split, instance_indices=indices, runs=args.runs,
            workers=args.workers, output_dir=args.results_dir,
            sigma_min_threshold=args.sigma_min, dwell=args.dwell, traces=args.traces,
            resume=args.resume, check_path=not args.no_path_check,
            retract=None if args.no_retract else args.retract,
            check_arm=not args.no_arm_check, arm_clearance=args.arm_clearance,
        )
        metadata = {
            "crossover_rates": list(args.crossover_rates),
            "mutation_rates": list(args.mutation_rates),
            "combinations": sorted(configs),
            "split": args.split,
            "instance_indices": indices,
            "runs_per_instance": args.runs,
            "n_runs": len(rows),
            "traces": args.traces,
            "seed_rule": "run_seed = 42 + 100 * instance_index + run; "
                         "shared by every combination",
            "objective": "cycle time (s), minimised; infeasible = +inf -> null",
            "base_config": base.to_dict(),
            "configs": {key: config.to_dict() for key, config in configs.items()},
            "sigma_min_threshold": args.sigma_min,
            "dwell": args.dwell,
            "check_path": not args.no_path_check,
            "retract": None if args.no_retract else args.retract,
            "check_arm": not args.no_arm_check,
            "arm_clearance": args.arm_clearance,
            "robot": {
                "model": "4-DOF arm on Puma 560 link geometry "
                         "(waist, shoulder, elbow, wrist pitch; roll joints locked)",
                **{k: getattr(Puma560Arm(), k) for k in ("d1", "a2", "a3", "d3", "d4", "tool")},
                "L3": Puma560Arm().L3,
                "max_reach": Puma560Arm().max_reach,
                "min_reach": Puma560Arm().min_reach,
                # Recorded in the manufacturer's own convention as well as
                # this model's, so a reader can check the travel against a
                # Puma 560 datasheet without redoing the offset arithmetic.
                "joint_travel_deg": [list(np.degrees(ArmLimits().q_min)),
                                     list(np.degrees(ArmLimits().q_max))],
                "puma_joints": list(PUMA_JOINT_INDEX),
                "puma_travel_deg": [list(t) for t in PUMA_TRAVEL],
                "joint_velocity_limits": list(ArmLimits().velocity),
                "joint_acceleration_limits": list(ArmLimits().acceleration),
            },
            "instances": {instances[i].name: describe(instances[i]) for i in indices},
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "cpu_count": os.cpu_count(),
                "workers": args.workers,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }
        meta_path = os.path.join(args.results_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, default=str)
        print(f"Wrote configuration metadata to {meta_path}")

    if "aggregate" in stages:
        crossover_rates, mutation_rates = args.crossover_rates, args.mutation_rates
        split, runs, n_instances = args.split, args.runs, len(indices)
        sigma_min, dwell = args.sigma_min, args.dwell
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
                sigma_min = recorded["sigma_min_threshold"]
                dwell = recorded["dwell"]
            else:
                print("No metadata.json in the results directory: the report will describe "
                      "this invocation's settings, which may not be the ones the runs used.")
        write_outputs(
            output_dir=args.results_dir, rows=rows, crossover_rates=crossover_rates,
            mutation_rates=mutation_rates, base=base, split=split, runs=runs,
            n_instances=n_instances, sigma_threshold=sigma_min, dwell=dwell,
        )

    print(f"\nDone in {(time.perf_counter() - started) / 60:.1f} min.")


if __name__ == "__main__":
    main()
