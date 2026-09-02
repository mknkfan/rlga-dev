"""
Run the static GA on the cycle-time objective and write the study.

    python -m arm_study.run_ga                      # 10 instances x 10 seeds
    python -m arm_study.run_ga --instances 0 --runs 3 --generations 60
    python -m arm_study.run_ga --only aggregate figures
    python -m arm_study.run_ga --animate            # also write cycle.gif

Everything is written under ``--results-dir`` (default
``arm_study/results/``)::

    raw/<instance>_run<k>.json   one JSON per run: scalars, the winning
                                 chromosome, the joint solution and traces
    runs.csv / runs.json         every run, scalars only
    summary.csv                  per instance: mean/median/best cycle time
    summary.md                   the readable report
    metadata.json                robot, operators, instance features, environment
    figures/<instance>/          layout.png, trajectory.png, cycle.html
                                 (and cycle.gif with --animate), best run
    figures/convergence.png      pooled across every run

Cycle time is **minimised** and an infeasible run has ``+inf``.  JSON has no
literal for that, so such a run is written as ``cycle_time: null`` with
``feasible: false``; the statistics are taken over feasible runs and every
table carries the feasible count.

Seeding: run seed ``= 42 + 100 * instance_index + run``, the same rule the
distance study used, so a given (instance, run) pair always starts from the
same initial population.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

# Run as a script or as a module: either way arm_study/ must be importable by
# its own modules, which import each other by bare name so the package can be
# lifted somewhere else unchanged.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluator import CycleTimeEvaluator  # noqa: E402
from ga import ConfigurableGA, GAConfig, RunResult, make_initial_population  # noqa: E402
from problems import ALL_SPLITS, build_instances, describe  # noqa: E402
from robot import (  # noqa: E402
    PUMA_JOINT_INDEX, PUMA_TRAVEL, ArmLimits, Puma560Arm,
)

DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
STAGES = ("run", "aggregate", "figures")

RUN_FIELDS = (
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


def run_seed_for(instance_index: int, run: int) -> int:
    """Seed shared by everything evaluated on a given (instance, run) pair."""
    return 42 + instance_index * 100 + run


def _jsonable(value: float) -> Optional[float]:
    value = float(value)
    return value if math.isfinite(value) else None


@dataclass
class Job:
    """One (instance, seed) run.  Picklable, so a worker needs nothing else."""

    split: str
    instance_index: int
    run: int
    config: GAConfig
    sigma_min_threshold: float
    dwell: float
    output_dir: str
    store_traces: bool = True
    check_path: bool = True
    retract: float = 0.06
    check_arm: bool = True
    arm_clearance: float = 0.005


def execute(job: Job) -> Dict:
    """Run one GA and write its JSON.  Executed in a worker process."""
    instance = build_instances(job.split)[job.instance_index]
    seed = run_seed_for(job.instance_index, job.run)

    evaluator = CycleTimeEvaluator(
        instance, sigma_min_threshold=job.sigma_min_threshold, dwell=job.dwell,
        check_path=job.check_path, check_arm=job.check_arm,
        arm_clearance=job.arm_clearance, retract=job.retract,
    )
    population, seeded = make_initial_population(instance, job.config.population_size, seed=seed)
    ga = ConfigurableGA(instance, evaluator=evaluator, config=job.config, seed=seed)
    result = ga.optimize(initial_population=population, seeded_fraction=seeded)

    outcome = evaluator.evaluate(evaluator.decode(result.best_chromosome))
    record: Dict = {
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
        payload["segment_times"] = evaluator.segment_times(outcome).round(6).tolist()
    if job.store_traces:
        payload["traces"] = {
            name: [_jsonable(round(float(v), 6)) for v in values]
            for name, values in result.trace_dict().items()
        }

    path = os.path.join(job.output_dir, "raw", f"{instance.name}_run{job.run}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    return record


def _progress(done: int, total: int, started: float, every: int = 5) -> None:
    if done != total and done % every:
        return
    elapsed = time.perf_counter() - started
    rate = done / max(elapsed, 1e-9)
    remaining = (total - done) / rate if rate > 0 else float("nan")
    print(f"\r  {done}/{total} runs  {elapsed / 60:.1f} min elapsed, "
          f"~{remaining / 60:.1f} min left   ", end="", flush=True)


def run_grid(jobs: Sequence[Job], workers: int) -> List[Dict]:
    print(f"Running {len(jobs)} runs "
          f"({'sequential' if workers <= 1 else f'{workers} workers'})")
    started = time.perf_counter()
    rows: List[Dict] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for k, row in enumerate(pool.map(execute, jobs, chunksize=1), start=1):
                rows.append(row)
                _progress(k, len(jobs), started)
    else:
        for k, job in enumerate(jobs, start=1):
            rows.append(execute(job))
            _progress(k, len(jobs), started)
    print()
    print(f"Finished in {(time.perf_counter() - started) / 60:.1f} min")
    return rows


def collect_records(output_dir: str) -> List[Dict]:
    raw = os.path.join(output_dir, "raw")
    if not os.path.isdir(raw):
        raise SystemExit(f"no raw runs under {raw}. Run the 'run' stage first.")
    rows: List[Dict] = []
    for name in sorted(os.listdir(raw)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(raw, name), encoding="utf-8") as fh:
            payload = json.load(fh)
        rows.append({field: payload[field] for field in RUN_FIELDS})
    if not rows:
        raise SystemExit(f"no run JSON files under {raw}.")
    return rows


# ----------------------------------------------------------------------
# summarising
# ----------------------------------------------------------------------


def _stats(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
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


def summarise(rows: Sequence[Dict]) -> List[Dict]:
    groups: Dict[str, List[Dict]] = {}
    for row in rows:
        groups.setdefault(row["instance"], []).append(row)

    table: List[Dict] = []
    for name, members in groups.items():
        feasible = [r for r in members if r["feasible"]]
        entry: Dict = {
            "instance": name,
            "instance_index": members[0]["instance_index"],
            "n_runs": len(members),
            "n_feasible": len(feasible),
        }
        for label, field in (("cycle_time", "cycle_time"), ("min_sigma", "min_sigma"),
                             ("cpu_time", "cpu_time")):
            source = feasible if field != "cpu_time" else members
            for stat, value in _stats(r[field] for r in source).items():
                entry[f"{label}_{stat}"] = value
        entry["evaluations_mean"] = float(np.mean([r["evaluations"] for r in members]))
        table.append(entry)
    table.sort(key=lambda e: e["instance_index"])
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


def write_report(path: str, summary: Sequence[Dict], rows: Sequence[Dict],
                 config: GAConfig, sigma_threshold: float, dwell: float,
                 split: str, runs: int) -> None:
    arm = Puma560Arm()
    limits = ArmLimits()
    feasible = [r for r in rows if r["feasible"]]

    lines: List[str] = []
    lines.append("# Static GA on robot cycle time -- Puma 560 workcell\n")
    lines.append(
        f"{len(summary)} '{split}' instances x {runs} seeds = {len(rows)} runs. "
        f"**Cycle time is minimised** (seconds); an infeasible layout scores `+inf` and is "
        f"excluded from the statistics, with `n_feasible` recording how many runs of each "
        f"instance produced a layout at all.\n"
    )
    lines.append("## What is being optimised\n")
    lines.append(
        "The objective is the time for one full cycle: home -> every loading port in process "
        f"order -> home, under a trapezoidal velocity profile per move, plus a {dwell:g} s dwell "
        "at each station. Each move is time-scaled so that no joint exceeds its rate or "
        "acceleration limit, and the move takes as long as the slowest joint needs -- so the "
        "cost of a layout depends on which joints a move loads, not merely on how far the "
        "tool travels.\n"
    )
    lines.append("A layout is feasible only if all five hold:\n")
    lines.append(
        "1. footprints inside the cell, clear of the robot column, and mutually disjoint;\n"
        "2. every port reachable with a vertical top-down approach;\n"
        "3. every joint inside its travel at every port;\n"
        f"4. `sigma_min` of the position Jacobian >= {sigma_threshold:g} m/rad at every port, "
        "which is the anti-singularity condition;\n"
        "5. at every station, no part of the robot within the clearance margin of any "
        "machine -- except the wrist and gripper against the machine the tool is "
        "descending onto, which is the approach corridor;\n\n"
        "and a **single** arm posture must satisfy 2-5 at every station, since a real cell "
        "does not reconfigure the arm mid-cycle. Criterion 5 governs the stations only: "
        "between them the moves interpolate in joint space and can still sweep the arm "
        "through a machine, which `--check-path` rejects and the README discusses.\n"
    )
    lines.append("## Robot\n")
    lines.append(
        f"4-DOF arm on Puma 560 link geometry: shoulder {arm.d1:g} m up, upper arm "
        f"{arm.a2:g} m, forearm {arm.L3:.4f} m (= hypot({arm.a3:g}, {arm.d4:g})), shoulder "
        f"offset {arm.d3:g} m, tool {arm.tool:g} m. Reach {arm.min_reach:.3f}-"
        f"{arm.max_reach:.3f} m from the shoulder. Joint rate limits "
        f"{', '.join(f'{v:g}' for v in limits.velocity)} rad/s; accelerations "
        f"{', '.join(f'{a:g}' for a in limits.acceleration)} rad/s^2.\n"
    )
    travel = ", ".join(
        f"q{k+1} {math.degrees(lo):.1f} to {math.degrees(hi):.1f} deg "
        f"(joint {PUMA_JOINT_INDEX[k] + 1}, {PUMA_TRAVEL[PUMA_JOINT_INDEX[k]][0]:.0f} to "
        f"{PUMA_TRAVEL[PUMA_JOINT_INDEX[k]][1]:.0f})"
        for k, (lo, hi) in enumerate(zip(limits.q_min, limits.q_max))
    )
    lines.append(
        "Joint travel is the manufacturer's, mapped into this model's angle convention by a "
        f"fixed per-joint offset and otherwise unaltered: {travel}. The two roll joints, 4 "
        "and 6, are locked at zero because a top-down pick does not need them.\n"
    )
    lines.append("## GA\n")
    lines.append(
        f"Population {config.population_size}, {config.generations} generations "
        f"({config.evaluation_budget()} evaluations), tournament k={config.tournament_k}, "
        f"uniform crossover p={config.crossover_rate:g}, mutation p={config.mutation_rate:g} "
        f"per machine, elitism {config.elitism_rate:.0%}. **No local search, no learned "
        "control.**\n"
    )
    if feasible:
        times = np.array([r["cycle_time"] for r in feasible], dtype=float)
        best = min(feasible, key=lambda r: r["cycle_time"])
        lines.append(
            f"Across every feasible run: cycle time {times.mean():.3f} +- {times.std(ddof=1):.3f} s, "
            f"best {times.min():.3f} s (`{best['instance']}` run {best['run']}, posture "
            f"{best['branch']}).\n"
        )

    lines.append("\n## Per instance\n")
    lines.append("| instance | feasible | cycle mean (s) | sd | median | best | "
                 "min sigma mean | evals | CPU mean (s) |")
    lines.append("|" + "---|" * 9)
    for entry in summary:
        lines.append(
            "| " + " | ".join((
                entry["instance"],
                f"{entry['n_feasible']}/{entry['n_runs']}",
                _cell(entry["cycle_time_mean"], ".3f"),
                _cell(entry["cycle_time_sd"], ".3f"),
                _cell(entry["cycle_time_median"], ".3f"),
                _cell(entry["cycle_time_min"], ".3f"),
                _cell(entry["min_sigma_mean"], ".4f"),
                _cell(entry["evaluations_mean"], ".0f"),
                _cell(entry["cpu_time_mean"], ".1f"),
            )) + " |"
        )

    postures: Dict[str, int] = {}
    for row in feasible:
        postures[row["branch"]] = postures.get(row["branch"], 0) + 1
    if postures:
        lines.append("\n## Posture chosen by the best layout\n")
        lines.append("| posture | runs |")
        lines.append("|---|---|")
        for name, count in sorted(postures.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {name} | {count} |")

    lines.append(
        "\n---\n\nWall-clock time is inflated when the study is run with `--workers > 1`; CPU "
        "time and the evaluation count are the reliable effort measures in that case.\n"
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------


def make_figures(output_dir: str, rows: Sequence[Dict], split: str,
                 sigma_threshold: float, dwell: float, animate: bool,
                 check_path: bool = True, spin: float = 360.0,
                 no_viewer: bool = False, check_arm: bool = True,
                 arm_clearance: float = 0.005, retract: float = 0.06) -> None:
    import visualize
    from ga import RunResult as _RunResult  # noqa: F401

    figures_dir = os.path.join(output_dir, "figures")
    instances = {inst.name: inst for inst in build_instances(split)}

    # The best run of each instance gets the layout and trajectory figures.
    best_by_instance: Dict[str, Dict] = {}
    for row in rows:
        if not row["feasible"]:
            continue
        current = best_by_instance.get(row["instance"])
        if current is None or row["cycle_time"] < current["cycle_time"]:
            best_by_instance[row["instance"]] = row

    for name, row in sorted(best_by_instance.items(),
                            key=lambda kv: kv[1]["instance_index"]):
        instance = instances[name]
        path = os.path.join(output_dir, "raw", f"{name}_run{row['run']}.json")
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        evaluator = CycleTimeEvaluator(
            instance, sigma_min_threshold=sigma_threshold, dwell=dwell,
            check_path=check_path, check_arm=check_arm,
            arm_clearance=arm_clearance, retract=retract,
        )
        outcome = evaluator.evaluate(
            evaluator.decode(np.asarray(payload["chromosome"], dtype=float))
        )
        target = os.path.join(figures_dir, name)
        visualize.figure_layout(instance, outcome, evaluator,
                                os.path.join(target, "layout.png"))
        visualize.figure_trajectory(instance, outcome, evaluator,
                                    os.path.join(target, "trajectory.png"))
        if not no_viewer:
            visualize.export_viewer(instance, outcome, evaluator,
                                    os.path.join(target, "cycle.html"))
        if animate:
            visualize.animate_cycle(instance, outcome, evaluator,
                                    os.path.join(target, "cycle.gif"), spin=spin)

    # Convergence needs the traces, so it reads the raw JSON back.
    results: List[RunResult] = []
    raw_dir = os.path.join(output_dir, "raw")
    for name in sorted(os.listdir(raw_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(raw_dir, name), encoding="utf-8") as fh:
            payload = json.load(fh)
        traces = payload.get("traces")
        if not traces:
            continue

        def column(key: str) -> np.ndarray:
            return np.array(
                [np.nan if v is None else float(v) for v in traces[key]], dtype=float
            )

        results.append(
            RunResult(
                label="GA (no LS)", instance_name=payload["instance"],
                seed=payload["run_seed"],
                best_fitness=payload["cycle_time"] if payload["cycle_time"] else math.inf,
                best_chromosome=np.asarray(payload["chromosome"], dtype=float),
                feasible=payload["feasible"], generations=payload["generations"],
                total_evaluations=payload["evaluations"],
                execution_time=payload["execution_time"], cpu_time=payload["cpu_time"],
                seeded_fraction=payload["seeded_fraction"],
                best_so_far=column("best_so_far"), gen_best=column("gen_best"),
                gen_avg=column("gen_avg"), diversity=column("diversity"),
                evaluations=column("evaluations"), elapsed=column("elapsed"),
                feasible_fraction=column("feasible_fraction"),
            )
        )
    visualize.figure_convergence(
        results, os.path.join(figures_dir, "convergence.png"),
        title=f"{len(results)} runs, {split} split",
    )


# ----------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------


def _positive_int(name: str):
    def parse(text: str) -> int:
        value = int(text)
        if value < 1:
            raise argparse.ArgumentTypeError(f"{name} must be >= 1, got {value}")
        return value

    return parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m arm_study.run_ga",
        description="Static GA minimising robot cycle time in a Puma 560 workcell.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    grid = parser.add_argument_group("what to run")
    grid.add_argument("--split", default="test", choices=list(ALL_SPLITS))
    grid.add_argument("--instances", type=int, nargs="+", default=None,
                      help="instance indices (default: the whole split)")
    grid.add_argument("--runs", type=_positive_int("--runs"), default=10,
                      help="independent seeds per instance")

    ga = parser.add_argument_group("GA operators (static study defaults)")
    ga.add_argument("--population-size", type=_positive_int("--population-size"), default=200)
    ga.add_argument("--generations", type=_positive_int("--generations"), default=300)
    ga.add_argument("--crossover-rate", type=float, default=0.9)
    ga.add_argument("--mutation-rate", type=float, default=0.1)
    ga.add_argument("--elitism-rate", type=float, default=0.1)
    ga.add_argument("--tournament-k", type=_positive_int("--tournament-k"), default=3)

    model = parser.add_argument_group("robot and objective")
    model.add_argument("--sigma-min", type=float, default=0.14,
                       help="manipulability threshold, m/rad")
    model.add_argument("--dwell", type=float, default=0.30,
                       help="seconds held at each station")
    model.add_argument("--no-path-check", action="store_true",
                       help="only test clearance at the stations, not along the moves. "
                            "With the retract in place the along-the-move test costs almost "
                            "nothing, so this is for comparison rather than for speed.")
    model.add_argument("--retract", type=float, default=0.06,
                       help="metres above the tallest machine that the tool lifts to "
                            "between stations (default 0.06). The cycle becomes descend / "
                            "dwell / retract / traverse, which is what keeps the arm out of "
                            "the machines between ports.")
    model.add_argument("--no-retract", action="store_true",
                       help="direct point-to-point moves, as in the earlier form of this "
                            "study. Note that whole-cycle clearance is then unsatisfiable, "
                            "so pair it with --no-path-check.")
    model.add_argument("--no-arm-check", action="store_true",
                       help="drop the arm/machine clearance criterion at the stations. "
                            "Only for reproducing results from before it existed: with it "
                            "off, converged layouts put the forearm inside a machine.")
    model.add_argument("--arm-clearance", type=float, default=0.005,
                       help="metres the robot must keep from every machine it is not "
                            "reaching into (default 0.005)")

    execution = parser.add_argument_group("execution and output")
    execution.add_argument("--workers", type=_positive_int("--workers"), default=1)
    execution.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    execution.add_argument("--only", nargs="+", choices=list(STAGES), default=None)
    execution.add_argument("--no-traces", action="store_true",
                           help="omit per-generation history from the run JSON")
    execution.add_argument("--animate", action="store_true",
                           help="also write an orbiting cycle.gif per instance")
    execution.add_argument("--spin", type=float, default=360.0,
                           help="degrees the GIF camera orbits over the animation; "
                                "0 keeps the old fixed viewpoint")
    execution.add_argument("--no-viewer", action="store_true",
                           help="skip the drag-to-rotate cycle.html viewer")
    execution.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    instances = build_instances(args.split)
    indices = list(range(len(instances))) if args.instances is None else list(args.instances)
    bad = [i for i in indices if not 0 <= i < len(instances)]
    if bad:
        raise SystemExit(f"instance index/indices {bad} out of range "
                         f"(0..{len(instances) - 1})")

    config = GAConfig(
        population_size=args.population_size,
        generations=args.generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        elitism_rate=args.elitism_rate,
        tournament_k=args.tournament_k,
    )
    stages = list(args.only or STAGES)
    total = len(indices) * args.runs

    print("Static GA on robot cycle time (Puma 560 workcell)")
    print(f"Results directory : {args.results_dir}")
    print(f"Stages            : {', '.join(stages)}")
    print(f"Design            : {len(indices)} '{args.split}' instances x "
          f"{args.runs} seeds = {total} runs")
    print(f"GA                : pop={config.population_size} gens={config.generations} "
          f"cx={config.crossover_rate:g} mut={config.mutation_rate:g} "
          f"elitism={config.elitism_rate:g} k={config.tournament_k} (no local search)")
    print(f"Objective         : cycle time, sigma_min >= {args.sigma_min:g} m/rad, "
          f"dwell {args.dwell:g} s"
          + ("" if args.no_path_check else ", clearance checked along every move")
          + (", direct moves (no retract)" if args.no_retract
             else f", retract {args.retract:g} m above the tallest machine")
          + (", ARM CLEARANCE OFF" if args.no_arm_check
             else f", arm clearance {args.arm_clearance:g} m"))
    print()
    if args.dry_run:
        print("--dry-run: nothing was executed.")
        return

    os.makedirs(args.results_dir, exist_ok=True)
    started = time.perf_counter()
    rows: List[Dict] = []

    if "run" in stages:
        jobs = [
            Job(split=args.split, instance_index=i, run=r, config=config,
                sigma_min_threshold=args.sigma_min, dwell=args.dwell,
                output_dir=args.results_dir, store_traces=not args.no_traces,
                check_path=not args.no_path_check, check_arm=not args.no_arm_check,
                arm_clearance=args.arm_clearance,
                retract=None if args.no_retract else args.retract)
            for i in indices for r in range(args.runs)
        ]
        rows = run_grid(jobs, args.workers)

        metadata = {
            "split": args.split,
            "instance_indices": indices,
            "runs_per_instance": args.runs,
            "n_runs": len(rows),
            "seed_rule": "run_seed = 42 + 100 * instance_index + run",
            "objective": "cycle time (s), minimised; infeasible = +inf -> null",
            "ga_config": config.to_dict(),
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
            "instances": {
                instances[i].name: describe(instances[i]) for i in indices
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "cpu_count": os.cpu_count(),
                "workers": args.workers,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }
        with open(os.path.join(args.results_dir, "metadata.json"), "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, default=str)
        print(f"Wrote {os.path.join(args.results_dir, 'metadata.json')}")

    if "aggregate" in stages or "figures" in stages:
        if not rows:
            rows = collect_records(args.results_dir)

    if "aggregate" in stages:
        rows = sorted(rows, key=lambda r: (r["instance_index"], r["run"]))
        summary = summarise(rows)
        with open(os.path.join(args.results_dir, "runs.json"), "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
        write_csv(os.path.join(args.results_dir, "runs.csv"), rows, RUN_FIELDS)
        write_csv(os.path.join(args.results_dir, "summary.csv"), summary)
        write_report(os.path.join(args.results_dir, "summary.md"), summary, rows,
                     config, args.sigma_min, args.dwell, args.split, args.runs)
        print(f"Wrote runs.csv, summary.csv and summary.md to {args.results_dir}")

    if "figures" in stages:
        print("Figures:")
        make_figures(args.results_dir, rows, args.split, args.sigma_min,
                     args.dwell, args.animate, not args.no_path_check, args.spin,
                     args.no_viewer, not args.no_arm_check, args.arm_clearance,
                     None if args.no_retract else args.retract)

    print(f"\nDone in {(time.perf_counter() - started) / 60:.1f} min.")


if __name__ == "__main__":
    main()
