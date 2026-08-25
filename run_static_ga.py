"""
Static-GA-only pipeline: no RL, no agent training.

This is the sibling of :mod:`run_all` with the reinforcement-learning half
removed.  It runs only the stages a static GA needs --

1. ``instances``  - write the instance catalogue with per-instance features.
2. ``main``       - the grid of selected static variants on the chosen split.
3. ``timing``     - a sequential replication (one run at a time) so the
                    wall-clock numbers are not distorted by parallel load.
4. ``analyze``    - statistics and report.
5. ``figures``    - all plots.

-- and, unlike ``run_all``, it lets you set the GA operators from the command
line: tournament size, elitism rate, crossover rate, mutation rate, local
search rate and local-search iterations.  The ``calibrate`` and ``train``
stages do not exist here; nothing loads a Q-table.

Variants
--------
The catalogue is the static half of :func:`optimization.variants.main_variants`:
0% / 10% / 50% / 100% local search, on offspring and on elites, plus the
deterministic (generation-scheduled) GA on each target.  ``custom`` builds a
variant at whatever ``--ls-rate`` / ``--ls-target`` you ask for.  List them
with ``--list-variants``.

Stacking runs
-------------
``--append`` merges a run into an existing ``runs.csv`` instead of replacing
it, so several invocations with different operator settings accumulate into
one analysis.  Give each invocation a ``--tag`` to keep its variants distinct
from the ones already in the file (without a tag, a repeat of the same variant
replaces its earlier rows).

Examples
--------
Everything, default operators::

    python -m run_static_ga

Two variants only, 5 seeds, bigger tournament::

    python -m run_static_ga --variants ls10-elites ls100-elites --runs 5 --tournament-k 5

Stack a second operator setting onto the first, then re-analyse both::

    python -m run_static_ga --variants elites --tag k3 --only main
    python -m run_static_ga --variants elites --tag k7 --tournament-k 7 --only main --append
    python -m run_static_ga --only analyze figures

A custom local-search rate::

    python -m run_static_ga --variants custom --ls-rate 0.25 --ls-target elites
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.problems import build_instances, split_size
from optimization.ga_core import (
    DEFAULT_LS_SCHEDULE,
    ConfigurableGA,
    GAConfig,
    make_initial_population,
)
from optimization.run_experiments import (
    _progress,
    environment_info,
    read_rows,
    run_seed_for,
    write_rows,
)
from paths import DEFAULT_RESULTS_DIR

#: Stages this entry point knows about, in execution order.
STAGES: Tuple[str, ...] = ("instances", "main", "timing", "analyze", "figures")

#: Default output location.  Deliberately *not* ``results/``: that directory
#: holds the archived RL-GA study, and a static-only ``runs.csv`` written over
#: it would silently invalidate the committed analysis and figures.
DEFAULT_STATIC_RESULTS_DIR = os.path.join(DEFAULT_RESULTS_DIR, "static_ga")

#: Where a variant may apply local search.
LS_TARGETS: Tuple[str, ...] = ("offspring", "elites")


# ----------------------------------------------------------------------
# the static variant catalogue
# ----------------------------------------------------------------------


def _label(name: str, target: str) -> str:
    suffix = {"offspring": "offspring", "elites": "elites", "none": "no LS"}[target]
    return f"{name} [LS on {suffix}]"


def _catalogue() -> Dict[str, Tuple[str, str, str, float, str]]:
    """``short name -> (variant key, ls_target, ls_controller, ls_rate, label)``.

    The keys match :func:`optimization.variants.main_variants` exactly, so a
    static-only ``runs.csv`` stays comparable with the full study's rows and
    the analysis/figure code recognises the same naming conventions.
    """
    entries: Dict[str, Tuple[str, str, str, float, str]] = {
        "no-ls": ("ga_ls00", "none", "static", 0.0, "GA no LS"),
    }
    for target in LS_TARGETS:
        for rate, tag in ((0.1, "ls10"), (0.5, "ls50"), (1.0, "ls100")):
            entries[f"{tag}-{target}"] = (
                f"ga_{tag}_{target}",
                target,
                "static",
                rate,
                _label(f"GA {int(rate * 100)}% LS", target),
            )
        entries[f"deterministic-{target}"] = (
            f"ga_sched_{target}",
            target,
            "schedule",
            0.1,
            _label("GA scheduled LS", target),
        )
    return entries


CATALOGUE = _catalogue()

#: Group aliases accepted by ``--variants`` alongside individual names.
GROUPS: Dict[str, Tuple[str, ...]] = {
    "all": tuple(CATALOGUE),
    "offspring": tuple(k for k in CATALOGUE if k.endswith("-offspring")),
    "elites": tuple(k for k in CATALOGUE if k.endswith("-elites")),
    "fixed": tuple(k for k in CATALOGUE if not k.startswith("deterministic")),
    "deterministic": tuple(k for k in CATALOGUE if k.startswith("deterministic")),
}


def print_catalogue() -> None:
    print("Static GA variants.  --variants accepts these names, the group aliases")
    print("below, or the full variant keys in the middle column.\n")
    width = max(len(name) for name in CATALOGUE)
    for name, (key, target, controller, rate, _) in CATALOGUE.items():
        rate_text = "scheduled" if controller == "schedule" else f"{rate:.0%}"
        print(f"  {name:<{width}}  {key:<20}  LS {rate_text} on {target}")
    print(f"  {'custom':<{width}}  {'ga_lsNN_<target>':<20}  LS --ls-rate on --ls-target")
    print("\nGroups:")
    for group, members in GROUPS.items():
        print(f"  {group:<{width}}  {', '.join(members)}")


def _tagged(key: str, tag: Optional[str]) -> str:
    """Insert ``tag`` before the trailing local-search target, if any.

    ``ga_ls10_elites`` + ``k7`` -> ``ga_ls10_k7_elites``.  Keeping the target
    last matters: the figure code groups variants by ``key.endswith(target)``.
    """
    if not tag:
        return key
    for target in LS_TARGETS:
        if key.endswith(f"_{target}"):
            return f"{key[: -len(target) - 1]}_{tag}_{target}"
    return f"{key}_{tag}"


def build_configs(
    names: Sequence[str],
    base: GAConfig,
    ls_rate: Optional[float],
    ls_target: str,
    tag: Optional[str],
) -> Dict[str, GAConfig]:
    """Resolve ``--variants`` names into ``{variant key: GAConfig}``.

    ``base`` carries the operator settings shared by every variant; only the
    local-search fields and the label differ between them.
    """
    selected: List[str] = []
    for name in names:
        members = GROUPS.get(name)
        if members is not None:
            candidates = list(members)
        elif name == "custom" or name in CATALOGUE:
            candidates = [name]
        else:
            # Also accept the full variant key, e.g. "ga_ls50_elites".
            candidates = [short for short, entry in CATALOGUE.items() if entry[0] == name]
            if not candidates:
                raise SystemExit(
                    f"unknown variant {name!r}. Run with --list-variants to see the catalogue."
                )
        for candidate in candidates:
            if candidate not in selected:
                selected.append(candidate)

    if ls_rate is not None and "custom" not in selected:
        raise SystemExit(
            "--ls-rate only applies to the 'custom' variant; the catalogue variants "
            "carry their own rate. Add 'custom' to --variants, or drop --ls-rate."
        )
    if "custom" in selected and ls_rate is None:
        raise SystemExit("--variants custom needs --ls-rate, e.g. --ls-rate 0.25")
    custom_rate = float(ls_rate) if ls_rate is not None else 0.0

    configs: Dict[str, GAConfig] = {}
    for name in selected:
        if name == "custom":
            # A rate of 0 is "no local search" whatever target was named, and
            # is exactly the catalogue's ga_ls00 - so it gets that key rather
            # than a second name for the same variant.
            target = "none" if custom_rate == 0.0 else ls_target
            stem = (
                "ga_ls00"
                if target == "none"
                else f"ga_ls{int(round(custom_rate * 100)):02d}_{target}"
            )
            key = _tagged(stem, tag)
            label = _label(f"GA {custom_rate:.0%} LS", target)
            controller, rate = "static", custom_rate
        else:
            key, target, controller, rate, label = CATALOGUE[name]
            key = _tagged(key, tag)
        configs[key] = replace(
            base,
            label=label + (f" ({tag})" if tag else ""),
            ls_target=target,
            ls_controller=controller,
            ls_rate=rate,
        )
    return configs


# ----------------------------------------------------------------------
# running the grid
# ----------------------------------------------------------------------


@dataclass
class StaticJob:
    """One (variant, instance, seed) combination.

    Picklable, and it carries the configuration itself: a worker therefore
    needs no variant registry and no agent, which is what lets the operator
    settings come straight from the command line.
    """

    variant_key: str
    config: GAConfig
    split: str
    instance_index: int
    run: int
    output_dir: str
    store_trace: bool = True


def execute(job: StaticJob) -> Dict:
    """Run one job.  Executed in a worker process."""
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
    # Every variant on a given (instance, seed) pair starts from the same
    # initial population, which is what makes the paired statistics valid.
    population = make_initial_population(
        instance.machines, instance.workspace_bounds, job.config.population_size, seed=seed
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
            q_table=np.zeros(0),
        )

    return {
        "variant": job.variant_key,
        "label": job.config.label,
        "split": job.split,
        "instance": instance.name,
        "instance_index": job.instance_index,
        "seed": job.run,
        "run_seed": seed,
        "agent_seed": -1,  # no agent; the column exists only for CSV compatibility
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


def merge_rows(existing: Sequence[Dict], fresh: Sequence[Dict]) -> List[Dict]:
    """Stack ``fresh`` onto ``existing``.

    A repeated (variant, instance, seed) keeps the fresh row, so re-running a
    variant updates it in place instead of duplicating it.
    """
    merged = {(r["variant"], r["instance"], r["seed"]): r for r in existing}
    merged.update({(r["variant"], r["instance"], r["seed"]): r for r in fresh})
    return [merged[key] for key in sorted(merged)]


def run_static_grid(
    configs: Dict[str, GAConfig],
    split: str,
    runs: int,
    workers: int,
    output_dir: str,
    instance_indices: Optional[Sequence[int]] = None,
    csv_name: str = "runs.csv",
    store_trace: bool = True,
    append: bool = False,
) -> List[Dict]:
    """Run every (variant, instance, seed) combination and write the results."""
    instances = build_instances(split)
    indices = list(instance_indices) if instance_indices is not None else list(range(len(instances)))
    out_of_range = [i for i in indices if not 0 <= i < len(instances)]
    if out_of_range:
        raise SystemExit(
            f"--instances {out_of_range} out of range: split {split!r} has "
            f"{len(instances)} instances (0..{len(instances) - 1})"
        )

    jobs = [
        StaticJob(
            variant_key=key,
            config=config,
            split=split,
            instance_index=index,
            run=run,
            output_dir=output_dir,
            store_trace=store_trace,
        )
        for index in indices
        for run in range(runs)
        for key, config in configs.items()
    ]

    os.makedirs(output_dir, exist_ok=True)
    print(
        f"Running {len(configs)} variants x {len(indices)} '{split}' instances x "
        f"{runs} seeds = {len(jobs)} runs "
        f"({'sequential' if workers <= 1 else f'{workers} workers'})"
    )

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

    csv_path = os.path.join(output_dir, csv_name)
    written: Sequence[Dict] = rows
    if append and os.path.exists(csv_path):
        previous = read_rows(csv_path)
        written = merge_rows(previous, rows)
        print(
            f"Appending to {csv_path}: {len(previous)} existing + {len(rows)} new "
            f"= {len(written)} records"
        )
    write_rows(csv_path, written)
    print(
        f"Wrote {len(written)} run records to {csv_path} "
        f"({time.perf_counter() - started:.0f}s)"
    )

    _write_metadata(configs, split, runs, workers, output_dir, csv_name, indices, instances, append)
    return rows


def _write_metadata(
    configs: Dict[str, GAConfig],
    split: str,
    runs: int,
    workers: int,
    output_dir: str,
    csv_name: str,
    indices: Sequence[int],
    instances: Sequence,
    append: bool,
) -> None:
    """Mirror the sidecar ``run_experiments`` writes, minus the RL fields, so
    the two result directories are readable in the same way.

    When appending, earlier variants and invocations are kept: the CSV holds
    runs from several invocations, so the metadata has to as well.
    """
    meta_path = os.path.join(output_dir, os.path.splitext(csv_name)[0] + "_metadata.json")
    metadata: Dict = {"variants": {}, "invocations": []}
    if append and os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            loaded = json.load(fh)
        metadata["variants"] = loaded.get("variants", {})
        metadata["invocations"] = loaded.get("invocations", [])

    metadata["pipeline"] = "static_ga"
    metadata["split"] = split
    metadata["runs_per_instance"] = runs
    metadata["instances"] = [instances[i].name for i in indices]
    metadata["seed_rule"] = "run_seed = 42 + 100 * instance_index + run; shared by all variants"
    metadata["split_sizes"] = {name: split_size(name) for name in ("train", "test")}
    metadata["environment"] = environment_info(workers)
    metadata["variants"].update({key: config.to_dict() for key, config in configs.items()})
    metadata["invocations"].append(
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "argv": sys.argv[1:],
            "variants": sorted(configs),
            "runs_per_instance": runs,
        }
    )
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)
    print(f"Wrote configuration metadata to {meta_path}")


# ----------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------


def _unit_interval(name: str, *, upper_open: bool = False):
    """argparse type for a rate in [0, 1] (or [0, 1) when ``upper_open``)."""

    def parse(text: str) -> float:
        try:
            value = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{name} must be a number, got {text!r}")
        if value < 0.0 or value > 1.0 or (upper_open and value == 1.0):
            top = "1 (exclusive)" if upper_open else "1"
            raise argparse.ArgumentTypeError(f"{name} must be between 0 and {top}, got {value}")
        return value

    return parse


def _positive_int(name: str):
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{name} must be an integer, got {text!r}")
        if value < 1:
            raise argparse.ArgumentTypeError(f"{name} must be at least 1, got {value}")
        return value

    return parse


def parse_schedule(entries: Sequence[str]) -> Tuple[Tuple[int, float], ...]:
    """``["0:0.1", "100:0.5"]`` -> ``((0, 0.1), (100, 0.5))``."""
    schedule: List[Tuple[int, float]] = []
    for entry in entries:
        generation_text, _, rate_text = entry.partition(":")
        try:
            generation, rate = int(generation_text), float(rate_text)
        except ValueError:
            raise SystemExit(f"--ls-schedule entries look like GEN:RATE, got {entry!r}")
        if generation < 0 or not 0.0 <= rate <= 1.0:
            raise SystemExit(
                f"--ls-schedule needs a non-negative generation and a rate in [0, 1], "
                f"got {entry!r}"
            )
        schedule.append((generation, rate))
    schedule.sort()
    if not schedule or schedule[0][0] != 0:
        raise SystemExit("--ls-schedule must start at generation 0, e.g. 0:0.1 100:0.5 200:1.0")
    return tuple(schedule)


def pick_reference_and_baseline(
    available: Sequence[str], reference: Optional[str], baseline: Optional[str]
) -> Tuple[str, str]:
    """Choose the two variants the report and figures compare against.

    The analysis was written around an RL reference.  With only static
    variants present the sensible default reference is the strongest
    local-search setting and the baseline the weakest, so the report's gain
    columns still answer a real question.
    """
    if reference and reference not in available:
        raise SystemExit(f"--reference {reference!r} is not in the results. Have: {list(available)}")
    if baseline and baseline not in available:
        raise SystemExit(f"--baseline {baseline!r} is not in the results. Have: {list(available)}")

    # An optional --tag sits between the rate and the target, so a family is
    # "<stem>[_tag]_<target>".  Matching the tag exactly rather than with .*
    # matters: "ga_ls10.*" would also swallow ga_ls100_*, and the whole point
    # of the baseline is that it is the *weakest* variant present.
    tag = r"(?:_[A-Za-z0-9-]+)?"

    def family(stem: str, target: str = "(?:offspring|elites)") -> str:
        return rf"{stem}{tag}_{target}"

    def first_matching(*patterns: str) -> Optional[str]:
        for pattern in patterns:
            for key in available:
                if re.fullmatch(pattern, key):
                    return key
        return None

    reference = (
        reference
        or first_matching(
            family("ga_ls100", "elites"),
            family("ga_ls100"),
            family("ga_sched"),
            family("ga_ls50"),
        )
        or available[0]
    )
    baseline = baseline or first_matching(
        rf"ga_ls00{tag}", family("ga_ls10", "offspring"), family("ga_ls10")
    )
    if baseline is None or baseline == reference:
        others = [key for key in available if key != reference]
        baseline = others[0] if others else reference
    return reference, baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_static_ga",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    stages = parser.add_argument_group("stages")
    stages.add_argument(
        "--only",
        nargs="+",
        choices=list(STAGES),
        default=None,
        help="run only these stages (default: all of them)",
    )
    stages.add_argument(
        "--skip",
        nargs="+",
        choices=list(STAGES),
        default=(),
        help="run everything except these stages",
    )

    grid = parser.add_argument_group("what to run")
    grid.add_argument(
        "--variants",
        nargs="+",
        default=["all"],
        metavar="NAME",
        help="variant names, group aliases or full variant keys (default: all). "
        "See --list-variants",
    )
    grid.add_argument(
        "--list-variants", action="store_true", help="print the variant catalogue and exit"
    )
    grid.add_argument(
        "--split",
        default="test",
        choices=("train", "test"),
        help="instance split to run on (default: test)",
    )
    grid.add_argument(
        "--instances",
        type=int,
        nargs="+",
        default=None,
        metavar="INDEX",
        help="restrict to these instance indices (default: every instance in the split)",
    )
    grid.add_argument(
        "--runs",
        type=_positive_int("--runs"),
        default=10,
        help="independent seeds per instance in the main grid (default: 10)",
    )
    grid.add_argument(
        "--timing-runs",
        type=_positive_int("--timing-runs"),
        default=3,
        help="seeds per instance in the sequential timing pass (default: 3)",
    )
    grid.add_argument(
        "--workers",
        type=_positive_int("--workers"),
        default=max(1, (os.cpu_count() or 4) - 2),
        help="parallel worker processes for the main grid; the timing stage always "
        "runs sequentially (default: cpu_count - 2)",
    )

    operators = parser.add_argument_group(
        "GA operators (shared by every variant in this invocation)"
    )
    operators.add_argument(
        "--population-size", type=_positive_int("--population-size"), default=200,
        help="default: 200",
    )
    operators.add_argument(
        "--generations", type=_positive_int("--generations"), default=300, help="default: 300"
    )
    operators.add_argument(
        "--tournament-k",
        type=_positive_int("--tournament-k"),
        default=3,
        help="tournament selection size; 1 makes selection uniformly random (default: 3)",
    )
    operators.add_argument(
        "--elitism-rate",
        type=_unit_interval("--elitism-rate", upper_open=True),
        default=0.1,
        help="fraction of the population carried over unchanged (default: 0.1)",
    )
    operators.add_argument(
        "--crossover-rate", type=_unit_interval("--crossover-rate"), default=0.8,
        help="default: 0.8",
    )
    operators.add_argument(
        "--mutation-rate", type=_unit_interval("--mutation-rate"), default=0.1,
        help="default: 0.1",
    )
    operators.add_argument(
        "--ls-iterations",
        type=_positive_int("--ls-iterations"),
        default=5,
        help="local-search steps per selected individual (default: 5)",
    )
    operators.add_argument(
        "--ls-rate",
        type=_unit_interval("--ls-rate"),
        default=None,
        help="local-search rate for the 'custom' variant; the catalogue variants "
        "carry their own rate",
    )
    operators.add_argument(
        "--ls-target",
        default="elites",
        choices=LS_TARGETS,
        help="where the 'custom' variant applies local search (default: elites)",
    )
    operators.add_argument(
        "--ls-schedule",
        nargs="+",
        default=None,
        metavar="GEN:RATE",
        help="local-search schedule for the deterministic variants, e.g. "
        "0:0.1 100:0.5 200:1.0 (default: "
        + " ".join(f"{g}:{r}" for g, r in DEFAULT_LS_SCHEDULE)
        + ")",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--results-dir",
        default=DEFAULT_STATIC_RESULTS_DIR,
        help=f"default: {DEFAULT_STATIC_RESULTS_DIR}",
    )
    output.add_argument(
        "--tag",
        default=None,
        help="suffix added to every variant key and label, so a second invocation with "
        "different operators stacks alongside the first instead of replacing it "
        "(letters, digits and dashes)",
    )
    output.add_argument(
        "--append",
        action="store_true",
        help="merge into an existing runs.csv instead of replacing it",
    )
    output.add_argument(
        "--no-trace",
        action="store_true",
        help="skip storing per-generation traces (the convergence and budget-matched "
        "analyses need them)",
    )
    output.add_argument(
        "--reference",
        default=None,
        help="variant the report and figures compare everything against "
        "(default: the strongest local-search variant present)",
    )
    output.add_argument(
        "--baseline",
        default=None,
        help="variant the per-instance gain is measured from "
        "(default: the weakest local-search variant present)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.list_variants:
        print_catalogue()
        return

    if args.tag is not None and not re.fullmatch(r"[A-Za-z0-9-]+", args.tag):
        raise SystemExit(f"--tag may only contain letters, digits and dashes, got {args.tag!r}")

    # Caught here because otherwise it surfaces generations deep as an opaque
    # "Sample larger than population" from random.sample inside selection.
    if args.tournament_k > args.population_size:
        raise SystemExit(
            f"--tournament-k ({args.tournament_k}) cannot exceed --population-size "
            f"({args.population_size}): selection draws that many distinct individuals."
        )

    schedule = parse_schedule(args.ls_schedule) if args.ls_schedule else DEFAULT_LS_SCHEDULE

    # The operator settings shared by every variant; build_configs fills in
    # the per-variant local-search fields on top of this.
    base = GAConfig(
        population_size=args.population_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        elitism_rate=args.elitism_rate,
        tournament_k=args.tournament_k,
        ls_iterations=args.ls_iterations,
        ls_schedule=schedule,
    )
    configs = build_configs(args.variants, base, args.ls_rate, args.ls_target, args.tag)

    stages = [s for s in (args.only or STAGES) if s not in args.skip]
    if not stages:
        raise SystemExit("--skip removed every stage; nothing to do.")

    print(f"Static GA pipeline -- stages: {', '.join(stages)}")
    print(f"Results directory : {args.results_dir}")
    # The operators and the variant list only describe what would be *run*;
    # an analyze/figures-only invocation reads both back from the CSV instead.
    if {"main", "timing"} & set(stages):
        print(
            "Operators         : "
            f"pop={args.population_size} gens={args.generations} "
            f"tournament_k={args.tournament_k} elitism={args.elitism_rate} "
            f"crossover={args.crossover_rate} mutation={args.mutation_rate} "
            f"ls_iterations={args.ls_iterations}"
        )
        if any(config.ls_controller == "schedule" for config in configs.values()):
            print("LS schedule       : " + " ".join(f"{g}:{r}" for g, r in schedule))
        print("Variants          :")
        for key, config in configs.items():
            rate = "scheduled" if config.ls_controller == "schedule" else f"{config.ls_rate:.0%}"
            print(f"  {key:<26} LS {rate} on {config.ls_target}")
    print()

    started = time.perf_counter()
    for stage in stages:
        print(f"\n{'=' * 78}\n== {stage}\n{'=' * 78}")
        stage_started = time.perf_counter()

        if stage == "instances":
            from core.problems import dump_instance_catalogue

            os.makedirs(args.results_dir, exist_ok=True)
            dump_instance_catalogue(os.path.join(args.results_dir, "instances.json"))

        elif stage == "main":
            run_static_grid(
                configs=configs,
                split=args.split,
                runs=args.runs,
                workers=args.workers,
                output_dir=args.results_dir,
                instance_indices=args.instances,
                csv_name="runs.csv",
                store_trace=not args.no_trace,
                append=args.append,
            )

        elif stage == "timing":
            print(
                "Sequential timing replication (one run at a time) so the wall-clock "
                "comparison is not distorted by parallel load."
            )
            run_static_grid(
                configs=configs,
                split=args.split,
                runs=args.timing_runs,
                workers=1,
                output_dir=args.results_dir,
                instance_indices=args.instances,
                csv_name="runs_timing.csv",
                store_trace=False,
                append=args.append,
            )

        elif stage in ("analyze", "figures"):
            csv_path = os.path.join(args.results_dir, "runs.csv")
            if not os.path.exists(csv_path):
                raise SystemExit(
                    f"{csv_path} does not exist -- run the 'main' stage first, or point "
                    f"--results-dir at a directory that already has one."
                )
            # Read the variants back out of the CSV rather than using the ones
            # this invocation built: with --append the file may hold more.
            available = sorted({row["variant"] for row in read_rows(csv_path)})
            reference, baseline = pick_reference_and_baseline(
                available, args.reference, args.baseline
            )
            print(f"Reference variant: {reference}   baseline: {baseline}")

            if stage == "analyze":
                from analysis.analyze import analyse

                analyse(
                    results_dir=args.results_dir,
                    csv_name="runs.csv",
                    split=args.split,
                    references=[reference],
                    baseline=baseline,
                )
            else:
                from visualization.plots import make_all_figures

                make_all_figures(
                    results_dir=args.results_dir,
                    csv_name="runs.csv",
                    split=args.split,
                    reference=reference,
                    baseline=baseline,
                )

        print(f"-- {stage} finished in {(time.perf_counter() - stage_started) / 60:.1f} min")

    print(f"\nStatic GA pipeline finished in {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
