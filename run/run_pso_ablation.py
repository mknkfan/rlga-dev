"""
Cognitive x social coefficient study for PSO.

Three settings of the acceleration coefficients, on the same instances, seeds
and initial populations the static study uses --

* ``c1 = 1, c2 = 3``   social-heavy: particles mostly follow the global best
* ``c1 = 2, c2 = 2``   balanced: the textbook symmetric setting
* ``c1 = 3, c2 = 1``   cognitive-heavy: particles mostly follow their own best

-- run ``--runs`` times on every selected instance (default: 20 runs x 10 test
instances = 200 runs per setting, 600 runs in total).

The three settings share ``c1 + c2 = 4``, so the *total* acceleration towards
the two attractors is the same in each and the only thing that changes is how
it is split between them.  Everything else is held at the baseline PSO of
``optimization.metaheuristics``: inertia ``w = 0.7298`` (Clerc & Kennedy's
constriction factor), velocities clamped to half of each variable's range,
population 200, 300 iterations, and the same 60 000-evaluation budget the GA
gets.  ``--pso-w`` / ``--pso-vmax-fraction`` move that shared setting if a
second sweep is wanted; ``--pairs`` replaces the three coefficient pairs.

Note that at ``c1 + c2 = 4`` the swarm is *not* in the Clerc & Kennedy
constricted regime, which needs ``c1 + c2 = phi = 4.1`` with ``chi = 0.7298``;
the baseline's ``c1 = c2 = 1.49618`` is that regime.  Holding the sum at 4
keeps the three cells comparable to each other, which is what this study asks
for, at the cost of being marginally more explosive than the baseline.  The
velocity clamp is what bounds the swarm in practice.

Pairing
-------
The run seed is ``optimization.run_experiments.run_seed_for`` -- the same rule
the main study uses -- so on a given (instance, run) pair all three settings
start from the *same* initial population and the same initial velocities.  The
cells are therefore paired, which is what makes the per-seed ranking and the
Wilcoxon tests in the summary meaningful.

What gets written (under ``--results-dir``, default ``results_pso_ablation/``)
------------------------------------------------------------------------------
``raw/<combo>/<instance>_run<k>.json``  one JSON per run: the coefficients,
                                        the scalars and (by default) the
                                        convergence traces.
``runs.json`` / ``runs.csv``            every run record, scalars only.
``summary.json`` / ``summary.csv``      per setting: mean and median of
                                        fitness, evaluations and compute time.
``summary_per_instance.csv``            the same, split by instance.
``pairwise.csv``                        every pair of settings: paired
                                        Wilcoxon, effect sizes, win/loss.
``summary.md``                          the readable report.
``metadata.json``                       settings, parameters, environment.
``figures/convergence.png``             median best-so-far per setting.
``figures/per_instance.png``            per-instance fitness distributions.

Fitness is *minimised*, and an infeasible run has fitness ``inf``.  JSON has
no ``inf`` literal, so such a run is written as ``best_fitness: null`` with
``feasible: false``; the fitness statistics are taken over feasible runs only
and every table carries the feasible count.

Examples
--------
The default study, 8 workers::

    python -m run.run_pso_ablation --workers 8

A quick smoke test first (seconds)::

    python -m run.run_pso_ablation --instances 0 --runs 2 \
        --generations 20 --population-size 30 \
        --results-dir results_pso_ablation/smoke

Resume an interrupted run (skips every (setting, instance, seed) already on
disk), then rebuild the tables and figures from whatever is there::

    python -m run.run_pso_ablation --workers 8 --resume
    python -m run.run_pso_ablation --only aggregate figures
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

from run.paths import DEFAULT_PSO_ABLATION_RESULTS_DIR
from analysis.stats_utils import cliffs_delta, holm_bonferroni, paired_effect_sizes
from core.problems import build_instances
from optimization.ga_core import make_initial_population
from optimization.metaheuristics import MetaConfig, build_optimizer
from optimization.run_experiments import environment_info, run_seed_for

#: The settings this study asks for: (c1, c2), all summing to 4.
DEFAULT_PAIRS: Tuple[Tuple[float, float], ...] = ((1.0, 3.0), (2.0, 2.0), (3.0, 1.0))

#: Repetitions per (setting, instance).
DEFAULT_RUNS: int = 20

#: Its own directory under ``results/``; the main study is never written to
#: from here.
DEFAULT_RESULTS_DIR: str = DEFAULT_PSO_ABLATION_RESULTS_DIR

#: Rough sequential cost of one PSO run at 300 iterations x population 200,
#: measured on this machine.  Used only for the estimate printed at startup.
CPU_SECONDS_PER_RUN: float = 3.4

STAGES: Tuple[str, ...] = ("run", "aggregate", "figures")

#: Traces kept in each run's JSON under ``--traces compact``: what a
#: convergence plot needs, without the RL columns PSO never fills.
COMPACT_TRACES: Tuple[str, ...] = (
    "best_so_far",
    "gen_best",
    "gen_avg",
    "diversity",
    "evaluations",
    "elapsed",
)

#: Scalar columns of ``runs.csv``, in order.
RUN_FIELDS: Tuple[str, ...] = (
    "combo",
    "c1",
    "c2",
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
# the settings
# ----------------------------------------------------------------------


def _num(value: float) -> str:
    """Compact, round-trippable number text: 1.0 -> '1', 1.49618 -> '1.49618'."""
    return f"{value:g}"


def combo_key(c1: float, c2: float) -> str:
    """Directory / row name of one setting, e.g. ``c1-1_c2-3``."""
    return f"c1-{_num(c1)}_c2-{_num(c2)}"


def build_configs(
    base: MetaConfig, pairs: Sequence[Tuple[float, float]]
) -> Dict[str, MetaConfig]:
    """``{combo key: MetaConfig}`` for the coefficient settings.

    ``base`` carries everything the settings share -- population size,
    iteration budget, inertia and the velocity clamp -- so two settings differ
    in ``c1`` and ``c2`` and nothing else.
    """
    configs: Dict[str, MetaConfig] = {}
    for c1, c2 in pairs:
        key = combo_key(c1, c2)
        if key in configs:
            raise SystemExit(f"duplicate coefficient pair: c1={_num(c1)}, c2={_num(c2)}")
        configs[key] = replace(
            base,
            label=f"PSO c1={_num(c1)} c2={_num(c2)}",
            pso_c1=float(c1),
            pso_c2=float(c2),
        )
    return configs


# ----------------------------------------------------------------------
# running
# ----------------------------------------------------------------------


@dataclass
class PSOJob:
    """One (setting, instance, seed) run.

    Picklable, and it carries its own configuration, so a worker needs no
    registry lookup of its own.
    """

    combo: str
    config: MetaConfig
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


def execute(job: PSOJob) -> Dict:
    """Run one job and write its JSON.  Executed in a worker process."""
    instance = build_instances(job.split)[job.instance_index]
    seed = run_seed_for(job.instance_index, job.run)

    optimizer = build_optimizer(
        "pso",
        instance.machines,
        instance.sequence,
        instance.robot_position,
        instance.workspace_bounds,
        config=job.config,
        seed=seed,
        instance_name=instance.name,
    )
    # The same seed gives every setting the same initial population and, since
    # the optimizer's own generator is seeded from it too, the same initial
    # velocities -- which is what pairs the settings against each other.
    population = make_initial_population(
        instance.machines, instance.workspace_bounds, job.config.population_size, seed=seed
    )
    result = optimizer.optimize(initial_population=population)

    record: Dict = {
        "combo": job.combo,
        "c1": job.config.pso_c1,
        "c2": job.config.pso_c2,
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
    configs: Dict[str, MetaConfig],
    split: str,
    instance_indices: Sequence[int],
    runs: int,
    workers: int,
    output_dir: str,
    traces: str,
    resume: bool,
) -> List[Dict]:
    """Run every (setting, instance, seed) and return the run records."""
    instances = build_instances(split)

    jobs: List[PSOJob] = []
    done: List[Dict] = []
    for index in instance_indices:
        for run in range(runs):
            for combo, config in configs.items():
                path = run_json_path(output_dir, combo, instances[index].name, run)
                if resume and os.path.exists(path):
                    done.append(load_record(path))
                    continue
                jobs.append(
                    PSOJob(
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
        f"Running {len(configs)} settings x {len(instance_indices)} '{split}' instances x "
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
    """Average rank of each setting across the paired (instance, seed) sets.

    Within one (instance, seed) every setting started from the same initial
    population, so the cells can be ranked directly: 1 is the best (lowest)
    fitness, tied runs share the average rank of their block, and an
    infeasible run sorts last.  Averaging those ranks compares the settings
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
    """Per-setting (optionally per-instance) mean / median tables."""
    ranks = {} if by_instance else mean_ranks(rows)

    groups: Dict[Tuple, List[Dict]] = {}
    for row in rows:
        key = (row["combo"], row["instance"]) if by_instance else (row["combo"],)
        groups.setdefault(key, []).append(row)

    table: List[Dict] = []
    for members in groups.values():
        first = members[0]
        feasible = [r for r in members if r["feasible"]]
        entry: Dict = {"combo": first["combo"], "c1": first["c1"], "c2": first["c2"]}
        if by_instance:
            entry["instance"] = first["instance"]
            entry["instance_index"] = first["instance_index"]
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

    table.sort(key=lambda e: (e["c1"], e.get("instance_index", -1)))
    return table


def _ordered_instances(rows: Sequence[Dict]) -> List[str]:
    """Instance names in split order -- ``test_10`` sorts after ``test_9``."""
    seen: Dict[str, int] = {}
    for row in rows:
        seen.setdefault(row["instance"], int(row["instance_index"]))
    return [name for name, _ in sorted(seen.items(), key=lambda item: item[1])]


def _paired_series(
    rows: Sequence[Dict], left: str, right: str
) -> Tuple[List[float], List[float], int]:
    """Fitness of two settings on the (instance, seed) pairs they both ran.

    A pair in which either side is infeasible carries no usable difference and
    is dropped; the count of those is returned so the report can state it.
    """
    indexed: Dict[str, Dict[Tuple[int, int], Optional[float]]] = {left: {}, right: {}}
    for row in rows:
        if row["combo"] in indexed:
            indexed[row["combo"]][(row["instance_index"], row["run"])] = row["best_fitness"]

    x: List[float] = []
    y: List[float] = []
    dropped = 0
    for key in sorted(set(indexed[left]) & set(indexed[right])):
        a, b = indexed[left][key], indexed[right][key]
        if a is None or b is None:
            dropped += 1
            continue
        x.append(float(a))
        y.append(float(b))
    return x, y, dropped


def _instance_win_counts(rows: Sequence[Dict], left: str, right: str) -> Tuple[int, int, int]:
    """Instances on which ``left`` has the lower / higher / equal mean fitness.

    Pooling 200 paired differences across instances mixes fitness scales; this
    counts instances instead, which does not.
    """
    per_instance: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        if row["combo"] in (left, right) and row["best_fitness"] is not None:
            per_instance.setdefault(row["instance"], {}).setdefault(row["combo"], []).append(
                float(row["best_fitness"])
            )

    wins = losses = ties = 0
    for members in per_instance.values():
        if left not in members or right not in members:
            continue
        a, b = float(np.mean(members[left])), float(np.mean(members[right]))
        if a < b:
            wins += 1
        elif a > b:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties


def pairwise_comparisons(rows: Sequence[Dict], combos: Sequence[str]) -> List[Dict]:
    """Paired Wilcoxon, effect sizes and win counts for every pair of settings.

    Every setting saw the same (instance, seed) pairs from the same initial
    population, so the comparison is paired.  ``p_holm`` is the Holm-Bonferroni
    adjustment over the pairs tested here.
    """
    comparisons: List[Dict] = []
    for i, left in enumerate(combos):
        for right in combos[i + 1 :]:
            x, y, dropped = _paired_series(rows, left, right)
            if not x:
                continue
            test = paired_effect_sizes(x, y)
            delta, magnitude = cliffs_delta(x, y)
            inst_wins, inst_losses, inst_ties = _instance_win_counts(rows, left, right)
            comparisons.append(
                {
                    "left": left,
                    "right": right,
                    "n_pairs": test.n_pairs,
                    "dropped_pairs": dropped,
                    "mean_difference": test.mean_difference,
                    "median_difference": test.median_difference,
                    "difference_ci_low": test.difference_ci_low,
                    "difference_ci_high": test.difference_ci_high,
                    "statistic": test.statistic,
                    "p_value": test.p_value,
                    "method": test.method,
                    "rank_biserial": test.rank_biserial,
                    "cohens_dz": test.cohens_dz,
                    "cliffs_delta": delta,
                    "cliffs_magnitude": magnitude,
                    "seed_wins": test.n_wins,
                    "seed_losses": test.n_losses,
                    "seed_ties": test.n_ties,
                    "instance_wins": inst_wins,
                    "instance_losses": inst_losses,
                    "instance_ties": inst_ties,
                }
            )

    for entry, adjusted in zip(
        comparisons, holm_bonferroni([e["p_value"] for e in comparisons])
    ):
        entry["p_holm"] = adjusted
    return comparisons


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


def _marker(p: Optional[float]) -> str:
    if p is None or not math.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def write_report(
    path: str,
    summary: Sequence[Dict],
    comparisons: Sequence[Dict],
    rows: Sequence[Dict],
    per_instance: Sequence[Dict],
    base: MetaConfig,
    split: str,
    runs: int,
    n_instances: int,
) -> None:
    """The readable report: the main table, the pairwise tests, per instance."""
    best = min(
        (e for e in summary if e["fitness_mean"] is not None),
        key=lambda e: e["fitness_mean"],
        default=None,
    )
    best_rank = min(
        (e for e in summary if e["mean_rank"] is not None),
        key=lambda e: e["mean_rank"],
        default=None,
    )
    label = {e["combo"]: f"c1={_num(e['c1'])}, c2={_num(e['c2'])}" for e in summary}

    lines: List[str] = []
    lines.append("# Cognitive x social coefficients - PSO\n")
    lines.append(
        f"{len(summary)} coefficient settings, {runs} runs on each of {n_instances} "
        f"'{split}' instances ({len(rows)} runs recorded).\n"
    )
    lines.append(
        f"Shared parameters: population {base.population_size}, {base.generations} iterations, "
        f"inertia w={base.pso_w}, velocity clamp {base.pso_vmax_fraction:g} x range, "
        f"budget {base.evaluation_budget()} evaluations.\n"
    )
    lines.append(
        "`c1` is the cognitive coefficient (pull towards the particle's own best) and `c2` the "
        "social one (pull towards the swarm's best). The settings share `c1 + c2`, so the total "
        "acceleration is fixed and only its split between the two attractors changes.\n"
    )
    lines.append(
        "**Fitness is minimised** (lower is better), and infeasible runs are excluded from the "
        "fitness statistics; `n_feasible` reports how many runs in each cell were feasible. "
        "Evaluation counts and times cover every run.\n"
    )
    lines.append(
        "`mean_rank` ranks the settings against each other within every (instance, seed) pair - "
        "all of which start from the same initial population - and averages those ranks. It is "
        "the scale-free comparison; the pooled fitness columns mix instances whose fitness "
        "scales differ.\n"
    )
    if best is not None:
        lines.append(
            f"Lowest mean fitness: **{label[best['combo']]}** (mean {best['fitness_mean']:.4f}, "
            f"median {_cell(best['fitness_median'])})."
        )
    if best_rank is not None:
        lines.append(
            f"Best mean rank: **{label[best_rank['combo']]}** ({best_rank['mean_rank']:.2f}).\n"
        )

    lines.append("\n## Per setting\n")
    lines.append(
        "| c1 | c2 | n | feasible | fitness mean | fitness sd | fitness median | fitness min | "
        "evals mean | wall mean (s) | CPU mean (s) | mean rank |"
    )
    lines.append("|" + "---|" * 12)
    for entry in summary:
        lines.append(
            "| "
            + " | ".join(
                (
                    _num(entry["c1"]),
                    _num(entry["c2"]),
                    str(entry["n_runs"]),
                    f"{entry['n_feasible']}/{entry['n_runs']}",
                    _cell(entry["fitness_mean"]),
                    _cell(entry["fitness_sd"]),
                    _cell(entry["fitness_median"]),
                    _cell(entry["fitness_min"]),
                    _cell(entry["evaluations_mean"], ".0f"),
                    _cell(entry["execution_time_mean"], ".2f"),
                    _cell(entry["cpu_time_mean"], ".2f"),
                    _cell(entry["mean_rank"], ".2f"),
                )
            )
            + " |"
        )

    if comparisons:
        lines.append("\n## Pairwise comparisons\n")
        lines.append(
            "Two-sided Wilcoxon signed-rank on the paired (instance, seed) runs, Holm-Bonferroni "
            "adjusted over the pairs below. A **negative** mean difference means the left setting "
            "is better. `seed w/l` counts the paired runs the left setting won and lost; "
            "`instance w/l` counts the instances where its mean fitness was lower - the count "
            "that does not pool different fitness scales.\n"
        )
        lines.append(
            "| comparison | n pairs | mean diff | 95% CI | median diff | p (Holm) | | "
            "Cliff's delta | seed w/l | instance w/l |"
        )
        lines.append("|" + "---|" * 10)
        for entry in comparisons:
            lines.append(
                "| "
                + " | ".join(
                    (
                        f"{label.get(entry['left'], entry['left'])} vs "
                        f"{label.get(entry['right'], entry['right'])}",
                        str(entry["n_pairs"]),
                        _cell(entry["mean_difference"]),
                        f"[{entry['difference_ci_low']:.4f}, {entry['difference_ci_high']:.4f}]",
                        _cell(entry["median_difference"]),
                        f"{entry['p_holm']:.3g}",
                        _marker(entry["p_holm"]),
                        f"{entry['cliffs_delta']:.3f} ({entry['cliffs_magnitude']})",
                        f"{entry['seed_wins']}/{entry['seed_losses']}",
                        f"{entry['instance_wins']}/{entry['instance_losses']}",
                    )
                )
                + " |"
            )
        dropped = sum(e["dropped_pairs"] for e in comparisons)
        if dropped:
            lines.append(
                f"\n{dropped} paired difference(s) were dropped across the comparisons above "
                "because one side of the pair was infeasible.\n"
            )

    instances = _ordered_instances(per_instance)
    combos = [e["combo"] for e in summary]
    indexed = {(e["combo"], e["instance"]): e for e in per_instance}
    for title, field, spec in (
        ("Mean fitness", "fitness_mean", ".3f"),
        ("Median fitness", "fitness_median", ".3f"),
        ("Feasible runs", "n_feasible", ".0f"),
    ):
        lines.append(f"\n## {title} per instance\n")
        lines.append("| instance | " + " | ".join(label.get(c, c) for c in combos) + " |")
        lines.append("|" + "---|" * (len(combos) + 1))
        for instance in instances:
            cells = [_cell(indexed.get((c, instance), {}).get(field), spec) for c in combos]
            lines.append(f"| {instance} | " + " | ".join(cells) + " |")

    lines.append(
        "\n---\n\nWall-clock time is inflated and noisy whenever the study was run with "
        "`--workers > 1`, because parallel runs share the machine; CPU time and the evaluation "
        "count are the reliable effort measures in that case.\n"
    )

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def write_outputs(
    output_dir: str,
    rows: Sequence[Dict],
    base: MetaConfig,
    split: str,
    runs: int,
    n_instances: int,
) -> List[Dict]:
    """``runs.*``, ``summary.*``, ``pairwise.csv`` and the report."""
    rows = sorted(rows, key=lambda r: (r["combo"], r["instance_index"], r["run"]))
    summary = summarise(rows)
    per_instance = summarise(rows, by_instance=True)
    comparisons = pairwise_comparisons(rows, [entry["combo"] for entry in summary])

    with open(os.path.join(output_dir, "runs.json"), "w", encoding="utf-8") as fh:
        json.dump(list(rows), fh, indent=1)
    write_csv(os.path.join(output_dir, "runs.csv"), rows, RUN_FIELDS)
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"per_setting": summary, "pairwise": comparisons}, fh, indent=1)
    write_csv(os.path.join(output_dir, "summary.csv"), summary)
    write_csv(os.path.join(output_dir, "summary_per_instance.csv"), per_instance)
    if comparisons:
        write_csv(os.path.join(output_dir, "pairwise.csv"), comparisons)
    write_report(
        os.path.join(output_dir, "summary.md"),
        summary,
        comparisons,
        rows,
        per_instance,
        base,
        split,
        runs,
        n_instances,
    )

    print(f"Wrote {len(rows)} run records to {os.path.join(output_dir, 'runs.json')} (and runs.csv)")
    print(f"Wrote {len(summary)} setting summaries to {os.path.join(output_dir, 'summary.csv')}")
    print(f"Wrote {len(comparisons)} pairwise tests to {os.path.join(output_dir, 'pairwise.csv')}")
    print(f"Wrote the report to {os.path.join(output_dir, 'summary.md')}")
    return summary


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------


def _load_traces(output_dir: str, combo: str) -> List[Dict]:
    """Every stored trace of one setting, keyed by instance for normalising."""
    combo_dir = os.path.join(output_dir, "raw", combo)
    traces: List[Dict] = []
    if not os.path.isdir(combo_dir):
        return traces
    for name in sorted(os.listdir(combo_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(combo_dir, name), encoding="utf-8") as fh:
            payload = json.load(fh)
        stored = payload.get("traces")
        if not stored or "best_so_far" not in stored:
            continue
        traces.append(
            {
                "instance": payload["instance"],
                "best_so_far": np.array(
                    [np.nan if v is None else float(v) for v in stored["best_so_far"]], dtype=float
                ),
            }
        )
    return traces


def make_figures(output_dir: str, rows: Sequence[Dict], summary: Sequence[Dict]) -> None:
    """Convergence curves and per-instance distributions, one series per setting."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    combos = [entry["combo"] for entry in summary]
    labels = {e["combo"]: f"c1={_num(e['c1'])}, c2={_num(e['c2'])}" for e in summary}
    colours = plt.cm.tab10(np.linspace(0, 1, 10))

    # -- convergence ---------------------------------------------------
    # Instances differ in fitness scale, so each run's curve is divided by
    # that instance's median final fitness across all settings before the
    # curves are pooled; 1.0 on the y axis is therefore "typical final
    # quality on this instance".
    finals: Dict[str, List[float]] = {}
    for row in rows:
        if row["best_fitness"] is not None:
            finals.setdefault(row["instance"], []).append(float(row["best_fitness"]))
    scale = {name: float(np.median(values)) for name, values in finals.items() if values}

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    plotted = False
    for index, combo in enumerate(combos):
        traces = _load_traces(output_dir, combo)
        curves = [
            trace["best_so_far"] / scale[trace["instance"]]
            for trace in traces
            if trace["instance"] in scale
        ]
        if not curves:
            continue
        width = max(curve.size for curve in curves)
        stacked = np.full((len(curves), width), np.nan)
        for i, curve in enumerate(curves):
            stacked[i, : curve.size] = curve
            # A run that stopped early holds its final value, so the pooled
            # median is not biased by which runs are still contributing.
            stacked[i, curve.size :] = curve[-1] if curve.size else np.nan
        with np.errstate(invalid="ignore"):
            median = np.nanmedian(stacked, axis=0)
            low = np.nanpercentile(stacked, 25, axis=0)
            high = np.nanpercentile(stacked, 75, axis=0)
        x = np.arange(width)
        colour = colours[index % len(colours)]
        ax.plot(x, median, color=colour, linewidth=2.0, label=labels.get(combo, combo))
        ax.fill_between(x, low, high, color=colour, alpha=0.15, linewidth=0)
        plotted = True

    if plotted:
        ax.set_xlabel("iteration")
        ax.set_ylabel("best-so-far fitness / instance median")
        ax.set_title("PSO convergence by acceleration coefficients (median, IQR band)")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25, linewidth=0.6)
        fig.tight_layout()
        path = os.path.join(figures_dir, "convergence.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)

    # -- per-instance distributions ------------------------------------
    instances = _ordered_instances(rows)
    fig, ax = plt.subplots(figsize=(max(7.5, 1.1 * len(instances)), 4.6))
    width = 0.8 / max(len(combos), 1)
    for index, combo in enumerate(combos):
        data = [
            [
                float(row["best_fitness"])
                for row in rows
                if row["combo"] == combo
                and row["instance"] == instance
                and row["best_fitness"] is not None
            ]
            for instance in instances
        ]
        positions = [i + (index - (len(combos) - 1) / 2) * width for i, _ in enumerate(instances)]
        keep = [(p, d) for p, d in zip(positions, data) if d]
        if not keep:
            continue
        colour = colours[index % len(colours)]
        box = ax.boxplot(
            [d for _, d in keep],
            positions=[p for p, _ in keep],
            widths=width * 0.85,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(colour)
            patch.set_alpha(0.55)
            patch.set_linewidth(0.8)
        ax.plot([], [], color=colour, linewidth=6, alpha=0.55, label=labels.get(combo, combo))

    ax.set_xticks(range(len(instances)))
    ax.set_xticklabels(instances, rotation=45, ha="right")
    ax.set_ylabel("best fitness (lower is better)")
    ax.set_title("Final fitness per instance by acceleration coefficients")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    path = os.path.join(figures_dir, "per_instance.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Wrote {path}")
    plt.close(fig)


# ----------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------


def _coefficient_pair(text: str) -> Tuple[float, float]:
    """Parser for ``--pairs``: ``"1,3"`` -> ``(1.0, 3.0)``."""
    parts = text.replace(":", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"--pairs expects 'c1,c2' (e.g. 1,3), got {text!r}")
    try:
        c1, c2 = (float(part) for part in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--pairs expects two numbers, got {text!r}") from None
    if c1 < 0 or c2 < 0:
        raise argparse.ArgumentTypeError(f"acceleration coefficients must be >= 0, got {text!r}")
    return (c1, c2)


def _non_negative_float(name: str):
    def parse(text: str) -> float:
        try:
            value = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{name} expects a number, got {text!r}")
        if value < 0:
            raise argparse.ArgumentTypeError(f"{name} must be >= 0, got {value}")
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
        prog="python -m run.run_pso_ablation",
        description="Cognitive x social acceleration coefficient study for PSO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    grid = parser.add_argument_group("the settings")
    grid.add_argument(
        "--pairs",
        type=_coefficient_pair,
        nargs="+",
        default=list(DEFAULT_PAIRS),
        metavar="c1,c2",
        help="acceleration coefficient pairs to compare",
    )
    grid.add_argument(
        "--runs",
        type=_positive_int("--runs"),
        default=DEFAULT_RUNS,
        help="repetitions of each setting on each instance",
    )
    grid.add_argument("--split", default="test", choices=("test", "train", "generalization"))
    grid.add_argument(
        "--instances",
        type=int,
        nargs="+",
        default=None,
        help="instance indices to use (default: the whole split)",
    )

    shared = parser.add_argument_group("shared PSO parameters")
    shared.add_argument("--population-size", type=_positive_int("--population-size"), default=200)
    shared.add_argument("--generations", type=_positive_int("--generations"), default=300)
    shared.add_argument(
        "--pso-w",
        type=_non_negative_float("--pso-w"),
        default=MetaConfig().pso_w,
        help="inertia weight, held fixed across the settings",
    )
    shared.add_argument(
        "--pso-vmax-fraction",
        type=_non_negative_float("--pso-vmax-fraction"),
        default=MetaConfig().pso_vmax_fraction,
        help="velocity clamp as a fraction of each variable's range",
    )

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
        help="how much per-iteration history each run's JSON keeps",
    )
    execution.add_argument(
        "--resume",
        action="store_true",
        help="skip any (setting, instance, seed) whose JSON already exists",
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

    n_available = len(build_instances(args.split))
    instance_indices = list(range(n_available)) if args.instances is None else list(args.instances)
    out_of_range = [i for i in instance_indices if not 0 <= i < n_available]
    if out_of_range:
        raise SystemExit(
            f"instance index/indices {out_of_range} out of range for split '{args.split}' "
            f"(0..{n_available - 1})"
        )

    base = MetaConfig(
        algorithm="pso",
        population_size=args.population_size,
        generations=args.generations,
        pso_w=args.pso_w,
        pso_vmax_fraction=args.pso_vmax_fraction,
    )
    configs = build_configs(base, args.pairs)
    stages = list(args.only or STAGES)

    total_runs = len(configs) * len(instance_indices) * args.runs
    scale = (args.generations * args.population_size) / (300 * 200)
    estimate = total_runs * CPU_SECONDS_PER_RUN * scale / 3600.0

    print("PSO acceleration coefficient study")
    print(f"Results directory : {args.results_dir}")
    print(f"Stages            : {', '.join(stages)}")
    print(
        "Settings          : "
        + ", ".join(f"c1={_num(c1)}/c2={_num(c2)}" for c1, c2 in args.pairs)
    )
    print(
        f"Design            : {len(configs)} settings x {len(instance_indices)} "
        f"'{args.split}' instances x {args.runs} runs = {total_runs} runs"
    )
    print(
        f"Shared parameters : pop={args.population_size} iters={args.generations} "
        f"w={args.pso_w} vmax={args.pso_vmax_fraction:g}x range "
        f"budget={base.evaluation_budget()} evals"
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
            "pairs": [[c1, c2] for c1, c2 in args.pairs],
            "settings": list(configs),
            "split": args.split,
            "instance_indices": instance_indices,
            "runs_per_instance": args.runs,
            "n_runs": len(rows),
            "traces": args.traces,
            "seed_rule": "run_seed = 42 + 100 * instance_index + run; shared by every setting",
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

    summary: List[Dict] = []
    if "aggregate" in stages or "figures" in stages:
        split, runs, n_instances = args.split, args.runs, len(instance_indices)
        # An aggregate-only invocation reads the runs back off disk, so it
        # covers whatever is there -- including earlier, interrupted passes.
        # The report must then describe the parameters those runs actually
        # used, not this invocation's defaults, so they come from the metadata.
        if not rows:
            rows = collect_records(args.results_dir)
            recorded = load_metadata(args.results_dir)
            if recorded is not None:
                stored = dict(recorded["base_config"])
                # to_dict() resolves max_evaluations to the number the runs
                # actually used; MetaConfig recomputes it from the budget.
                stored.pop("max_evaluations", None)
                base = MetaConfig(**stored)
                split = recorded["split"]
                runs = recorded["runs_per_instance"]
                n_instances = len(recorded["instance_indices"])
            else:
                print(
                    "No metadata.json in the results directory: the report will describe this "
                    "invocation's parameters, which may not be the ones the runs used."
                )
        if "aggregate" in stages:
            summary = write_outputs(
                output_dir=args.results_dir,
                rows=rows,
                base=base,
                split=split,
                runs=runs,
                n_instances=n_instances,
            )

    if "figures" in stages:
        make_figures(args.results_dir, rows, summary or summarise(rows))

    print(f"\nDone in {(time.perf_counter() - started) / 60:.1f} min.")


if __name__ == "__main__":
    main()
