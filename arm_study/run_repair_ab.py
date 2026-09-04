"""
Does the placement repair actually buy anything?

    python -m arm_study.run_repair_ab                    # 10 instances x 15 seeds
    python -m arm_study.run_repair_ab --runs 5 --generations 100
    python -m arm_study.run_repair_ab --only aggregate

The repair projects an unplaceable child back onto the placeable set instead of
letting it score ``+inf``.  That is obviously good for the *placement* pass
rate; whether it is good for the *cycle time* is a different question and the
one this script answers, because placement is only the first of the
constraints.  A repaired layout still has to be reachable, still has to clear
the manipulability threshold, and still has to let the arm swing between
stations without hitting anything.  Repair buys evaluations that are no longer
wasted on geometry, and pays for them in two currencies: it collapses distinct
infeasible children onto the constraint boundary, which costs diversity, and it
biases the population towards layouts that sit *on* that boundary, which is
exactly where the arm has the least room.

Design
------
Paired, and the pairing is the whole point.  For each (instance, seed) the two
arms are handed the *same* rejection-sampled initial population and the same
seed, so the runs start identical and diverge only where the repair operator
fires.  Differences are therefore taken within a pair, and the statistics are
over pairs rather than over runs -- which removes the instance-to-instance
spread in cycle time, far the largest source of variance here, from the
comparison.

Both arms get the same evaluation budget, so "repair helps" has to mean a
better answer for the same spend.  Charging the repair nothing for its own
compute is the generous reading; ``cpu_time`` is reported so the reader can see
what it cost in seconds as well.

What is reported
----------------
``cycle_time``      the answer.  Minimised; ``null`` when a run never found a
                    feasible layout.
``feasible``        whether it found one at all.  Repair could in principle pay
                    for itself entirely by rescuing runs that would otherwise
                    return nothing, so the discordant pairs are counted
                    separately (McNemar) rather than folded into the means.
``mean_feasible``   mean over generations of the fraction of the population
                    with finite fitness -- the mechanism the repair acts on.
``evals_to_first``  evaluations spent before the first feasible layout.
``mean_diversity``  the cost side: mean gene spread over the run.

Significance is by paired sign-flip permutation, exact where the pair count
allows it and sampled otherwise, plus a Wilcoxon signed-rank statistic.  Both
are distribution-free, which matters because cycle-time differences here are
neither normal nor symmetric.  No scipy: the two tests are short enough to
write out, and doing so keeps the study's dependency list at numpy.

Seeding: run seed ``= 42 + 100 * instance_index + run``, the rule the rest of
the study uses, so a pair here starts from the same population as the
corresponding run of ``run_ga``.
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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluator import CycleTimeEvaluator  # noqa: E402
from ga import ConfigurableGA, GAConfig, make_initial_population  # noqa: E402
from problems import ALL_SPLITS, build_instances, describe  # noqa: E402

DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results_repair"
)
STAGES = ("run", "aggregate", "figures")
ARMS = ("baseline", "repair")

RUN_FIELDS = (
    "arm", "instance", "instance_index", "run", "run_seed",
    "cycle_time", "feasible", "generations", "evaluations",
    "execution_time", "cpu_time", "seeded_fraction",
    "mean_feasible", "final_feasible", "evals_to_first", "mean_diversity",
)


def run_seed_for(instance_index: int, run: int) -> int:
    """Seed shared by everything evaluated on a given (instance, run) pair."""
    return 42 + instance_index * 100 + run


def _jsonable(value: float) -> Optional[float]:
    value = float(value)
    return value if math.isfinite(value) else None


# ----------------------------------------------------------------------
# one run
# ----------------------------------------------------------------------


@dataclass
class Job:
    """One (arm, instance, seed) run.  Picklable, so a worker needs nothing."""

    arm: str
    split: str
    instance_index: int
    run: int
    config: GAConfig
    sigma_min_threshold: float
    dwell: float
    output_dir: str


def run_json_path(output_dir: str, arm: str, instance_name: str, run: int) -> str:
    return os.path.join(output_dir, "raw", f"{arm}_{instance_name}_run{run}.json")


def execute(job: Job) -> Dict:
    """Run one GA and write its JSON.  Executed in a worker process."""
    instance = build_instances(job.split)[job.instance_index]
    seed = run_seed_for(job.instance_index, job.run)

    evaluator = CycleTimeEvaluator(
        instance, sigma_min_threshold=job.sigma_min_threshold, dwell=job.dwell
    )
    ga = ConfigurableGA(instance, evaluator=evaluator, config=job.config, seed=seed)

    # The pairing: both arms are handed the identical starting population, so
    # the only thing that can separate them is the operator under test.
    population, seeded = make_initial_population(
        instance, job.config.population_size, seed=seed
    )
    result = ga.optimize(initial_population=population, seeded_fraction=seeded)

    feasible_fraction = np.asarray(result.feasible_fraction, dtype=float)
    evaluations = np.asarray(result.evaluations, dtype=float)
    best_so_far = np.asarray(result.best_so_far, dtype=float)

    reached = np.flatnonzero(np.isfinite(best_so_far))
    evals_to_first = float(evaluations[reached[0]]) if reached.size else None

    record = {
        "arm": job.arm,
        "instance": instance.name,
        "instance_index": job.instance_index,
        "run": job.run,
        "run_seed": seed,
        "cycle_time": _jsonable(result.best_fitness),
        "feasible": bool(result.feasible),
        "generations": int(result.generations),
        "evaluations": int(result.total_evaluations),
        "execution_time": round(float(result.execution_time), 4),
        "cpu_time": round(float(result.cpu_time), 4),
        "seeded_fraction": round(float(result.seeded_fraction), 4),
        "mean_feasible": round(float(feasible_fraction.mean()), 6),
        "final_feasible": round(float(feasible_fraction[-1]), 6),
        "evals_to_first": evals_to_first,
        "mean_diversity": round(float(np.nanmean(result.diversity)), 6),
        "best_so_far": [_jsonable(v) for v in best_so_far],
        "feasible_fraction_trace": [round(float(v), 5) for v in feasible_fraction],
        "evaluations_trace": [int(v) for v in evaluations],
        "chromosome": [round(float(v), 6) for v in result.best_chromosome],
    }

    path = run_json_path(job.output_dir, job.arm, instance.name, job.run)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    return record


def _progress(done: int, total: int, started: float, every: int = 10) -> None:
    if done % every and done != total:
        return
    elapsed = time.perf_counter() - started
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = (total - done) / rate if rate > 0 else float("nan")
    print(
        f"  {done:>4}/{total} runs  {elapsed / 60:5.1f} min elapsed"
        f"  ~{remaining / 60:5.1f} min left",
        flush=True,
    )


def run_grid(jobs: Sequence[Job], workers: int, resume: bool) -> List[Dict]:
    pending: List[Job] = []
    records: List[Dict] = []
    for job in jobs:
        instance_name = f"{job.split}_{job.instance_index + 1}"
        path = run_json_path(job.output_dir, job.arm, instance_name, job.run)
        if resume and os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                records.append(json.load(handle))
        else:
            pending.append(job)

    if records:
        print(f"  resuming: {len(records)} runs already on disk")
    if not pending:
        return records

    started = time.perf_counter()
    done = 0
    if workers <= 1:
        for job in pending:
            records.append(execute(job))
            done += 1
            _progress(done, len(pending), started)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for record in pool.map(execute, pending):
                records.append(record)
                done += 1
                _progress(done, len(pending), started)
    return records


# ----------------------------------------------------------------------
# paired statistics, written out rather than imported
# ----------------------------------------------------------------------


def wilcoxon_signed_rank(
    differences: Sequence[float],
) -> Tuple[Optional[float], Optional[float]]:
    """``(W, two-sided p)`` by the normal approximation, ties corrected.

    Zero differences are dropped, the standard Wilcoxon handling.  The normal
    approximation wants roughly ten or more non-zero pairs; below that the
    permutation test below is the one to read, and this returns ``None``.
    """
    d = np.asarray([x for x in differences if x != 0.0], dtype=float)
    n = d.size
    if n < 10:
        return None, None

    order = np.argsort(np.abs(d))
    magnitudes = np.abs(d)[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and magnitudes[j + 1] == magnitudes[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0  # mid-rank for the tie group
        i = j + 1

    w_plus = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())
    w = min(w_plus, w_minus)

    mean = n * (n + 1) / 4.0
    _, counts = np.unique(magnitudes, return_counts=True)
    tie_term = float((counts.astype(float) ** 3 - counts).sum())
    variance = (n * (n + 1) * (2 * n + 1) - 0.5 * tie_term) / 24.0
    if variance <= 0:
        return w, None
    z = (w - mean) / math.sqrt(variance)
    return w, float(math.erfc(abs(z) / math.sqrt(2.0)))


def sign_flip_test(
    differences: Sequence[float], samples: int = 200_000, seed: int = 0
) -> Optional[float]:
    """Two-sided p for ``mean(difference) == 0`` under sign exchangeability.

    Exact enumeration while ``2**n`` is small enough to enumerate, sampled
    above that.  The null is that a pair's sign is a coin flip, which is what
    the pairing buys: it needs no distributional assumption at all.
    """
    d = np.asarray([x for x in differences if math.isfinite(x)], dtype=float)
    n = d.size
    if n == 0:
        return None
    observed = abs(float(d.mean()))

    if n <= 20:
        signs = 1.0 - 2.0 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
        means = np.abs((signs * d).mean(axis=1))
        return float((means >= observed - 1e-15).mean())

    rng = np.random.default_rng(seed)
    hits = 0
    drawn = 0
    while drawn < samples:
        take = min(4096, samples - drawn)
        signs = rng.choice((-1.0, 1.0), size=(take, n))
        hits += int((np.abs((signs * d).mean(axis=1)) >= observed - 1e-15).sum())
        drawn += take
    return (hits + 1) / (samples + 1)


def bootstrap_ci(
    differences: Sequence[float], samples: int = 20_000, seed: int = 0
) -> Tuple[Optional[float], Optional[float]]:
    """Percentile 95 % interval for the mean paired difference."""
    d = np.asarray([x for x in differences if math.isfinite(x)], dtype=float)
    if d.size < 2:
        return None, None
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(samples, d.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar_exact(only_a: int, only_b: int) -> Optional[float]:
    """Two-sided exact McNemar p from the discordant counts alone.

    The concordant pairs carry no information about which arm is better at
    finding *a* feasible layout, so the test conditions on the discordant ones
    and asks whether they split like a fair coin.
    """
    n = only_a + only_b
    if n == 0:
        return None
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * tail))


# ----------------------------------------------------------------------
# aggregation
# ----------------------------------------------------------------------


def collect_records(output_dir: str) -> List[Dict]:
    raw = os.path.join(output_dir, "raw")
    if not os.path.isdir(raw):
        return []
    records = []
    for name in sorted(os.listdir(raw)):
        if name.endswith(".json"):
            with open(os.path.join(raw, name), encoding="utf-8") as handle:
                records.append(json.load(handle))
    return records


def pair_up(records: Sequence[Dict]) -> List[Dict]:
    """Match each baseline run to the repair run with the same seed."""
    index = {(r["arm"], r["instance"], r["run"]): r for r in records}
    pairs = []
    for (arm, instance, run), record in sorted(index.items()):
        if arm != "baseline":
            continue
        other = index.get(("repair", instance, run))
        if other is not None:
            pairs.append({"instance": instance, "run": run,
                          "baseline": record, "repair": other})
    return pairs


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    kept = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(kept)) if kept else None


def compare(pairs: Sequence[Dict], metric: str, lower_is_better: bool = True) -> Dict:
    """Paired summary of one metric: means, win counts, and two p-values.

    Pairs where either arm is missing the metric are dropped and counted, so a
    comparison can never quietly be taken over a different set of pairs than
    the one it claims.
    """
    usable: List[Tuple[float, float]] = []
    dropped = 0
    for pair in pairs:
        a, b = pair["baseline"].get(metric), pair["repair"].get(metric)
        if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
            dropped += 1
            continue
        usable.append((float(a), float(b)))

    if not usable:
        return {"metric": metric, "n_pairs": 0, "dropped": dropped}

    baseline = np.array([a for a, _ in usable])
    repair = np.array([b for _, b in usable])
    # Signed so that a positive difference always means "repair did better".
    delta = (baseline - repair) if lower_is_better else (repair - baseline)

    _, wilcoxon_p = wilcoxon_signed_rank(delta)
    low, high = bootstrap_ci(delta)
    return {
        "metric": metric,
        "n_pairs": len(usable),
        "dropped": dropped,
        "baseline_mean": float(baseline.mean()),
        "repair_mean": float(repair.mean()),
        "mean_gain": float(delta.mean()),
        "median_gain": float(np.median(delta)),
        "ci_low": low,
        "ci_high": high,
        "repair_wins": int((delta > 0).sum()),
        "baseline_wins": int((delta < 0).sum()),
        "ties": int((delta == 0).sum()),
        "permutation_p": sign_flip_test(delta),
        "wilcoxon_p": wilcoxon_p,
    }


def feasibility_table(pairs: Sequence[Dict]) -> Dict:
    both = only_baseline = only_repair = neither = 0
    for pair in pairs:
        a, b = pair["baseline"]["feasible"], pair["repair"]["feasible"]
        both += a and b
        only_baseline += a and not b
        only_repair += b and not a
        neither += not a and not b
    return {
        "both": both, "only_baseline": only_baseline,
        "only_repair": only_repair, "neither": neither,
        "n_pairs": len(pairs),
        "mcnemar_p": mcnemar_exact(only_baseline, only_repair),
    }


def per_instance(pairs: Sequence[Dict]) -> List[Dict]:
    names = sorted({pair["instance"] for pair in pairs}, key=lambda s: (len(s), s))
    rows = []
    for name in names:
        subset = [p for p in pairs if p["instance"] == name]
        both = [p for p in subset
                if p["baseline"]["cycle_time"] is not None
                and p["repair"]["cycle_time"] is not None]
        gains = [p["baseline"]["cycle_time"] - p["repair"]["cycle_time"] for p in both]
        rows.append({
            "instance": name,
            "n_pairs": len(subset),
            "n_both_feasible": len(both),
            "baseline_mean": _mean([p["baseline"]["cycle_time"] for p in both]),
            "repair_mean": _mean([p["repair"]["cycle_time"] for p in both]),
            "mean_gain": float(np.mean(gains)) if gains else None,
            "repair_wins": int(sum(g > 0 for g in gains)),
            "baseline_feasible": sum(p["baseline"]["feasible"] for p in subset),
            "repair_feasible": sum(p["repair"]["feasible"] for p in subset),
            "baseline_mean_feasible": _mean([p["baseline"]["mean_feasible"] for p in subset]),
            "repair_mean_feasible": _mean([p["repair"]["mean_feasible"] for p in subset]),
        })
    return rows


# ----------------------------------------------------------------------
# output
# ----------------------------------------------------------------------


def write_csv(path: str, rows: Sequence[Dict], fields: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _cell(value: Optional[float], spec: str = ".4f") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "--"
    return format(value, spec)


def _verdict(summary: Dict, unit: str, alpha: float = 0.05) -> str:
    if not summary.get("n_pairs"):
        return "no usable pairs"
    p = summary.get("permutation_p")
    gain = summary["mean_gain"]
    suffix = f" {unit}" if unit else ""
    if p is None or p > alpha:
        return (f"no detectable difference (mean {gain:+.4f}{suffix}, "
                f"p = {_cell(p, '.4f')})")
    direction = "repair better" if gain > 0 else "repair worse"
    return f"{direction} by {abs(gain):.4f}{suffix} on average (p = {p:.4f})"


def write_report(path: str, summaries: Dict, feasibility: Dict,
                 rows: Sequence[Dict], metadata: Dict) -> None:
    lines: List[str] = []
    add = lines.append

    add("# Does the placement repair help?\n")
    add(f"{feasibility['n_pairs']} paired runs "
        f"({metadata['n_instances']} instances x {metadata['runs']} seeds), "
        f"population {metadata['population_size']}, "
        f"{metadata['generations']} generations, "
        f"{metadata['budget']} evaluations per run.\n")
    add("Each pair is one (instance, seed): both arms start from the identical "
        "rejection-sampled population and differ only in whether an unplaceable "
        "child is projected back onto the placeable set after mutation. "
        "Differences are taken within a pair and signed so **positive means "
        "repair did better**.\n")

    add("\n## Verdict\n")
    add(f"- **Cycle time** -- {_verdict(summaries['cycle_time'], 's')}")
    add(f"- **Feasible population share** -- {_verdict(summaries['mean_feasible'], '')}")
    add(f"- **Evaluations to first feasible layout** -- "
        f"{_verdict(summaries['evals_to_first'], 'evals')}")
    add(f"- **Gene diversity** -- {_verdict(summaries['mean_diversity'], '')} "
        f"(but see the caveat below: this metric is nearly blind to the thing "
        f"repair would actually collapse)")
    add(f"- **CPU time** -- {_verdict(summaries['cpu_time'], 's')}")

    add("\n## Runs that found a feasible layout at all\n")
    add("| both | baseline only | repair only | neither | McNemar p |")
    add("| ---: | ---: | ---: | ---: | ---: |")
    add(f"| {feasibility['both']} | {feasibility['only_baseline']} | "
        f"{feasibility['only_repair']} | {feasibility['neither']} | "
        f"{_cell(feasibility['mcnemar_p'], '.4f')} |")
    add("\nOnly the discordant pairs carry information here; the exact test "
        "conditions on them.\n")

    add("\n## Paired metrics\n")
    add("| metric | pairs | baseline | repair | mean gain | 95% CI | "
        "repair wins | baseline wins | perm p | Wilcoxon p |")
    add("| --- | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: |")
    for key, spec in (("cycle_time", ".4f"), ("mean_feasible", ".4f"),
                      ("final_feasible", ".4f"), ("evals_to_first", ".0f"),
                      ("mean_diversity", ".4f"), ("cpu_time", ".2f")):
        s = summaries[key]
        if not s.get("n_pairs"):
            add(f"| `{key}` | 0 | -- | -- | -- | -- | -- | -- | -- | -- |")
            continue
        ci = f"[{_cell(s['ci_low'], spec)}, {_cell(s['ci_high'], spec)}]"
        add(f"| `{key}` | {s['n_pairs']} | {_cell(s['baseline_mean'], spec)} | "
            f"{_cell(s['repair_mean'], spec)} | {_cell(s['mean_gain'], spec)} | {ci} | "
            f"{s['repair_wins']} | {s['baseline_wins']} | "
            f"{_cell(s['permutation_p'], '.4f')} | {_cell(s['wilcoxon_p'], '.4f')} |")
    add("\n`cycle_time` and `evals_to_first` are taken over the pairs where "
        "**both** arms found a feasible layout; the dropped counts are in "
        "`pairs_summary.csv`. Every row is signed so that positive favours "
        "repair -- for `mean_diversity` that means repair held *more* spread, "
        "and for `cpu_time` that it spent *less* time.\n")
    add("\n> **Caveat on `mean_diversity`.** `population_diversity` averages the "
        "per-gene standard deviation over every gene, and the rotation genes "
        "span 0-360 while the position genes span about 1.15 m. Rotation "
        "spread is therefore two orders of magnitude larger and the average is "
        "essentially rotation spread alone -- which is the one gene the repair "
        "never touches. Read this row as a check that repair leaves orientation "
        "diversity undisturbed, **not** as a measurement of whether it presses "
        "machines together onto the placement boundary. Positional collapse is "
        "not measured here: it would need a position-only trace in `ga.py`, "
        "and adding one changes what every archived study reports.\n")

    add("\n## Per instance\n")
    add("| instance | pairs | both feasible | baseline s | repair s | gain s | "
        "repair wins | baseline feas. | repair feas. | baseline feas. share | "
        "repair feas. share |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        add(f"| {row['instance']} | {row['n_pairs']} | {row['n_both_feasible']} | "
            f"{_cell(row['baseline_mean'])} | {_cell(row['repair_mean'])} | "
            f"{_cell(row['mean_gain'])} | {row['repair_wins']} | "
            f"{row['baseline_feasible']} | {row['repair_feasible']} | "
            f"{_cell(row['baseline_mean_feasible'])} | "
            f"{_cell(row['repair_mean_feasible'])} |")

    add("\n## Reading it\n")
    add("**The budget is equal in evaluations, not in compute, and the two are "
        "far apart here.** An infeasible layout is rejected by the placement "
        "test and costs about 0.36 ms; a feasible one runs inverse kinematics, "
        "the manipulability check and the swept-path collision test over four "
        "branches, and costs about 6.1 ms -- some seventeen times more. Repair "
        "works by converting the cheap evaluations into expensive ones, so the "
        "repair arm buys strictly more computation for the same nominal "
        "budget. That is why `cpu_time` is in the table and why it should be "
        "read alongside the cycle time rather than as a footnote: a win at "
        "equal evaluations is a weaker claim than a win at equal seconds, and "
        "only the `cpu_time` row says which one this is.\n")
    add("The feasible-share column is the mechanism and the cycle-time column "
        "is the outcome, and they are not the same question. Repair can lift "
        "the share of the population that scores at all -- that is close to "
        "arithmetic -- without the extra evaluations landing anywhere useful, "
        "because a layout projected onto the placement boundary is a layout "
        "with the machines pressed up against each other and against the "
        "robot's column, which is where reach and manipulability are hardest "
        "to satisfy. If the share rises and the cycle time does not, that gap "
        "is the finding, not a bug.\n")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def make_figures(output_dir: str, records: Sequence[Dict], pairs: Sequence[Dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = os.path.join(output_dir, "figures")
    os.makedirs(figures, exist_ok=True)
    colours = {"baseline": "#4C72B0", "repair": "#C44E52"}

    def stack(arm: str, key: str) -> np.ndarray:
        traces = [np.asarray([np.nan if v is None else v for v in r[key]], dtype=float)
                  for r in records if r["arm"] == arm]
        width = max(t.size for t in traces)
        padded = np.full((len(traces), width), np.nan)
        for i, trace in enumerate(traces):
            padded[i, :trace.size] = trace
        return padded

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for arm in ARMS:
        best = stack(arm, "best_so_far")
        # All-NaN columns are the generations before any run had found a
        # feasible layout; nanmedian warns rather than failing on them.
        with np.errstate(invalid="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                median = np.nanmedian(best, axis=0)
                low = np.nanpercentile(best, 25, axis=0)
                high = np.nanpercentile(best, 75, axis=0)
        x = np.arange(median.size)
        axes[0].plot(x, median, color=colours[arm], label=arm, lw=1.6)
        axes[0].fill_between(x, low, high, color=colours[arm], alpha=0.15, lw=0)
        axes[1].plot(np.nanmedian(stack(arm, "feasible_fraction_trace"), axis=0),
                     color=colours[arm], label=arm, lw=1.6)

    axes[0].set_xlabel("generation")
    axes[0].set_ylabel("best cycle time (s)")
    axes[0].set_title("Convergence (median, IQR band)")
    axes[1].set_xlabel("generation")
    axes[1].set_ylabel("share of population with finite fitness")
    axes[1].set_title("Feasible share of the population (median)")
    for ax in axes:
        ax.legend(frameon=False)
        ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(figures, "convergence.png"), dpi=150)
    plt.close(fig)

    gains = [p["baseline"]["cycle_time"] - p["repair"]["cycle_time"]
             for p in pairs
             if p["baseline"]["cycle_time"] is not None
             and p["repair"]["cycle_time"] is not None]
    if gains:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.hist(gains, bins=25, color="#55A868", edgecolor="white")
        ax.axvline(0.0, color="0.2", lw=1.2)
        ax.axvline(float(np.mean(gains)), color="#C44E52", lw=1.4, ls="--",
                   label=f"mean {np.mean(gains):+.3f} s")
        ax.set_xlabel("baseline - repair cycle time (s);  positive = repair better")
        ax.set_ylabel("pairs")
        ax.set_title("Paired difference in cycle time")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25, lw=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(figures, "paired_gain.png"), dpi=150)
        plt.close(fig)
    print(f"  figures -> {figures}")


# ----------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------


def _positive_int(name: str):
    def parse(text: str) -> int:
        value = int(text)
        if value <= 0:
            raise argparse.ArgumentTypeError(f"{name} must be positive, got {value}")
        return value
    return parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m arm_study.run_repair_ab",
        description="Paired A/B of the placement repair on the cycle-time objective.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split", default="test", choices=list(ALL_SPLITS))
    parser.add_argument("--instances", type=int, nargs="+", default=None,
                        help="instance indices; default every instance of the split")
    parser.add_argument("--runs", type=_positive_int("--runs"), default=15)
    parser.add_argument("--population-size", type=_positive_int("--population-size"),
                        default=200)
    parser.add_argument("--generations", type=_positive_int("--generations"), default=300)
    parser.add_argument("--crossover-rate", type=float, default=0.9)
    parser.add_argument("--mutation-rate", type=float, default=0.1)
    parser.add_argument("--elitism-rate", type=float, default=0.1)
    parser.add_argument("--mutation-sigma", type=float, default=0.10)
    parser.add_argument("--repair-margin", type=float, default=0.005,
                        help="slack projected past each placement constraint")
    parser.add_argument("--sigma-min", type=float, default=0.14)
    parser.add_argument("--dwell", type=float, default=0.30)
    parser.add_argument("--workers", type=_positive_int("--workers"), default=1)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--resume", action="store_true",
                        help="reuse any run already written under raw/")
    parser.add_argument("--only", nargs="+", choices=list(STAGES), default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stages = tuple(args.only) if args.only else STAGES
    output_dir = args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    instances = build_instances(args.split)
    indices = args.instances if args.instances is not None else list(range(len(instances)))
    for index in indices:
        if not 0 <= index < len(instances):
            raise SystemExit(f"instance index {index} outside 0..{len(instances) - 1}")

    def config_for(arm: str) -> GAConfig:
        return GAConfig(
            label=arm,
            population_size=args.population_size,
            generations=args.generations,
            crossover_rate=args.crossover_rate,
            mutation_rate=args.mutation_rate,
            elitism_rate=args.elitism_rate,
            mutation_sigma=args.mutation_sigma,
            repair=(arm == "repair"),
            repair_margin=args.repair_margin,
            record_diversity=True,
        )

    metadata = {
        "split": args.split,
        "n_instances": len(indices),
        "runs": args.runs,
        "population_size": args.population_size,
        "generations": args.generations,
        "budget": config_for("baseline").evaluation_budget(),
        "arms": {arm: config_for(arm).to_dict() for arm in ARMS},
        "instances": {instances[i].name: describe(instances[i]) for i in indices},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }

    if "run" in stages:
        jobs = [
            Job(arm=arm, split=args.split, instance_index=index, run=run,
                config=config_for(arm), sigma_min_threshold=args.sigma_min,
                dwell=args.dwell, output_dir=output_dir)
            for index in indices
            for run in range(args.runs)
            for arm in ARMS
        ]
        print(f"running {len(jobs)} runs on {args.workers} worker(s)")
        run_grid(jobs, args.workers, args.resume)

    if "aggregate" in stages:
        records = collect_records(output_dir)
        if not records:
            raise SystemExit("no runs found; run the 'run' stage first")
        pairs = pair_up(records)
        print(f"aggregating {len(records)} runs into {len(pairs)} pairs")

        summaries = {
            "cycle_time": compare(pairs, "cycle_time"),
            "mean_feasible": compare(pairs, "mean_feasible", lower_is_better=False),
            "final_feasible": compare(pairs, "final_feasible", lower_is_better=False),
            "evals_to_first": compare(pairs, "evals_to_first"),
            "mean_diversity": compare(pairs, "mean_diversity", lower_is_better=False),
            "cpu_time": compare(pairs, "cpu_time"),
        }
        feasibility = feasibility_table(pairs)
        rows = per_instance(pairs)

        write_csv(os.path.join(output_dir, "runs.csv"), records, RUN_FIELDS)
        write_csv(os.path.join(output_dir, "pairs_summary.csv"),
                  list(summaries.values()),
                  ("metric", "n_pairs", "dropped", "baseline_mean", "repair_mean",
                   "mean_gain", "median_gain", "ci_low", "ci_high", "repair_wins",
                   "baseline_wins", "ties", "permutation_p", "wilcoxon_p"))
        if rows:
            write_csv(os.path.join(output_dir, "per_instance.csv"), rows, tuple(rows[0]))
        with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as h:
            json.dump({**metadata, "feasibility": feasibility}, h, indent=2)
        write_report(os.path.join(output_dir, "summary.md"),
                     summaries, feasibility, rows, metadata)
        print(f"  report  -> {os.path.join(output_dir, 'summary.md')}")

    if "figures" in stages:
        records = collect_records(output_dir)
        if records:
            make_figures(output_dir, records, pair_up(records))


if __name__ == "__main__":
    main()
