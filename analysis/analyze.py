"""
Turn raw runs into summary tables and a report.

Covers per-instance and pooled descriptive statistics (median, mean, sd, IQR,
bootstrap CIs) for solution quality and runtime; paired Wilcoxon signed-rank
tests with Holm correction and effect sizes (rank-biserial, Cohen's dz,
Cliff's delta); budget-matched comparisons (best fitness at a common
evaluation or wall-clock budget, and effort to reach a common quality
target); and per-instance outcomes joined with instance features.

Outputs (written under ``results/analysis``):

===========================  ====================================================
``summary_overall.csv``      one row per variant, pooled over instances
``summary_per_instance.csv`` one row per (variant, instance)
``paired_tests.csv``         every variant compared against each RL-GA reference
``budget_matched.csv``       equal-evaluation / equal-time comparison
``instance_features.csv``    features joined with the RL-GA gain per instance
``report.md``                the same tables rendered for reading
===========================  ====================================================

Run:  python -m analysis.analyze
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from core.problems import build_instances, instance_features
from optimization.run_experiments import RESULTS_DIR, read_rows
from analysis.stats_utils import (
    Summary,
    budget_to_target,
    cliffs_delta,
    describe,
    holm_bonferroni,
    paired_effect_sizes,
    significance_marker,
    value_at_budget,
)

ANALYSIS_DIRNAME = "analysis"

#: Variants used as the reference in the paired comparisons.
DEFAULT_REFERENCES: Tuple[str, ...] = ("rlga_elites", "rlga_offspring")


# ----------------------------------------------------------------------
# organising the raw rows
# ----------------------------------------------------------------------


def index_rows(rows: Sequence[Dict]) -> Dict[Tuple[str, str, int], Dict]:
    """Index runs by (variant, instance, seed)."""
    return {(r["variant"], r["instance"], r["seed"]): r for r in rows}


def variants_in(rows: Sequence[Dict]) -> List[str]:
    seen: List[str] = []
    for row in rows:
        if row["variant"] not in seen:
            seen.append(row["variant"])
    return seen


def instances_in(rows: Sequence[Dict]) -> List[str]:
    seen: List[str] = []
    for row in rows:
        if row["instance"] not in seen:
            seen.append(row["instance"])
    return seen


def label_of(rows: Sequence[Dict], variant: str) -> str:
    for row in rows:
        if row["variant"] == variant:
            return row["label"]
    return variant


def values_of(rows: Sequence[Dict], variant: str, column: str, instance: Optional[str] = None) -> List[float]:
    return [
        row[column]
        for row in rows
        if row["variant"] == variant and (instance is None or row["instance"] == instance)
    ]


def write_csv(path: str, fieldnames: Sequence[str], records: Iterable[Dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for record in records:
            writer.writerow(record)


# ----------------------------------------------------------------------
# descriptive tables
# ----------------------------------------------------------------------


def _summary_record(prefix: str, summary: Summary) -> Dict[str, float]:
    return {
        f"{prefix}_n": summary.n,
        f"{prefix}_median": summary.median,
        f"{prefix}_mean": summary.mean,
        f"{prefix}_std": summary.std,
        f"{prefix}_iqr": summary.iqr,
        f"{prefix}_q1": summary.q1,
        f"{prefix}_q3": summary.q3,
        f"{prefix}_min": summary.minimum,
        f"{prefix}_max": summary.maximum,
        f"{prefix}_mean_ci_low": summary.mean_ci_low,
        f"{prefix}_mean_ci_high": summary.mean_ci_high,
        f"{prefix}_median_ci_low": summary.median_ci_low,
        f"{prefix}_median_ci_high": summary.median_ci_high,
    }


SUMMARY_METRICS = (
    ("fitness", "best_fitness"),
    ("distance", "total_distance"),
    ("time", "execution_time"),
    ("cpu", "cpu_time"),
    ("evals", "evaluations"),
)


def summarise(rows: Sequence[Dict], per_instance: bool) -> List[Dict]:
    """Descriptive statistics per variant, optionally split by instance."""
    records: List[Dict] = []
    groups: List[Tuple[str, Optional[str]]] = []
    for variant in variants_in(rows):
        if per_instance:
            groups.extend((variant, instance) for instance in instances_in(rows))
        else:
            groups.append((variant, None))

    for variant, instance in groups:
        subset = [
            row
            for row in rows
            if row["variant"] == variant and (instance is None or row["instance"] == instance)
        ]
        if not subset:
            continue
        record: Dict = {
            "variant": variant,
            "label": label_of(rows, variant),
            "n_runs": len(subset),
            "feasible_runs": sum(row["feasible"] for row in subset),
        }
        if instance is not None:
            record["instance"] = instance
        for prefix, column in SUMMARY_METRICS:
            record.update(
                _summary_record(prefix, describe([row[column] for row in subset]))
            )
        records.append(record)
    return records


# ----------------------------------------------------------------------
# paired comparisons
# ----------------------------------------------------------------------


def paired_table(
    rows: Sequence[Dict],
    reference: str,
    metric: str = "best_fitness",
    scope: str = "pooled",
) -> List[Dict]:
    """Compare every variant against ``reference`` on matched (instance, seed) pairs.

    ``scope='pooled'`` pairs every run of every instance (n = instances x
    seeds); ``scope='per_instance'`` runs one test per instance (n = seeds).
    The p-values within each family are Holm-corrected.
    """
    indexed = index_rows(rows)
    instances = instances_in(rows)
    seeds = sorted({row["seed"] for row in rows})
    others = [v for v in variants_in(rows) if v != reference]

    def collect(variant: str, subset_instances: Sequence[str]) -> Tuple[List[float], List[float]]:
        a, b = [], []
        for instance in subset_instances:
            for seed in seeds:
                key_ref = (reference, instance, seed)
                key_var = (variant, instance, seed)
                if key_ref in indexed and key_var in indexed:
                    a.append(indexed[key_ref][metric])
                    b.append(indexed[key_var][metric])
        return a, b

    records: List[Dict] = []
    scopes = [("pooled", instances)] if scope == "pooled" else [(i, [i]) for i in instances]

    for scope_name, subset_instances in scopes:
        block: List[Dict] = []
        for variant in others:
            reference_values, variant_values = collect(variant, subset_instances)
            if not reference_values:
                continue
            test = paired_effect_sizes(reference_values, variant_values)
            delta, delta_label = cliffs_delta(reference_values, variant_values)
            ref_median = float(np.median(reference_values))
            var_median = float(np.median(variant_values))
            block.append(
                {
                    "scope": scope_name,
                    "metric": metric,
                    "reference": reference,
                    "variant": variant,
                    "label": label_of(rows, variant),
                    "n_pairs": test.n_pairs,
                    "reference_median": ref_median,
                    "variant_median": var_median,
                    "median_difference": test.median_difference,
                    "mean_difference": test.mean_difference,
                    "diff_ci_low": test.difference_ci_low,
                    "diff_ci_high": test.difference_ci_high,
                    "relative_median_difference_pct": (
                        100.0 * (ref_median - var_median) / var_median if var_median else float("nan")
                    ),
                    "wilcoxon_W": test.statistic,
                    "p_value": test.p_value,
                    "test_method": test.method,
                    "rank_biserial": test.rank_biserial,
                    "cohens_dz": test.cohens_dz,
                    "cliffs_delta": delta,
                    "cliffs_delta_magnitude": delta_label,
                    "reference_wins": test.n_wins,
                    "reference_losses": test.n_losses,
                }
            )

        adjusted = holm_bonferroni([record["p_value"] for record in block])
        for record, p_adj in zip(block, adjusted):
            record["p_value_holm"] = p_adj
            record["significant_holm_0.05"] = int(p_adj < 0.05)
            record["marker"] = significance_marker(p_adj)
        records.extend(block)

    return records


# ----------------------------------------------------------------------
# budget-matched comparison
# ----------------------------------------------------------------------


@lru_cache(maxsize=4096)
def load_trace(results_dir: str, variant: str, instance: str, run: int) -> Optional[Dict[str, np.ndarray]]:
    """Per-generation trace of one run, or ``None`` if it was not stored.

    Cached because the figures re-read the same traces on several axes.
    """
    path = os.path.join(results_dir, "raw", variant, f"{instance}_run{run}.npz")
    if not os.path.exists(path):
        return None
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def budget_matched_table(
    rows: Sequence[Dict],
    results_dir: str,
    reference: str = "rlga_elites",
    quantile: float = 1.0,
) -> List[Dict]:
    """Best fitness at a common evaluation budget and a common time budget.

    For each (instance, seed) the budget is the *smallest* total spend of any
    variant on that pair, so no method is credited with effort another method
    never got.  ``quantile`` scales that budget (1.0 = the full common budget).
    Also reports the effort each variant needs to reach a common quality
    target - the median final fitness of the cheapest variant on that pair.
    """
    variants = variants_in(rows)
    instances = instances_in(rows)
    seeds = sorted({row["seed"] for row in rows})

    traces: Dict[Tuple[str, str, int], Dict[str, np.ndarray]] = {}
    for variant in variants:
        for instance in instances:
            for seed in seeds:
                trace = load_trace(results_dir, variant, instance, seed)
                if trace is not None:
                    traces[(variant, instance, seed)] = trace
    if not traces:
        return []

    per_variant_eval: Dict[str, List[float]] = defaultdict(list)
    per_variant_time: Dict[str, List[float]] = defaultdict(list)
    per_variant_evals_to_target: Dict[str, List[float]] = defaultdict(list)
    per_variant_time_to_target: Dict[str, List[float]] = defaultdict(list)

    for instance in instances:
        for seed in seeds:
            available = [v for v in variants if (v, instance, seed) in traces]
            if len(available) < 2:
                continue

            eval_budget = quantile * min(
                float(traces[(v, instance, seed)]["evaluations"][-1]) for v in available
            )
            time_budget = quantile * min(
                float(traces[(v, instance, seed)]["elapsed"][-1]) for v in available
            )
            # Common quality target: the worst final fitness among the
            # variants, i.e. a level every method can be asked to reach.
            finals = [float(traces[(v, instance, seed)]["best_so_far"][-1]) for v in available]
            finals = [f for f in finals if np.isfinite(f)]
            target = max(finals) if finals else float("inf")

            for variant in available:
                trace = traces[(variant, instance, seed)]
                per_variant_eval[variant].append(
                    value_at_budget(trace["evaluations"], trace["best_so_far"], eval_budget)
                )
                per_variant_time[variant].append(
                    value_at_budget(trace["elapsed"], trace["best_so_far"], time_budget)
                )
                per_variant_evals_to_target[variant].append(
                    budget_to_target(trace["evaluations"], trace["best_so_far"], target)
                )
                per_variant_time_to_target[variant].append(
                    budget_to_target(trace["elapsed"], trace["best_so_far"], target)
                )

    records: List[Dict] = []
    for variant in variants:
        if variant not in per_variant_eval:
            continue
        at_evals = describe(per_variant_eval[variant])
        at_time = describe(per_variant_time[variant])
        evals_to_target = [v for v in per_variant_evals_to_target[variant] if np.isfinite(v)]
        time_to_target = [v for v in per_variant_time_to_target[variant] if np.isfinite(v)]
        records.append(
            {
                "variant": variant,
                "label": label_of(rows, variant),
                "n_pairs": at_evals.n,
                "fitness_at_common_evals_median": at_evals.median,
                "fitness_at_common_evals_mean": at_evals.mean,
                "fitness_at_common_evals_std": at_evals.std,
                "fitness_at_common_evals_ci_low": at_evals.mean_ci_low,
                "fitness_at_common_evals_ci_high": at_evals.mean_ci_high,
                "fitness_at_common_time_median": at_time.median,
                "fitness_at_common_time_mean": at_time.mean,
                "fitness_at_common_time_std": at_time.std,
                "fitness_at_common_time_ci_low": at_time.mean_ci_low,
                "fitness_at_common_time_ci_high": at_time.mean_ci_high,
                "median_evals_to_common_target": float(np.median(evals_to_target)) if evals_to_target else float("nan"),
                "median_seconds_to_common_target": float(np.median(time_to_target)) if time_to_target else float("nan"),
                "target_reached_fraction": len(evals_to_target) / max(len(per_variant_evals_to_target[variant]), 1),
            }
        )

    # Paired tests on the budget-matched quantities, against the reference.
    if reference in per_variant_eval:
        p_values = []
        for record in records:
            variant = record["variant"]
            if variant == reference:
                record["p_at_common_evals"] = float("nan")
                record["p_at_common_time"] = float("nan")
                p_values.append(1.0)
                continue
            test_evals = paired_effect_sizes(
                per_variant_eval[reference], per_variant_eval[variant]
            )
            test_time = paired_effect_sizes(
                per_variant_time[reference], per_variant_time[variant]
            )
            record["p_at_common_evals"] = test_evals.p_value
            record["p_at_common_time"] = test_time.p_value
            record["rank_biserial_at_common_evals"] = test_evals.rank_biserial
            record["rank_biserial_at_common_time"] = test_time.rank_biserial
            p_values.append(test_evals.p_value)
        for record, p_adj in zip(records, holm_bonferroni(p_values)):
            record["p_at_common_evals_holm"] = p_adj

    return records


# ----------------------------------------------------------------------
# instance-level explanation
# ----------------------------------------------------------------------


def instance_feature_table(
    rows: Sequence[Dict], split: str, reference: str, baseline: str
) -> List[Dict]:
    """Join instance features with the reference-vs-baseline gain per instance."""
    instances = {inst.name: inst for inst in build_instances(split)}
    indexed = index_rows(rows)
    seeds = sorted({row["seed"] for row in rows})

    records: List[Dict] = []
    for name in instances_in(rows):
        if name not in instances:
            continue
        reference_values, baseline_values = [], []
        for seed in seeds:
            if (reference, name, seed) in indexed and (baseline, name, seed) in indexed:
                reference_values.append(indexed[(reference, name, seed)]["best_fitness"])
                baseline_values.append(indexed[(baseline, name, seed)]["best_fitness"])
        if not reference_values:
            continue

        test = paired_effect_sizes(reference_values, baseline_values)
        baseline_median = float(np.median(baseline_values))
        record = {
            "instance": name,
            "reference": reference,
            "baseline": baseline,
            "reference_median_fitness": float(np.median(reference_values)),
            "baseline_median_fitness": baseline_median,
            "median_gain": baseline_median - float(np.median(reference_values)),
            "median_gain_pct": (
                100.0 * (baseline_median - float(np.median(reference_values))) / baseline_median
                if baseline_median
                else float("nan")
            ),
            "p_value": test.p_value,
            "rank_biserial": test.rank_biserial,
        }
        record.update(instance_features(instances[name]))
        records.append(record)

    # Correlate each feature with the gain, so the discussion can point at
    # measurable instance characteristics instead of anecdotes.
    if len(records) >= 3:
        gains = np.array([r["median_gain_pct"] for r in records], dtype=float)
        feature_names = [
            key
            for key in records[0]
            if key
            not in {
                "instance",
                "reference",
                "baseline",
                "reference_median_fitness",
                "baseline_median_fitness",
                "median_gain",
                "median_gain_pct",
                "p_value",
                "rank_biserial",
            }
        ]
        correlations = {}
        for feature in feature_names:
            column = np.array([r[feature] for r in records], dtype=float)
            if np.std(column) > 0 and np.std(gains) > 0:
                correlations[feature] = float(np.corrcoef(column, gains)[0, 1])
        for record in records:
            record["_feature_gain_correlations"] = correlations

    return records


# ----------------------------------------------------------------------
# report rendering
# ----------------------------------------------------------------------


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def render_report(
    rows: Sequence[Dict],
    overall: Sequence[Dict],
    per_instance: Sequence[Dict],
    paired: Sequence[Dict],
    budget: Sequence[Dict],
    features: Sequence[Dict],
    references: Sequence[str],
    split: str,
) -> str:
    n_instances = len(instances_in(rows))
    n_seeds = len({row["seed"] for row in rows})

    parts: List[str] = []
    parts.append("# Revised experimental results\n")
    parts.append(
        f"Split `{split}`: {n_instances} instances x {n_seeds} independent seeds per "
        f"instance x {len(variants_in(rows))} algorithm variants = {len(rows)} runs.\n"
        "All variants sharing an (instance, seed) pair start from the same initial "
        "population and the same random seed, so every comparison below is paired.\n"
    )

    # ---- descriptive ---------------------------------------------------
    parts.append("\n## 1. Solution quality and runtime (pooled over instances)\n")
    parts.append(
        "Median with interquartile range is the headline metric; the mean with its "
        "95 % bootstrap confidence interval and the standard deviation are given "
        "alongside. The minimum column is shown for completeness only - it "
        "characterises the luckiest run, not expected performance.\n"
    )
    headers = [
        "variant",
        "fitness median [IQR]",
        "fitness mean ± sd",
        "fitness mean 95% CI",
        "fitness min",
        "time median [IQR] (s)",
        "time mean ± sd (s)",
        "evaluations median",
    ]
    body = []
    for record in sorted(overall, key=lambda r: r["fitness_median"]):
        body.append(
            [
                record["label"],
                f"{record['fitness_median']:.2f} [{record['fitness_q1']:.2f}, {record['fitness_q3']:.2f}]",
                f"{record['fitness_mean']:.2f} ± {record['fitness_std']:.2f}",
                f"[{record['fitness_mean_ci_low']:.2f}, {record['fitness_mean_ci_high']:.2f}]",
                f"{record['fitness_min']:.2f}",
                f"{record['time_median']:.1f} [{record['time_q1']:.1f}, {record['time_q3']:.1f}]",
                f"{record['time_mean']:.1f} ± {record['time_std']:.1f}",
                f"{record['evals_median']:.0f}",
            ]
        )
    parts.append(_markdown_table(headers, body))

    # ---- paired tests --------------------------------------------------
    metric_titles = {
        "best_fitness": ("solution quality", "fitness units"),
        "execution_time": ("runtime", "seconds"),
        "evaluations": ("computational effort", "objective-function evaluations"),
    }
    section = 2
    for reference in references:
        for metric, (metric_title, unit) in metric_titles.items():
            subset = [
                r
                for r in paired
                if r["reference"] == reference and r["scope"] == "pooled" and r["metric"] == metric
            ]
            if not subset:
                continue
            parts.append(
                f"\n## {section}. Paired comparison against `{reference}` - {metric_title}\n"
            )
            parts.append(
                f"Difference = reference - variant on matched seeds, in {unit}, so a "
                "**negative** difference means the reference is lower "
                f"({'better' if metric == 'best_fitness' else 'cheaper'}). "
                "Two-sided Wilcoxon signed-rank test, Holm-corrected within this family; "
                "effect sizes are the matched-pairs rank-biserial correlation and "
                "Cliff's delta.\n"
            )
            headers = [
                "variant",
                "median diff",
                "mean diff [95% CI]",
                "reference lower / higher",
                "p (Holm)",
                "rank-biserial",
                "Cliff's delta",
            ]
            body = []
            for record in sorted(subset, key=lambda r: r["median_difference"]):
                body.append(
                    [
                        record["label"],
                        f"{record['median_difference']:+.3f}",
                        f"{record['mean_difference']:+.3f} [{record['diff_ci_low']:+.3f}, {record['diff_ci_high']:+.3f}]",
                        f"{record['reference_wins']}/{record['reference_losses']}",
                        f"{record['p_value_holm']:.4f} {record['marker']}",
                        f"{record['rank_biserial']:+.3f}",
                        f"{record['cliffs_delta']:+.3f} ({record['cliffs_delta_magnitude']})",
                    ]
                )
            parts.append(_markdown_table(headers, body))
            section += 1

    # ---- budget matched ------------------------------------------------
    if budget:
        parts.append(f"\n## {section}. Equal-budget comparison\n")
        section += 1
        parts.append(
            "Equal generation counts are not an equal computational budget: variants "
            "with a higher local-search rate spend several times more objective-function "
            "evaluations per generation. Each (instance, seed) pair is therefore "
            "truncated at the largest budget every variant actually reached, and the "
            "best-so-far fitness at that point is compared. The last two columns give "
            "the effort needed to reach a quality level all variants attain.\n"
        )
        headers = [
            "variant",
            "fitness @ common evaluations (median)",
            "fitness @ common wall-clock (median)",
            "evaluations to common target (median)",
            "seconds to common target (median)",
        ]
        body = []
        for record in sorted(budget, key=lambda r: r["fitness_at_common_evals_median"]):
            body.append(
                [
                    record["label"],
                    f"{record['fitness_at_common_evals_median']:.2f}",
                    f"{record['fitness_at_common_time_median']:.2f}",
                    f"{record['median_evals_to_common_target']:.0f}",
                    f"{record['median_seconds_to_common_target']:.1f}",
                ]
            )
        parts.append(_markdown_table(headers, body))

    # ---- per-instance --------------------------------------------------
    parts.append(f"\n## {section}. Per-instance medians (fitness)\n")
    section += 1
    instances = instances_in(rows)
    variants = variants_in(rows)
    headers = ["variant"] + instances
    body = []
    lookup = {(r["variant"], r["instance"]): r for r in per_instance}
    for variant in variants:
        row_cells = [label_of(rows, variant)]
        for instance in instances:
            record = lookup.get((variant, instance))
            row_cells.append(f"{record['fitness_median']:.2f}" if record else "-")
        body.append(row_cells)
    parts.append(_markdown_table(headers, body))

    # ---- per-instance minima -------------------------------------------
    parts.append(f"\n## {section}. Per-instance best fitness (minimum over seeds)\n")
    section += 1
    parts.append(
        "The best layout any seed found for each instance. This is a best-of-n "
        "statistic, so it rewards a lucky run and is not an estimate of expected "
        "performance - the medians above are. It is reported because a practitioner "
        "who can afford several restarts and keep the best layout cares about this "
        "column, and because it shows which variants can reach a good solution at "
        "all, as opposed to reaching one reliably.\n"
    )
    headers = ["variant"] + instances
    body = []
    for variant in variants:
        row_cells = [label_of(rows, variant)]
        for instance in instances:
            record = lookup.get((variant, instance))
            value = record["fitness_min"] if record else float("nan")
            row_cells.append(f"{value:.2f}" if np.isfinite(value) else "-")
        body.append(row_cells)
    parts.append(_markdown_table(headers, body))

    # Summary of the table above.  Everything here is derived from the same
    # numbers, so it cannot drift away from the table when the runs change.
    min_wins: Dict[str, List[str]] = defaultdict(list)
    median_wins: Dict[str, List[str]] = defaultdict(list)
    spreads: List[float] = []
    overall_best = float("inf")
    overall_best_where: List[Tuple[str, str]] = []

    for instance in instances:
        for key, wins in (("fitness_min", min_wins), ("fitness_median", median_wins)):
            cells = [
                (float(lookup[(v, instance)][key]), v)
                for v in variants
                if (v, instance) in lookup and np.isfinite(lookup[(v, instance)][key])
            ]
            if not cells:
                continue
            best = min(value for value, _ in cells)
            for value, variant in cells:
                if value == best:
                    wins[variant].append(instance)
            if key != "fitness_min":
                continue
            worst = max(value for value, _ in cells)
            if best > 0:
                spreads.append(100.0 * (worst - best) / best)
            if best < overall_best:
                overall_best = best
                overall_best_where = [(v, instance) for value, v in cells if value == best]

    if overall_best_where:
        holders = ", ".join(
            f"{label_of(rows, variant)} on {instance}"
            for variant, instance in overall_best_where
        )
        leaders = sorted(min_wins, key=lambda v: (-len(min_wins[v]), v))
        top = len(min_wins[leaders[0]])
        tied = [v for v in leaders if len(min_wins[v]) == top]
        leader_text = ", ".join(label_of(rows, v) for v in tied)
        verb = "share the lead with" if len(tied) > 1 else "leads with"
        summary = (
            f"\nThe best layout found anywhere in the experiment scores "
            f"{overall_best:.2f} ({holders}). Counting instances by which variant "
            f"reaches the lowest minimum, {leader_text} {verb} {top} of "
            f"{len(instances)}. "
        )
        if spreads:
            summary += (
                f"Variant choice is not incidental to this column: on the median "
                f"instance the gap between the best and the worst variant minimum "
                f"is {float(np.median(spreads)):.1f} % "
                f"(range {min(spreads):.1f} - {max(spreads):.1f} %). "
            )
        crossover = [v for v in tied if not median_wins.get(v)]
        if crossover:
            summary += (
                f"The ordering is not the ordering by median: "
                f"{', '.join(label_of(rows, v) for v in crossover)} "
                f"{'lead' if len(crossover) > 1 else 'leads'} on best-of-n without "
                f"winning a single instance on the median. "
            )
        summary += (
            "That is the expected behaviour of a best-of-n statistic - it rewards "
            "the variant with the widest spread across seeds, not the most "
            "dependable one - and it is why the two tables should be read together "
            "rather than either one alone.\n"
        )
        parts.append(summary)

        headers = ["variant", "instances won on minimum", "instances won on median"]
        body = []
        for variant in sorted(variants, key=lambda v: (-len(min_wins[v]), v)):
            if not min_wins[variant] and not median_wins[variant]:
                continue
            body.append(
                [
                    label_of(rows, variant),
                    f"{len(min_wins[variant])} ({', '.join(min_wins[variant])})"
                    if min_wins[variant]
                    else "0",
                    f"{len(median_wins[variant])} ({', '.join(median_wins[variant])})"
                    if median_wins[variant]
                    else "0",
                ]
            )
        parts.append(_markdown_table(headers, body))

    # ---- instance features ---------------------------------------------
    if features:
        parts.append(f"\n## {section}. Where the RL-GA helps, and where it does not\n")
        correlations = features[0].get("_feature_gain_correlations", {})
        headers = ["instance", "median gain (%)", "p", "occupancy", "feasibility rate", "mean aspect ratio"]
        body = []
        for record in sorted(features, key=lambda r: -r["median_gain_pct"]):
            body.append(
                [
                    record["instance"],
                    f"{record['median_gain_pct']:+.2f}",
                    f"{record['p_value']:.3f}",
                    f"{record['occupancy']:.3f}",
                    f"{record['random_feasibility_rate']:.4f}",
                    f"{record['mean_aspect_ratio']:.2f}",
                ]
            )
        parts.append(_markdown_table(headers, body))
        if correlations:
            ranked = sorted(correlations.items(), key=lambda kv: -abs(kv[1]))[:8]
            parts.append(
                "\nPearson correlation between instance features and the relative gain:\n"
            )
            parts.append(
                _markdown_table(
                    ["feature", "correlation with gain (%)"],
                    [[name, f"{value:+.3f}"] for name, value in ranked],
                )
            )

    return "\n".join(parts) + "\n"


# ----------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------


def analyse(
    results_dir: str = RESULTS_DIR,
    csv_name: str = "runs.csv",
    split: str = "test",
    references: Sequence[str] = DEFAULT_REFERENCES,
    baseline: str = "ga_ls10_offspring",
    output_subdir: str = ANALYSIS_DIRNAME,
) -> Dict[str, List[Dict]]:
    rows = read_rows(os.path.join(results_dir, csv_name))
    if not rows:
        raise SystemExit(f"no runs found in {os.path.join(results_dir, csv_name)}")

    available = variants_in(rows)
    references = [r for r in references if r in available]
    if not references:
        references = [available[0]]
    if baseline not in available:
        baseline = available[0]

    out_dir = os.path.join(results_dir, output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    overall = summarise(rows, per_instance=False)
    per_instance = summarise(rows, per_instance=True)

    paired: List[Dict] = []
    for reference in references:
        for metric in ("best_fitness", "execution_time", "evaluations"):
            paired.extend(paired_table(rows, reference, metric=metric, scope="pooled"))
        paired.extend(paired_table(rows, reference, metric="best_fitness", scope="per_instance"))

    budget = budget_matched_table(rows, results_dir, reference=references[0])
    features = instance_feature_table(rows, split, references[0], baseline)

    write_csv(os.path.join(out_dir, "summary_overall.csv"), list(overall[0]), overall)
    write_csv(os.path.join(out_dir, "summary_per_instance.csv"), list(per_instance[0]), per_instance)
    if paired:
        write_csv(os.path.join(out_dir, "paired_tests.csv"), list(paired[0]), paired)
    if budget:
        fieldnames = sorted({key for record in budget for key in record})
        write_csv(
            os.path.join(out_dir, "budget_matched.csv"),
            fieldnames,
            [{key: record.get(key, "") for key in fieldnames} for record in budget],
        )
    if features:
        cleaned = [{k: v for k, v in r.items() if not k.startswith("_")} for r in features]
        write_csv(os.path.join(out_dir, "instance_features.csv"), list(cleaned[0]), cleaned)
        with open(os.path.join(out_dir, "feature_gain_correlations.json"), "w", encoding="utf-8") as fh:
            json.dump(features[0].get("_feature_gain_correlations", {}), fh, indent=2)

    report = render_report(rows, overall, per_instance, paired, budget, features, references, split)
    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    print(f"Analysis written to {out_dir}")
    print(f"  report: {report_path}")
    return {
        "overall": overall,
        "per_instance": per_instance,
        "paired": paired,
        "budget": budget,
        "features": features,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--csv-name", default="runs.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--references", nargs="+", default=list(DEFAULT_REFERENCES))
    parser.add_argument("--baseline", default="ga_ls10_offspring")
    parser.add_argument("--output-subdir", default=ANALYSIS_DIRNAME)
    args = parser.parse_args()

    analyse(
        results_dir=args.results_dir,
        csv_name=args.csv_name,
        split=args.split,
        references=args.references,
        baseline=args.baseline,
        output_subdir=args.output_subdir,
    )


if __name__ == "__main__":
    main()
