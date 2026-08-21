"""
Figures for the study: convergence, distributions, RL behaviour and
instance-feature plots. Everything uses matplotlib only.

Covers:
* convergence plotted against generations, cumulative evaluations and
  cumulative wall-clock time
* per-instance distributions - box/violin plots, paired fitness-difference
  plots, and a runtime-versus-quality scatter with the Pareto front
* RL behaviour aggregated over runs and instances - action frequencies,
  parameter trajectories, state occupancy, and the learned Q-table as a heat
  map
* instance features plotted against the RL-GA gain
* training reward per episode

Run:  python -m visualization.plots
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.analyze import (
    instances_in,
    label_of,
    load_trace,
    variants_in,
)
from optimization.run_experiments import RESULTS_DIR, read_rows
from analysis.stats_utils import describe, value_at_budget

FIGURE_DIRNAME = "figures"
DPI = 200

#: Stable colour/style per variant family so figures read consistently.
_COLOURS = plt.cm.tab10(np.linspace(0, 1, 10))


def variant_style(variant: str, index: int) -> Dict:
    """Colour and line style: RL-GA solid and heavy, baselines lighter."""
    is_rl = variant.startswith("rlga")
    is_schedule = "sched" in variant
    return {
        "color": _COLOURS[index % len(_COLOURS)],
        "linewidth": 2.4 if is_rl else 1.5,
        "linestyle": "-" if is_rl else ("--" if is_schedule else ":"),
        "zorder": 3 if is_rl else 2,
    }


def _finalise(fig, save_path: Optional[str]) -> None:
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# convergence
# ----------------------------------------------------------------------


def _aligned_curves(
    traces: Sequence[Dict[str, np.ndarray]],
    axis_key: str,
    grid: np.ndarray,
) -> np.ndarray:
    """Resample every run's best-so-far curve onto a common budget grid."""
    curves = np.full((len(traces), grid.size), np.nan)
    for i, trace in enumerate(traces):
        axis = np.asarray(trace[axis_key], dtype=float)
        values = np.asarray(trace["best_so_far"], dtype=float)
        for j, budget in enumerate(grid):
            curves[i, j] = value_at_budget(axis, values, budget)
    return curves


def plot_convergence(
    rows: Sequence[Dict],
    results_dir: str,
    save_path: Optional[str] = None,
    instance: Optional[str] = None,
    variants: Optional[Sequence[str]] = None,
    band: str = "iqr",
) -> None:
    """Median best-so-far against generations, evaluations and wall-clock time.

    The shaded band is the interquartile range across the independent seeds
    (all instances pooled when ``instance`` is None, after normalising each
    instance so that instances with different fitness scales contribute
    equally).
    """
    variants = list(variants or variants_in(rows))
    instances = [instance] if instance else instances_in(rows)
    seeds = sorted({row["seed"] for row in rows})

    # Each entry is (instance_name, trace); keeping the instance alongside the
    # trace is what makes the per-instance normalisation below unambiguous.
    collected: Dict[str, List[Tuple[str, Dict[str, np.ndarray]]]] = defaultdict(list)
    for variant in variants:
        for inst in instances:
            for seed in seeds:
                trace = load_trace(results_dir, variant, inst, seed)
                if trace is not None:
                    collected[variant].append((inst, trace))
    if not collected:
        return

    pooled = instance is None and len(instances) > 1
    # When pooling across instances, express fitness relative to the best value
    # any method reached on that instance, so instance scale does not dominate.
    per_instance_reference: Dict[str, float] = {}
    if pooled:
        for inst in instances:
            values = [
                float(np.nanmin(trace["best_so_far"]))
                for entries in collected.values()
                for name, trace in entries
                if name == inst
            ]
            finite = [v for v in values if np.isfinite(v)]
            per_instance_reference[inst] = min(finite) if finite else 1.0

    axes_specs = (
        ("generation", "Generation", None),
        ("evaluations", "Cumulative objective-function evaluations", None),
        ("elapsed", "Cumulative wall-clock time (s)", None),
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for ax, (axis_key, xlabel, _) in zip(axes, axes_specs):
        for index, variant in enumerate(variants):
            entries = collected.get(variant, [])
            if not entries:
                continue
            names = [name for name, _ in entries]
            traces = [trace for _, trace in entries]

            if axis_key == "generation":
                length = min(t["best_so_far"].size for t in traces)
                grid = np.arange(length, dtype=float)
                curves = np.vstack([t["best_so_far"][:length] for t in traces])
            else:
                upper = np.median([t[axis_key][-1] for t in traces])
                grid = np.linspace(
                    max(float(np.min([t[axis_key][0] for t in traces])), 1e-6), upper, 120
                )
                curves = _aligned_curves(traces, axis_key, grid)

            if pooled:
                scale = np.array([per_instance_reference[name] for name in names], dtype=float)
                curves = curves / scale[:, None]

            with np.errstate(invalid="ignore"):
                median = np.nanmedian(np.where(np.isfinite(curves), curves, np.nan), axis=0)
                q1 = np.nanpercentile(np.where(np.isfinite(curves), curves, np.nan), 25, axis=0)
                q3 = np.nanpercentile(np.where(np.isfinite(curves), curves, np.nan), 75, axis=0)

            style = variant_style(variant, index)
            ax.plot(grid, median, label=label_of(rows, variant), **style)
            if band == "iqr":
                ax.fill_between(grid, q1, q3, color=style["color"], alpha=0.12, linewidth=0)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(
            "Best fitness / best found on instance" if pooled else "Best fitness so far"
        )
        ax.grid(alpha=0.3)
        if axis_key != "generation":
            ax.set_xscale("log")

    title = f"Convergence on {instance}" if instance else "Convergence (median over runs and instances)"
    fig.suptitle(title + " - lower is better; shaded band = IQR over runs", y=1.02)
    axes[0].legend(fontsize=8, loc="upper right")
    _finalise(fig, save_path)


# ----------------------------------------------------------------------
# distributions
# ----------------------------------------------------------------------


def plot_per_instance_distributions(
    rows: Sequence[Dict],
    save_path: Optional[str] = None,
    metric: str = "best_fitness",
    variants: Optional[Sequence[str]] = None,
) -> None:
    """Box plots of the per-run outcome for every instance and variant."""
    variants = list(variants or variants_in(rows))
    instances = instances_in(rows)

    n_cols = min(5, len(instances))
    n_rows = int(np.ceil(len(instances) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.4 * n_rows), squeeze=False)

    for k, instance in enumerate(instances):
        ax = axes[k // n_cols][k % n_cols]
        data, labels, colours = [], [], []
        for index, variant in enumerate(variants):
            values = [
                row[metric]
                for row in rows
                if row["variant"] == variant and row["instance"] == instance and np.isfinite(row[metric])
            ]
            if values:
                data.append(values)
                labels.append(variant.replace("ga_", "").replace("rlga", "RL"))
                colours.append(variant_style(variant, index)["color"])

        if data:
            parts = ax.boxplot(data, patch_artist=True, showmeans=True, widths=0.6)
            for patch, colour in zip(parts["boxes"], colours):
                patch.set_facecolor(colour)
                patch.set_alpha(0.45)
            for median in parts["medians"]:
                median.set_color("black")
            ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_title(instance, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        if k % n_cols == 0:
            ax.set_ylabel("Best fitness" if metric == "best_fitness" else metric)

    for k in range(len(instances), n_rows * n_cols):
        axes[k // n_cols][k % n_cols].axis("off")

    fig.suptitle("Per-instance distribution over independent runs (lower is better)", y=1.0)
    _finalise(fig, save_path)


def plot_paired_differences(
    rows: Sequence[Dict],
    reference: str,
    save_path: Optional[str] = None,
    metric: str = "best_fitness",
    variants: Optional[Sequence[str]] = None,
) -> None:
    """Per-seed paired differences (reference - variant), with the mean and its CI."""
    # Colours are keyed to the variant's position in the *unfiltered* registry,
    # not to its position after the reference is dropped.  Otherwise every
    # variant after the reference shifts colour from one reference's figure to
    # the next, and the figures cannot be compared side by side.
    colour_index = {v: i for i, v in enumerate(variants or variants_in(rows))}
    variants = [v for v in (variants or variants_in(rows)) if v != reference]
    indexed = {(r["variant"], r["instance"], r["seed"]): r for r in rows}
    instances = instances_in(rows)
    seeds = sorted({row["seed"] for row in rows})

    fig, ax = plt.subplots(figsize=(1.5 + 1.5 * len(variants), 5.5))
    positions = np.arange(len(variants))

    for index, variant in enumerate(variants):
        differences = []
        for instance in instances:
            for seed in seeds:
                a = indexed.get((reference, instance, seed))
                b = indexed.get((variant, instance, seed))
                if a and b and np.isfinite(a[metric]) and np.isfinite(b[metric]):
                    differences.append(a[metric] - b[metric])
        if not differences:
            continue

        jitter = (np.random.default_rng(index).random(len(differences)) - 0.5) * 0.28
        ax.scatter(
            positions[index] + jitter,
            differences,
            s=12,
            alpha=0.45,
            color=variant_style(variant, colour_index.get(variant, index))["color"],
            edgecolors="none",
        )
        summary = describe(differences)
        ax.errorbar(
            positions[index],
            summary.mean,
            yerr=[[summary.mean - summary.mean_ci_low], [summary.mean_ci_high - summary.mean]],
            fmt="o",
            color="black",
            capsize=5,
            markersize=6,
            zorder=5,
        )
        ax.scatter(positions[index], summary.median, marker="_", s=380, color="crimson", zorder=6)

    ax.axhline(0.0, color="grey", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels([label_of(rows, v) for v in variants], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(f"{reference} - variant  (negative = {reference} better)")
    ax.set_title(
        f"Paired per-seed differences against {reference}\n"
        "black = mean with 95% bootstrap CI, red dash = median",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.3)
    _finalise(fig, save_path)


def plot_quality_runtime_pareto(
    rows: Sequence[Dict],
    save_path: Optional[str] = None,
    variants: Optional[Sequence[str]] = None,
    time_column: str = "execution_time",
) -> None:
    """Median runtime versus median quality, with the Pareto front highlighted."""
    variants = list(variants or variants_in(rows))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, (x_column, x_label) in zip(
        axes, ((time_column, "Median wall-clock time per run (s)"), ("evaluations", "Median objective-function evaluations"))
    ):
        points = []
        for index, variant in enumerate(variants):
            subset = [row for row in rows if row["variant"] == variant]
            quality = [row["best_fitness"] for row in subset if np.isfinite(row["best_fitness"])]
            effort = [row[x_column] for row in subset]
            if not quality or not effort:
                continue
            quality_summary = describe(quality)
            x = float(np.median(effort))
            y = quality_summary.median
            points.append((x, y, variant, index))

            ax.errorbar(
                x,
                y,
                yerr=[
                    [y - quality_summary.median_ci_low],
                    [quality_summary.median_ci_high - y],
                ],
                fmt="o",
                markersize=9 if variant.startswith("rlga") else 6,
                color=variant_style(variant, index)["color"],
                capsize=3,
                label=label_of(rows, variant),
            )
            ax.annotate(
                variant.replace("ga_", "").replace("rlga", "RL"),
                (x, y),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7,
            )

        # Pareto front: minimise both effort and fitness.
        front = []
        for x, y, variant, _ in sorted(points):
            if not front or y < front[-1][1]:
                front.append((x, y, variant))
        if len(front) > 1:
            ax.plot(
                [p[0] for p in front],
                [p[1] for p in front],
                color="black",
                linewidth=1.2,
                linestyle="--",
                alpha=0.7,
                zorder=1,
                label="Pareto front",
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel("Median best fitness (95 % CI)")
        ax.set_xscale("log")
        ax.grid(alpha=0.3)

    axes[1].legend(fontsize=7, loc="best")
    fig.suptitle("Quality versus computational effort (lower-left is better)", y=1.0)
    _finalise(fig, save_path)


# ----------------------------------------------------------------------
# RL behaviour
# ----------------------------------------------------------------------


def plot_rl_behaviour(
    rows: Sequence[Dict],
    results_dir: str,
    variant: str,
    save_path: Optional[str] = None,
    action_labels: Optional[Sequence[str]] = None,
) -> None:
    """Aggregate the learned control behaviour over every run and instance."""
    instances = instances_in(rows)
    seeds = sorted({row["seed"] for row in rows})

    traces = [
        trace
        for instance in instances
        for seed in seeds
        if (trace := load_trace(results_dir, variant, instance, seed)) is not None
    ]
    if not traces:
        return

    # Prefer the Q-table width: an action that was never selected still belongs
    # on the axis, and dropping it would silently misalign the labels.
    q_widths = [t["q_table"].shape[1] for t in traces if t.get("q_table") is not None and t["q_table"].ndim == 2]
    n_actions = (
        int(q_widths[0])
        if q_widths
        else int(max(int(np.max(t["rl_action"])) for t in traces) + 1)
    )
    # Only use supplied labels if they actually match this agent's action set:
    # different studies have different action spaces (7 increment actions in the
    # first design, 5 or 15 set-points in the second), and silently truncating
    # would mislabel every bar.
    if not action_labels or len(action_labels) != n_actions:
        action_labels = [f"a{i}" for i in range(n_actions)]
    action_labels = list(action_labels)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # (a) action frequency, aggregated over all runs
    ax = axes[0][0]
    counts = np.zeros(n_actions)
    for trace in traces:
        actions = trace["rl_action"]
        for action in actions[actions >= 0]:
            counts[int(action)] += 1
    frequencies = counts / max(counts.sum(), 1)
    ax.bar(np.arange(n_actions), frequencies, color=_COLOURS[: n_actions])
    ax.set_xticks(np.arange(n_actions))
    ax.set_xticklabels(action_labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of control decisions")
    ax.set_title(f"(a) Action frequency over {len(traces)} runs", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # (b) state occupancy
    ax = axes[0][1]
    states = np.concatenate([t["rl_state"][t["rl_state"] >= 0] for t in traces])
    n_states = int(states.max()) + 1 if states.size else 9
    occupancy = np.bincount(states, minlength=n_states) / max(states.size, 1)
    ax.bar(np.arange(n_states), occupancy, color="steelblue")
    ax.set_xticks(np.arange(n_states))
    ax.set_xlabel("State index (3 x diversity bin + improvement bin)")
    ax.set_ylabel("Fraction of generations")
    ax.set_title("(b) State occupancy", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # (c) controlled-parameter trajectories, median with IQR band
    ax = axes[1][0]
    length = min(t["ls_rate"].size for t in traces)
    generations = np.arange(length)
    for key, colour, name in (
        ("ls_rate", "tab:red", "local-search rate"),
        ("mutation_rate", "tab:blue", "mutation rate"),
        ("crossover_rate", "tab:green", "crossover rate"),
    ):
        stacked = np.vstack([t[key][:length] for t in traces])
        median = np.nanmedian(stacked, axis=0)
        q1 = np.nanpercentile(stacked, 25, axis=0)
        q3 = np.nanpercentile(stacked, 75, axis=0)
        ax.plot(generations, median, color=colour, label=name, linewidth=1.8)
        ax.fill_between(generations, q1, q3, color=colour, alpha=0.15, linewidth=0)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Parameter value")
    ax.set_title("(c) Controlled parameters (median and IQR over runs)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (d) learned Q-table (averaged over the runs, which all start from a
    #     trained table and keep adapting online)
    ax = axes[1][1]
    q_tables = [t["q_table"] for t in traces if t.get("q_table") is not None and t["q_table"].size]
    if q_tables:
        mean_q = np.mean(np.stack(q_tables), axis=0)
        image = ax.imshow(mean_q, cmap="RdYlGn", aspect="auto")
        fig.colorbar(image, ax=ax, label="Q value")
        ax.set_xticks(np.arange(mean_q.shape[1]))
        ax.set_xticklabels(action_labels, rotation=35, ha="right", fontsize=7)
        ax.set_yticks(np.arange(mean_q.shape[0]))
        ax.set_ylabel("State")
        for i in range(mean_q.shape[0]):
            best = int(np.argmax(mean_q[i]))
            ax.add_patch(plt.Rectangle((best - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.6))
        ax.set_title("(d) Mean Q-table (boxed = greedy action)", fontsize=10)

    fig.suptitle(f"Learned control behaviour: {variant}", y=1.0)
    _finalise(fig, save_path)


def plot_training_summary(
    agent_paths: Sequence[str], save_path: Optional[str] = None, title: str = ""
) -> None:
    """Reward against training episode for independently trained agents,
    showing whether the agent has actually converged.
    """
    histories = []
    for path in agent_paths:
        if os.path.exists(path):
            with np.load(path, allow_pickle=True) as data:
                histories.append({key: data[key] for key in data.files})
    if not histories:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    def rolling(values: np.ndarray, window: int = 15) -> np.ndarray:
        if values.size < window:
            return values
        kernel = np.ones(window) / window
        return np.convolve(values, kernel, mode="valid")

    ax = axes[0][0]
    for k, history in enumerate(histories):
        rewards = history["episode_reward_mean"]
        ax.plot(rewards, alpha=0.25, color=_COLOURS[k], linewidth=0.8)
        smoothed = rolling(rewards)
        ax.plot(
            np.arange(smoothed.size) + (rewards.size - smoothed.size) / 2,
            smoothed,
            color=_COLOURS[k],
            linewidth=2.0,
            label=f"seed {int(history['agent_seed'])}",
        )
    ax.set_xlabel("Training episode (one GA run)")
    ax.set_ylabel("Mean reward per control decision")
    ax.set_title("(a) Reward against training episode", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    for k, history in enumerate(histories):
        fitness = history["episode_best_fitness"]
        ax.plot(fitness, alpha=0.25, color=_COLOURS[k], linewidth=0.8)
        smoothed = rolling(fitness)
        ax.plot(
            np.arange(smoothed.size) + (fitness.size - smoothed.size) / 2,
            smoothed,
            color=_COLOURS[k],
            linewidth=2.0,
        )
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Best fitness reached")
    ax.set_title("(b) Training-run quality (instances are interleaved)", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    for k, history in enumerate(histories):
        ax.plot(history["episode_q_delta"], color=_COLOURS[k], alpha=0.7, linewidth=1.0)
    ax.set_xlabel("Training episode")
    ax.set_ylabel(r"$\|\Delta Q\|_F$ per episode")
    ax.set_title("(c) Q-table movement (convergence indicator)", fontsize=10)
    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    labels = [str(name) for name in histories[0]["action_names"]]
    width = 0.8 / len(histories)
    for k, history in enumerate(histories):
        counts = history["action_counts"].sum(axis=0)
        frequencies = counts / max(counts.sum(), 1)
        ax.bar(
            np.arange(len(labels)) + k * width,
            frequencies,
            width=width,
            color=_COLOURS[k],
            label=f"seed {int(history['agent_seed'])}",
        )
    ax.set_xticks(np.arange(len(labels)) + 0.4 - width / 2)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of training decisions")
    ax.set_title("(d) Action usage during training", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Agent training summary{': ' + title if title else ''}", y=1.0)
    _finalise(fig, save_path)


# ----------------------------------------------------------------------
# instance features
# ----------------------------------------------------------------------


def plot_feature_gain(
    features: Sequence[Dict], save_path: Optional[str] = None, top_k: int = 4
) -> None:
    """Scatter the strongest instance-feature correlates of the RL-GA gain."""
    if not features:
        return
    correlations = features[0].get("_feature_gain_correlations", {})
    if not correlations:
        return

    ranked = sorted(correlations.items(), key=lambda kv: -abs(kv[1]))[:top_k]
    gains = np.array([record["median_gain_pct"] for record in features], dtype=float)

    fig, axes = plt.subplots(1, len(ranked), figsize=(4.2 * len(ranked), 4.0), squeeze=False)
    for ax, (feature, correlation) in zip(axes[0], ranked):
        values = np.array([record[feature] for record in features], dtype=float)
        ax.scatter(values, gains, s=45, color="tab:blue", alpha=0.8)
        for record, x, y in zip(features, values, gains):
            ax.annotate(
                record["instance"].replace("test_", "P"),
                (x, y),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=7,
            )
        if np.std(values) > 0:
            slope, intercept = np.polyfit(values, gains, 1)
            grid = np.linspace(values.min(), values.max(), 50)
            ax.plot(grid, slope * grid + intercept, color="crimson", linewidth=1.2)
        ax.axhline(0.0, color="grey", linewidth=0.8)
        ax.set_xlabel(feature)
        ax.set_ylabel("Median gain over baseline (%)")
        ax.set_title(f"r = {correlation:+.2f}", fontsize=10)
        ax.grid(alpha=0.3)

    fig.suptitle("Instance characteristics against the RL-GA gain", y=1.02)
    _finalise(fig, save_path)


# ----------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------


def make_all_figures(
    results_dir: str = RESULTS_DIR,
    csv_name: str = "runs.csv",
    split: str = "test",
    reference: str = "rlga_elites",
    baseline: str = "ga_ls10_offspring",
    output_subdir: str = FIGURE_DIRNAME,
    per_instance_convergence: bool = True,
) -> None:
    from analysis.analyze import instance_feature_table
    from optimization.rl_control import BASE_ACTIONS

    rows = read_rows(os.path.join(results_dir, csv_name))
    if not rows:
        raise SystemExit(f"no runs found in {os.path.join(results_dir, csv_name)}")
    out_dir = os.path.join(results_dir, output_subdir)
    os.makedirs(out_dir, exist_ok=True)
    variants = variants_in(rows)
    reference = reference if reference in variants else variants[0]
    baseline = baseline if baseline in variants else variants[0]

    print("  convergence (generations / evaluations / wall-clock)")
    plot_convergence(rows, results_dir, os.path.join(out_dir, "convergence_pooled.png"))

    # Convergence within each local-search target, so the comparison between
    # controllers on the same target is legible.
    for target in ("offspring", "elites"):
        subset = [v for v in variants if v.endswith(target) or v == "ga_ls00"]
        if len(subset) > 1:
            plot_convergence(
                rows,
                results_dir,
                os.path.join(out_dir, f"convergence_{target}.png"),
                variants=subset,
            )

    if per_instance_convergence:
        for instance in instances_in(rows):
            plot_convergence(
                rows,
                results_dir,
                os.path.join(out_dir, "per_instance", f"convergence_{instance}.png"),
                instance=instance,
            )

    print("  distributions and paired differences")
    plot_per_instance_distributions(rows, os.path.join(out_dir, "per_instance_boxplots.png"))

    # One paired-difference figure per main RL controller, not just the single
    # ``reference``: the two local-search targets are different controllers and
    # each needs its own comparison against the static baselines.  Ablation and
    # control-interval variants (rlga_lsonly_*, rlga_*_ci5, ...) are excluded,
    # since one figure each would bury the two that matter.
    paired_references = [
        key for key in ("rlga_elites", "rlga_offspring") if key in variants
    ]
    if reference not in paired_references:
        paired_references.insert(0, reference)
    for key in paired_references:
        plot_paired_differences(
            rows, key, os.path.join(out_dir, f"paired_differences_{key}.png")
        )

    print("  quality versus effort")
    plot_quality_runtime_pareto(rows, os.path.join(out_dir, "quality_vs_effort.png"))

    print("  RL behaviour")
    for variant in variants:
        if variant.startswith("rlga"):
            plot_rl_behaviour(
                rows,
                results_dir,
                variant,
                os.path.join(out_dir, f"rl_behaviour_{variant}.png"),
                action_labels=BASE_ACTIONS,
            )

    print("  instance features")
    features = instance_feature_table(rows, split, reference, baseline)
    plot_feature_gain(features, os.path.join(out_dir, "feature_gain.png"))

    print(f"Figures written to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--csv-name", default="runs.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--reference", default="rlga_elites")
    parser.add_argument("--baseline", default="ga_ls10_offspring")
    parser.add_argument("--output-subdir", default=FIGURE_DIRNAME)
    parser.add_argument("--no-per-instance", action="store_true")
    args = parser.parse_args()

    make_all_figures(
        results_dir=args.results_dir,
        csv_name=args.csv_name,
        split=args.split,
        reference=args.reference,
        baseline=args.baseline,
        output_subdir=args.output_subdir,
        per_instance_convergence=not args.no_per_instance,
    )


if __name__ == "__main__":
    main()
