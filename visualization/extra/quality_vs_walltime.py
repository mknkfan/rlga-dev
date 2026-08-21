"""Median solution quality against median wall-clock time per run.

The wall-clock companion to ``quality_vs_evaluations.py``: identical layout,
identical estimator (marginal medians, matching ``analysis/report.md`` §1), only
the cost axis differs.

**Read it with the timing caveat.**  The runs were executed with 22 parallel
workers, which inflates wall-clock and adds noise to it while leaving the
evaluation counter exact.  Evaluations are therefore the cost axis this study
leads with, and this figure is corroboration: it should reproduce the *ordering*
of the evaluation figure, and the fact that it does is the evidence that the
ordering is not an artefact of how evaluations are counted.  Pass ``--cpu`` to
plot CPU time instead, which is far less sensitive to co-scheduled processes.

Run:  python visualization/extra/quality_vs_walltime.py [output.png] [--cpu]
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)  # sibling module below

from analysis.analyze import label_of, variants_in  # noqa: E402
from visualization.plots import variant_style  # noqa: E402
from optimization.run_experiments import read_rows  # noqa: E402
from analysis.stats_utils import describe  # noqa: E402
from paths import DEFAULT_RESULTS_DIR, EXTRA_FIGURES_DIR  # noqa: E402
from quality_vs_evaluations import MARKERS, place_labels, target_of  # noqa: E402

RESULTS = DEFAULT_RESULTS_DIR


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_cpu = "--cpu" in sys.argv

    column = "cpu_time" if use_cpu else "execution_time"
    axis_label = (
        "Median CPU time per run (s)" if use_cpu
        else "Median wall-clock time per run (s)"
    )
    default_name = "quality_vs_cputime.png" if use_cpu else "quality_vs_walltime.png"
    os.makedirs(EXTRA_FIGURES_DIR, exist_ok=True)
    out_path = args[0] if args else os.path.join(EXTRA_FIGURES_DIR, default_name)

    rows = read_rows(os.path.join(RESULTS, "runs.csv"))
    variants = variants_in(rows)

    points = []
    for index, variant in enumerate(variants):
        subset = [r for r in rows if r["variant"] == variant]
        quality = [r["best_fitness"] for r in subset if np.isfinite(r["best_fitness"])]
        cost = [r[column] for r in subset]
        if not quality or not cost:
            continue
        summary = describe(quality)
        points.append((float(np.median(cost)), summary.median,
                       summary.median_ci_low, summary.median_ci_high, variant, index))
    points.sort()

    fig, ax = plt.subplots(figsize=(9.6, 5.4))

    # --- Pareto front (minimise both cost and fitness) ----------------------
    front = []
    for x, y, _, _, variant, _ in points:
        if not front or y < front[-1][1]:
            front.append((x, y, variant))
    if len(front) > 1:
        ax.plot(
            [p[0] for p in front], [p[1] for p in front],
            color="0.35", linewidth=1.3, linestyle="--", alpha=0.9,
            zorder=1, label="Pareto front",
        )

    # --- points, error bars and direct labels -------------------------------
    labelled = []
    for x, y, lo, hi, variant, index in points:
        is_rl = variant.startswith("rlga")
        colour = variant_style(variant, index)["color"]
        ax.errorbar(
            x, y,
            yerr=[[y - lo], [hi - y]],
            fmt=MARKERS[target_of(variant)],
            markersize=10 if is_rl else 7,
            markerfacecolor=colour,
            markeredgecolor="black" if is_rl else colour,
            markeredgewidth=1.2 if is_rl else 0.0,
            ecolor=colour,
            elinewidth=1.4,
            capsize=4,
            capthick=1.4,
            zorder=4 if is_rl else 3,
            label=label_of(rows, variant),
        )
        labelled.append((x, y, variant, is_rl))

    # --- axes ---------------------------------------------------------------
    # Linear, unlike the evaluations figure: the cost range here is only about
    # 2x (8.4 s to 17.7 s) against 3.7x there, and a log axis over that span
    # buys no separation while making the tick labels harder to read.
    ax.set_xlabel(axis_label)
    ax.set_ylabel("Median best fitness (95 % bootstrap CI)")
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    x_span = max(p[0] for p in points) - min(p[0] for p in points)
    ax.set_xlim(min(p[0] for p in points) - 0.06 * x_span,
                max(p[0] for p in points) + 0.14 * x_span)

    y_lo = min(p[2] for p in points)
    y_hi = max(p[3] for p in points)
    y_span = y_hi - y_lo
    ax.set_ylim(y_lo - 0.30 * y_span, y_hi + 0.10 * y_span)

    ax.set_title(
        "Solution quality against computational effort (lower-left is better)",
        fontsize=12, pad=10,
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        fontsize=7, loc="lower center", ncol=3,
        frameon=True, facecolor="white", edgecolor="0.85", framealpha=0.95,
        borderpad=0.6, columnspacing=1.4, handletextpad=0.6,
    )

    fig.tight_layout()
    place_labels(fig, ax, labelled)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {out_path}  ({column})\n")
    print(f"{'variant':<22}{'seconds':>10}{'fitness':>9}")
    for x, y, _, _, variant, _ in points:
        on_front = any(variant == f[2] for f in front)
        print(f"{variant:<22}{x:>10.1f}{y:>9.2f}{'   <- front' if on_front else ''}")


if __name__ == "__main__":
    main()
