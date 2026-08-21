"""Median solution quality against median objective-function evaluations.

A single-panel, presentation-ready version of the right half of
``figures/quality_vs_effort.png``.  Three things differ from the main figure:

  * only the evaluation axis is drawn, since evaluations - not wall-clock - are
    the cost measure this study leads with (the timing pass ran 22 workers in
    parallel, which inflates seconds and leaves the evaluation counter exact);
  * the legend sits **outside** the axes, so it can never cover a data point or
    the Pareto front;
  * the per-point labels are anchored to the top or the bottom of each error
    bar on a four-phase cycle, so labels of neighbouring points cannot collide
    even where four variants sit within 10 % of each other on the x axis.

Identity is carried by colour *and* marker shape (circle = local search on
offspring, square = on elites, diamond = no local search) so the two variants
that share a tab10 colour are still separable, and every point is directly
labelled.

Run:  python visualization/extra/quality_vs_evaluations.py [output.png]
      (paths are anchored to the project's root directory, so any working directory works)
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from analysis.analyze import label_of, variants_in  # noqa: E402
from visualization.plots import variant_style  # noqa: E402
from optimization.run_experiments import read_rows  # noqa: E402
from analysis.stats_utils import describe  # noqa: E402
from paths import DEFAULT_RESULTS_DIR, EXTRA_FIGURES_DIR  # noqa: E402

RESULTS = DEFAULT_RESULTS_DIR
DEFAULT_OUT = os.path.join(EXTRA_FIGURES_DIR, "quality_vs_evaluations.png")

#: Marker per local-search target.  A second visual channel beside colour, so
#: the 11th variant wrapping tab10 back to colour 0 is still unambiguous.
MARKERS = {"offspring": "o", "elites": "s", "none": "D"}

#: Candidate vertical label offsets in points, tried in order, alternating
#: above and below the marker and moving further out at each step.  A label is
#: placed at the first candidate whose rendered box clears every label already
#: placed, which is what keeps the three variants clustered between 76k and 79k
#: evaluations - about 30 px apart on this axis, far narrower than a label -
#: from stacking on top of one another.
#:
#: Offsets are measured from the marker rather than from the end of the error
#: bar: the bars span roughly two fitness units, so a label parked at the cap
#: ends up nearer a neighbouring point than its own.  The white bounding box
#: keeps it legible where it crosses a bar.
LABEL_OFFSETS = (12.0, -14.0, 26.0, -28.0, 40.0, -42.0, 54.0, -56.0)

#: Padding in points added around each label box before testing for overlap.
LABEL_PAD = 1.5


def short_name(variant: str) -> str:
    """Compact label for direct annotation."""
    if variant == "ga_ls00":
        return "no LS"
    if variant.startswith("rlga_"):
        return "RL-GA " + variant.split("_", 1)[1]
    if variant.startswith("ga_sched_"):
        return "sched " + variant.split("ga_sched_", 1)[1]
    rate, target = variant.replace("ga_ls", "").split("_", 1)
    return f"{rate}% {target}"


def target_of(variant: str) -> str:
    if variant.endswith("elites"):
        return "elites"
    if variant.endswith("offspring"):
        return "offspring"
    return "none"


def place_labels(fig, ax, labelled) -> None:
    """Annotate each point, alternating above and below to avoid collisions.

    Candidate offsets are tried in :data:`LABEL_OFFSETS` order and the first
    one whose *rendered* box clears every label already placed is kept, so the
    result is checked against real text extents rather than assumed from the
    spacing of the points.
    """
    renderer = fig.canvas.get_renderer()
    placed = []

    for x, y, variant, is_rl in labelled:
        for offset in LABEL_OFFSETS:
            annotation = ax.annotate(
                short_name(variant),
                xy=(x, y),
                textcoords="offset points",
                xytext=(0, offset),
                ha="center",
                va="bottom" if offset > 0 else "top",
                fontsize=7.5,
                fontweight="bold" if is_rl else "normal",
                color="0.10" if is_rl else "0.30",
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.85),
            )
            box = annotation.get_window_extent(renderer).expanded(1.0, 1.0)
            box = Bbox.from_extents(
                box.x0 - LABEL_PAD, box.y0 - LABEL_PAD,
                box.x1 + LABEL_PAD, box.y1 + LABEL_PAD,
            )
            if not any(box.overlaps(other) for other in placed):
                placed.append(box)
                break
            # Collided: drop this attempt and try the next offset.  The last
            # candidate is kept regardless, so a label is never lost.
            if offset != LABEL_OFFSETS[-1]:
                annotation.remove()
            else:
                placed.append(box)


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    rows = read_rows(os.path.join(RESULTS, "runs.csv"))
    variants = variants_in(rows)

    fig, ax = plt.subplots(figsize=(9.6, 5.4))

    points = []
    for index, variant in enumerate(variants):
        subset = [r for r in rows if r["variant"] == variant]
        quality = [r["best_fitness"] for r in subset if np.isfinite(r["best_fitness"])]
        effort = [r["evaluations"] for r in subset]
        if not quality or not effort:
            continue
        summary = describe(quality)
        x = float(np.median(effort))
        points.append((x, summary.median, summary.median_ci_low, summary.median_ci_high,
                       variant, index))

    points.sort()

    # --- Pareto front (minimise both effort and fitness) --------------------
    front = []
    for x, y, _, _, variant, _ in points:
        if not front or y < front[-1][1]:
            front.append((x, y, variant))
    if len(front) > 1:
        ax.plot(
            [p[0] for p in front],
            [p[1] for p in front],
            color="0.35",
            linewidth=1.3,
            linestyle="--",
            alpha=0.9,
            zorder=1,
            label="Pareto front",
        )

    # --- points, error bars and direct labels -------------------------------
    labelled = []
    for x, y, lo, hi, variant, index in points:
        is_rl = variant.startswith("rlga")
        colour = variant_style(variant, index)["color"]
        ax.errorbar(
            x,
            y,
            yerr=[[y - lo], [hi - y]],
            fmt=MARKERS[target_of(variant)],
            markersize=10 if is_rl else 7,
            markerfacecolor=colour,
            markeredgecolor="black" if is_rl else colour,
            markeredgewidth=1.2 if is_rl else 0.0,
            ecolor=colour,
            elinewidth=1.4,
            capsize=4,
            # Without an explicit capthick the caps inherit ``markeredgewidth``,
            # which is 0 for the non-RL variants - so their interval ends were
            # being drawn at zero width, i.e. invisibly.
            capthick=1.4,
            zorder=4 if is_rl else 3,
            label=label_of(rows, variant),
        )

        labelled.append((x, y, variant, is_rl))

    # --- axes ---------------------------------------------------------------
    # Logarithmic effort axis: the variants span 60k-222k evaluations and a
    # linear axis leaves the seven cheapest bunched into the left third.  The
    # tick labels are written out in full so the axis reads without the scale
    # having to be named in the caption.
    ax.set_xscale("log")
    ticks = [60_000, 80_000, 100_000, 150_000, 200_000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t // 1000}k" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Median objective-function evaluations per run")
    ax.set_ylabel("Median best fitness (95 % bootstrap CI)")
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Multiplicative padding, since the axis is logarithmic.  Asymmetric, and
    # only just wide enough on the right: the right-most variant's label is
    # centred on its marker and so overhangs it by half a label width, but any
    # more headroom than that reads as dead space at the end of the axis.
    ax.set_xlim(min(p[0] for p in points) * 0.90, max(p[0] for p in points) * 1.06)

    # Headroom above for the outermost labels, and a deliberately clear band
    # below every error bar for the legend to sit in.
    y_lo = min(p[2] for p in points)
    y_hi = max(p[3] for p in points)
    y_span = y_hi - y_lo
    ax.set_ylim(y_lo - 0.30 * y_span, y_hi + 0.10 * y_span)

    ax.set_title(
        "Solution quality against evaluations (lower-left is better)",
        fontsize=12,
        pad=10,
    )

    # Legend inside the axes, in the empty band reserved below the data.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        fontsize=7,
        loc="lower center",
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="0.85",
        framealpha=0.95,
        borderpad=0.6,
        columnspacing=1.4,
        handletextpad=0.6,
    )

    # Labels go on last, after tight_layout has fixed the axes geometry, so the
    # overlap test is done against the positions the saved figure will use.
    fig.tight_layout()
    place_labels(fig, ax, labelled)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {out_path}")
    print("\nPareto front (median evaluations, median fitness):")
    for x, y, variant in front:
        print(f"  {variant:<22} {x:>10,.0f}  {y:6.2f}")


if __name__ == "__main__":
    main()
