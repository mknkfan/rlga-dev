"""Quality against effort, estimated on **paired per-seed** data.

Same figure as ``quality_vs_evaluations.py`` - one panel, one point per variant -
but the fitness axis is estimated on the paired data rather than marginally.

``quality_vs_evaluations.py`` uses each variant's *marginal* median: its own
median over its 100 runs, ignoring which run was paired with which.  That throws
away the design.  All eleven variants share an initial population and random
stream on every (instance, seed) pair, and the variation *between* those pairs
is far larger than the variation between variants - 2.47 fitness units between
instances and 3.09 seed-to-seed, against 0.30 between variant means.  A marginal
median therefore carries the block noise into every point, which is why the
marginal and paired statistics disagree in sign for several comparisons
(``rlga_elites`` against ``ga_ls00`` is +0.12 marginally and -0.47 paired).

This script removes the block effect without designating a reference variant,
which is the standard estimate for a randomised complete block design:

  1. within each of the 100 (instance, seed) blocks, take the mean across the
     eleven variants - that is the block's difficulty;
  2. subtract it from every run in the block, leaving each run's performance
     *relative to what that particular starting point yielded*;
  3. take the mean of a variant's 100 centred values, and add the grand mean
     back so the axis is still in fitness units.

Differences between two points are then paired differences, free of instance
and seed effects, and no variant is privileged as the origin: the gap between
any two points is exactly the mean paired difference tabulated in
``analysis/report.md`` §2/§5.  See ``block_centred`` for why the mean, and not
the median, is the only aggregation with that property.

Note that in a balanced design the blocked mean equals the marginal mean, so the
positions match the mean column of §1; what the pairing buys is the width of the
intervals, which measure the variant effect rather than the problem's spread.

Run:  python visualization/extra/paired_quality_vs_evaluations.py [out.png]
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

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
from analysis.stats_utils import bootstrap_ci  # noqa: E402
from run.paths import DEFAULT_RESULTS_DIR, EXTRA_FIGURES_DIR  # noqa: E402
from quality_vs_evaluations import MARKERS, place_labels, target_of  # noqa: E402

RESULTS = DEFAULT_RESULTS_DIR
DEFAULT_OUT = os.path.join(EXTRA_FIGURES_DIR, "paired_quality_vs_evaluations.png")


def block_centred(rows, variants, metric):
    """Per-variant median of block-centred values, plus a bootstrap interval.

    Returns ``{variant: (estimate, ci_low, ci_high)}`` in the metric's own
    units: the block medians are subtracted and the grand median added back.
    """
    blocks = defaultdict(dict)
    for row in rows:
        value = row[metric]
        if np.isfinite(value):
            blocks[(row["instance"], row["seed"])][row["variant"]] = float(value)

    # Only blocks in which every variant produced a finite result can be
    # centred, otherwise the block median is computed on a different set of
    # variants than the one being centred - which would reintroduce a bias.
    complete = {k: v for k, v in blocks.items() if len(v) == len(variants)}

    # The block centre is the *mean* over the eleven variants, not the median.
    # With an odd number of variants the block median is literally one of their
    # values, so that variant's centred value is exactly zero; across 100 blocks
    # this puts an atom at zero in every variant's distribution and collapses
    # five of the eleven estimates onto the same tied value.  The mean has no
    # such atom.
    centred = defaultdict(list)
    for values in complete.values():
        centre = float(np.mean(list(values.values())))
        for variant, value in values.items():
            centred[variant].append(value - centre)

    # Added back only so the axis reads in fitness units rather than as a
    # deviation around zero; it is a constant shift and moves every point alike.
    #
    # The anchor is the grand *mean*, matching the statistic the estimates are
    # aggregated with below.  The two differ by 0.34 units here (35.505 against
    # a median of 35.166) because the fitness distribution is right-skewed, so
    # mixing them would shift the whole axis relative to the table it is read
    # against.
    grand = float(np.mean([v for values in complete.values() for v in values.values()]))

    # Aggregated with the **mean**, which is the only choice that makes a
    # vertical gap in this figure mean what a reader will assume it means.
    #
    # The mean commutes with the centring, so the gap between two points is
    # exactly the mean paired difference tabulated in ``analysis/report.md``
    # §2/§5 - ``rlga_elites`` against ``rlga_offspring`` gives +0.153 in both.
    # The median does not commute: median(x_i - c_i) is not median(x_i - y_i),
    # and here they disagree in sign (-0.058 against the table's +0.530), so a
    # median-aggregated figure ranks the two RL variants the opposite way round
    # from the study's own tables.  That is not fixable by a different centring:
    # paired medians are not additive - median(A - B) + median(B - C) is not
    # median(A - C) - so no assignment of one number per variant can reproduce
    # them, and any single-axis figure implies additive differences.
    #
    # The cost of the mean is sensitivity to the heavy tails (single pairs
    # differ by up to 19 fitness units); the bootstrap intervals below carry
    # that sensitivity honestly.
    out = {}
    for variant in variants:
        data = np.asarray(centred[variant], dtype=float)
        if data.size == 0:
            continue
        lo, hi = bootstrap_ci(data, np.mean)
        out[variant] = (float(np.mean(data)) + grand, lo + grand, hi + grand)
    return out, len(complete)


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    rows = read_rows(os.path.join(RESULTS, "runs.csv"))
    variants = variants_in(rows)

    fitness, n_blocks = block_centred(rows, variants, "best_fitness")

    # The effort axis is *not* block-centred.  A variant's evaluation count is
    # fixed by its local-search rate and barely varies across instances or seeds
    # (interquartile range over median is 0.000-0.006 for every static and
    # scheduled variant), so there is no block effect to remove - centring would
    # only subtract a constant that depends on which variants are in the mix and
    # would strip the axis of its absolute meaning.  The plain marginal median
    # is used, matching ``analysis/report.md`` §1 exactly.
    effort = {}
    for variant in variants:
        values = [r["evaluations"] for r in rows if r["variant"] == variant]
        if values:
            effort[variant] = float(np.median(values))

    points = []
    for index, variant in enumerate(variants):
        if variant not in fitness or variant not in effort:
            continue
        y, y_lo, y_hi = fitness[variant]
        points.append((effort[variant], y, y_lo, y_hi, variant, index))
    points.sort()

    fig, ax = plt.subplots(figsize=(9.6, 5.4))

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
    ax.set_xscale("log")
    ticks = [60_000, 80_000, 100_000, 150_000, 200_000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t // 1000}k" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Median objective-function evaluations per run")
    ax.set_ylabel("Paired mean best fitness (95 % bootstrap CI)")
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.set_xlim(min(p[0] for p in points) * 0.90, max(p[0] for p in points) * 1.06)

    y_lo = min(p[2] for p in points)
    y_hi = max(p[3] for p in points)
    y_span = y_hi - y_lo
    ax.set_ylim(y_lo - 0.45 * y_span, y_hi + 0.18 * y_span)

    ax.set_title(
        "Paired median solution quality (fitness) against computational effort (evaluation)",
        fontsize=12,
        pad=10,
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

    print(f"wrote {out_path}   ({n_blocks} complete (instance, seed) blocks)\n")
    print(f"{'variant':<22}{'evaluations':>12}{'fitness':>9}{'95 % CI':>18}")
    for x, y, lo, hi, variant, _ in points:
        on_front = any(variant == f[2] for f in front)
        print(f"{variant:<22}{x:>12,.0f}{y:>9.2f}   [{lo:5.2f}, {hi:5.2f}]"
              f"{'   <- front' if on_front else ''}")


if __name__ == "__main__":
    main()
