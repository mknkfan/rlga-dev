"""Convergence of all eleven variants, one figure per budget axis.

Differs from the main ``figures/convergence_*.png`` in four ways, all
requested for the manuscript version:

  * elites and offspring appear on the **same** axes, so the two local-search
    targets can be compared directly rather than across two figures;
  * one figure per budget axis (generations, evaluations, wall-clock) instead
    of three panels side by side, so each gets the full page width;
  * **no interquartile band** - a single median line per variant, since eleven
    shaded bands overlap into illegibility;
  * fitness is plotted in its **own units**, not divided by the best value
    found on each instance.  Pooling raw values across instances of different
    difficulty widens the spread, but the axis is then readable against every
    table in the study.

Line identity is carried by colour *and* dash pattern rather than colour alone:
colour encodes the local-search rate family (none / 10 % / 50 % / 100 % /
scheduled / RL) and the dash pattern the target (solid = elites, dashed =
offspring).  Eleven lines cannot be separated by hue alone at print size.

Run:  python visualization/extra/convergence_by_metric.py [output_dir]
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

from analysis.analyze import instances_in, label_of, load_trace, variants_in  # noqa: E402
from visualization.plots import _aligned_curves  # noqa: E402
from optimization.run_experiments import read_rows  # noqa: E402
from paths import DEFAULT_RESULTS_DIR, EXTRA_FIGURES_DIR  # noqa: E402

RESULTS = DEFAULT_RESULTS_DIR

#: Colour per local-search rate family.  Chosen so the three static rates read
#: as an ordered ramp (blue -> green -> red) and the two adaptive controllers
#: sit outside it.
FAMILY_COLOUR = {
    "none": "#4d4d4d",
    "ls10": "#1f77b4",
    "ls50": "#2ca02c",
    "ls100": "#d62728",
    "sched": "#9467bd",
    "rl": "#ff7f0e",
}

#: Dash pattern per local-search target.  The second visual channel.
TARGET_DASH = {
    "elites": (None, None),      # solid
    "offspring": (6, 2),         # dashed
    "none": (1, 1.6),            # dotted, for the no-local-search baseline
}

AXES = (
    ("generation", "Generation", False, "convergence_generations.png"),
    ("evaluations", "Cumulative objective-function evaluations", True,
     "convergence_evaluations.png"),
    ("elapsed", "Cumulative wall-clock time (s)", True, "convergence_walltime.png"),
)


def family_of(variant: str) -> str:
    if variant == "ga_ls00":
        return "none"
    if variant.startswith("rlga"):
        return "rl"
    if "sched" in variant:
        return "sched"
    # ls100 must be tested before ls10, which is a prefix of it.
    if "ls100" in variant:
        return "ls100"
    if "ls50" in variant:
        return "ls50"
    return "ls10"


def target_of(variant: str) -> str:
    if variant.endswith("elites"):
        return "elites"
    if variant.endswith("offspring"):
        return "offspring"
    return "none"


def style_of(variant: str) -> dict:
    is_rl = variant.startswith("rlga")
    return {
        "color": FAMILY_COLOUR[family_of(variant)],
        "dashes": TARGET_DASH[target_of(variant)],
        "linewidth": 2.6 if is_rl else 1.5,
        "zorder": 4 if is_rl else 3,
        "alpha": 1.0 if is_rl else 0.9,
    }


def collect(variants, instances, seeds):
    """``variant -> list of per-run trace dicts``."""
    out = {}
    for variant in variants:
        traces = []
        for instance in instances:
            for seed in seeds:
                trace = load_trace(RESULTS, variant, instance, seed)
                if trace is not None:
                    traces.append(trace)
        if traces:
            out[variant] = traces
    return out


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else EXTRA_FIGURES_DIR
    os.makedirs(out_dir, exist_ok=True)
    rows = read_rows(os.path.join(RESULTS, "runs.csv"))
    variants = variants_in(rows)
    instances = instances_in(rows)
    seeds = sorted({r["seed"] for r in rows})

    print(f"loading traces for {len(variants)} variants "
          f"x {len(instances)} instances x {len(seeds)} seeds ...")
    collected = collect(variants, instances, seeds)

    for axis_key, xlabel, log_axis, filename in AXES:
        fig, ax = plt.subplots(figsize=(10.0, 5.8))
        finals = []
        drawn = []

        for variant in variants:
            traces = collected.get(variant, [])
            if not traces:
                continue

            if axis_key == "generation":
                length = min(t["best_so_far"].size for t in traces)
                grid = np.arange(length, dtype=float)
                curves = np.vstack([t["best_so_far"][:length] for t in traces])
            else:
                lo = max(float(np.min([t[axis_key][0] for t in traces])), 1e-6)
                hi = float(np.median([t[axis_key][-1] for t in traces]))
                grid = np.geomspace(lo, hi, 160) if log_axis else np.linspace(lo, hi, 160)
                curves = _aligned_curves(traces, axis_key, grid)

            with np.errstate(invalid="ignore"):
                median = np.nanmedian(
                    np.where(np.isfinite(curves), curves, np.nan), axis=0
                )
            ax.plot(grid, median, label=label_of(rows, variant), **style_of(variant))
            finals.append(float(np.nanmin(median)))
            drawn.append((grid, median))

        # Clip to the band where the variants actually separate.  Raw fitness
        # starts near 110 and ends near 35, so an unclipped axis spends four
        # fifths of its height on an initial descent during which all eleven
        # curves are indistinguishable, and compresses the final 1.3-unit
        # spread - the thing being compared - into a few pixels.  The early
        # generations run off the top of the frame by design.
        floor = min(finals)
        top = floor * 1.32
        ax.set_ylim(floor - 0.4, top)

        if log_axis:
            ax.set_xscale("log")
            # Start the axis just before the first curve drops into the visible
            # band, instead of at the first evaluation of the first generation:
            # otherwise a third of the width carries no data.
            entries = [
                float(g[np.argmax(m <= top)])
                for g, m in drawn
                if np.any(m <= top)
            ]
            if entries:
                ax.set_xlim(min(entries) * 0.55, max(g[-1] for g, _ in drawn) * 1.05)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Median best fitness so far")
        ax.grid(alpha=0.25, linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        ax.set_title(
            "Convergence over 100 runs per variant  (lower is better)",
            fontsize=12, pad=10,
        )
        # On the budget axes the curves sweep down through the top-right, so the
        # legend goes bottom-left where the panel is empty; on the generation
        # axis the reverse holds.
        legend_loc = "lower left" if log_axis else "upper right"
        ax.legend(fontsize=7.5, loc=legend_loc, ncol=2, frameon=True,
                  facecolor="white", edgecolor="0.85", framealpha=0.95,
                  borderpad=0.6, columnspacing=1.2, handlelength=3.0)

        fig.tight_layout()
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
