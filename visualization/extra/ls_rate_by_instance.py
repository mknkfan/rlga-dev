"""Local-search rate during testing, per instance, for both RL-GA variants.

Reads the per-generation ``ls_rate`` trace from every RL run in ``raw/`` (10
instances x 10 seeds per variant) and writes **three separate figures**:

  ``ls_rate_trajectory_elites.png``     the rate over the 300 generations, one
  ``ls_rate_trajectory_offspring.png``  line per instance (median across that
                                        instance's 10 seeds), pooled median in
                                        bold, static baselines for reference;
  ``ls_rate_by_instance.png``           the per-instance time-averaged rate with
                                        its spread across seeds, both variants
                                        side by side.

Each instance's 10 runs use agent seeds 0/1/2 in the same 4/3/3 proportion, so
per-instance differences are not confounded by which agent was loaded.

Every reference line is named in the legend rather than annotated in place, and
every legend sits outside the axes: with eleven trajectories filling the panel
there is no interior region that stays clear at all rates, so inline labels for
the static baselines collided with whichever curve happened to pass through
them.

Run:  python visualization/extra/ls_rate_by_instance.py [output_dir]
      (paths are anchored to the project's root directory, so any working directory works)
"""
from __future__ import annotations

import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from run.paths import DEFAULT_RESULTS_DIR, EXTRA_FIGURES_DIR  # noqa: E402

RAW = os.path.join(DEFAULT_RESULTS_DIR, "raw")
VARIANTS = ("rlga_elites", "rlga_offspring")
TITLE = {"rlga_elites": "RL-GA [LS on elites]", "rlga_offspring": "RL-GA [LS on offspring]"}
SHORT = {"rlga_elites": "elites", "rlga_offspring": "offspring"}

STATIC_50 = "#dc2626"
STATIC_10 = "#2563eb"


def load(variant: str) -> dict:
    """instance name -> (seeds x generations) array of local-search rate."""
    per_instance: dict = {}
    for path in sorted(glob.glob(os.path.join(RAW, variant, "*.npz"))):
        name = re.match(r"(test_\d+)_run\d+\.npz", os.path.basename(path)).group(1)
        per_instance.setdefault(name, []).append(np.load(path)["ls_rate"])
    return {k: np.vstack(v) for k, v in per_instance.items()}


def trajectory_figure(data, instances, colours, out_dir):
    """Both variants side by side, sharing one y-axis and one legend.

    The two panels carry identical series, so a legend per panel would repeat
    thirteen entries; a single legend outside the axes serves both and cannot
    overlap either.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2), sharey=True)

    for ax, variant in zip(axes, VARIANTS):
        for colour, name in zip(colours, instances):
            ax.plot(np.median(data[variant][name], axis=0), color=colour, lw=1.0,
                    alpha=0.85, label=name)
        pooled = np.median(np.vstack([data[variant][n] for n in instances]), axis=0)
        ax.plot(pooled, color="black", lw=2.6, label="pooled median")

        # Named in the legend, not annotated on the axes: at 0.5 and 0.1 these
        # lines run straight through the band the trajectories occupy.
        ax.axhline(0.5, color=STATIC_50, ls="--", lw=1.2, label="static 50 % rate")
        ax.axhline(0.1, color=STATIC_10, ls=":", lw=1.4, label="static 10 % rate")

        ax.set_xlabel("Generation")
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlim(0, 299)
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_title(TITLE[variant], fontsize=11)

    axes[0].set_ylabel("Local-search rate")
    fig.suptitle("Median local-search rate per instance, over the 300 generations",
                 fontsize=12.5, y=1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, loc="upper left",
               bbox_to_anchor=(0.90, 0.93), borderaxespad=0.0, frameon=False,
               handletextpad=0.7)

    fig.tight_layout(rect=(0, 0, 0.89, 1))
    path = os.path.join(out_dir, "ls_rate_trajectory.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def summary_figure(data, instances, out_dir):
    fig, ax = plt.subplots(figsize=(10.0, 5.2))

    width = 0.36
    offsets = {VARIANTS[0]: -width / 2, VARIANTS[1]: +width / 2}
    style = {VARIANTS[0]: ("#1d4ed8", "o"), VARIANTS[1]: ("#b45309", "s")}
    x = np.arange(len(instances))

    for variant in VARIANTS:
        per_run = np.array([data[variant][n].mean(axis=1) for n in instances])
        med = np.median(per_run, axis=1)
        q1, q3 = np.percentile(per_run, [25, 75], axis=1)
        colour, marker = style[variant]
        overall = per_run.mean()
        ax.errorbar(x + offsets[variant], med, yerr=[med - q1, q3 - med], fmt=marker,
                    color=colour, ms=7, capsize=4, lw=1.4,
                    label=f"{TITLE[variant]} — median, IQR over 10 seeds")
        # The overall mean is a legend entry too, so no text sits on the axes.
        ax.axhline(overall, color=colour, ls="-", lw=1.1, alpha=0.45,
                   label=f"{TITLE[variant]} — overall mean {overall:.2f}")

    ax.axhline(0.5, color=STATIC_50, ls="--", lw=1.2, label="static 50 % rate")
    ax.axhline(0.1, color=STATIC_10, ls=":", lw=1.4, label="static 10 % rate")

    ax.set_xticks(x)
    ax.set_xticklabels(instances)
    ax.set_ylabel("Time-averaged local-search rate over the run")
    ax.set_xlabel("Test instance")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(-0.6, len(instances) - 0.4)
    ax.grid(alpha=0.3, axis="y")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title("Per-instance time-averaged local-search rate", fontsize=11.5, pad=10)
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.015, 1.0),
              borderaxespad=0.0, frameon=False, handletextpad=0.7)

    fig.tight_layout()
    path = os.path.join(out_dir, "ls_rate_by_instance.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else EXTRA_FIGURES_DIR
    os.makedirs(out_dir, exist_ok=True)
    data = {v: load(v) for v in VARIANTS}
    instances = sorted(data[VARIANTS[0]], key=lambda s: int(s.split("_")[1]))
    colours = plt.cm.viridis(np.linspace(0, 0.9, len(instances)))

    print(f"wrote {trajectory_figure(data, instances, colours, out_dir)}")
    print(f"wrote {summary_figure(data, instances, out_dir)}\n")

    for variant in VARIANTS:
        per_run = np.array([data[variant][n].mean(axis=1) for n in instances])
        print(f"{TITLE[variant]}: overall mean rate {per_run.mean():.3f}")
        print(f"  per-instance means : {np.round(per_run.mean(axis=1), 3)}")
        print(f"  spread across instances {np.ptp(per_run.mean(axis=1)):.3f} | "
              f"mean spread across seeds within an instance "
              f"{np.mean(per_run.max(1) - per_run.min(1)):.3f}")
    lo = np.array([data["rlga_elites"][n].mean(axis=1).mean() for n in instances])
    hi = np.array([data["rlga_offspring"][n].mean(axis=1).mean() for n in instances])
    print(f"\nelite rate exceeds offspring rate on {int((lo > hi).sum())}/{len(instances)} instances")


if __name__ == "__main__":
    main()
