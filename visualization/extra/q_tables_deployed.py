"""The deployed Q-tables of the two RL variants, side by side (walkthrough section 6.4).

Colour map is viridis, deliberately: it is perceptually uniform and monotone in
luminance, so brighter always means a higher Q value.  A diverging map such as
RdYlGn is brightest in the MIDDLE of its range (luminance 0.15 -> 0.98 -> 0.31),
which would encode the greedy cell as lighter than its neighbours in low-valued
rows and darker in high-valued rows -- unreadable for a quantity with no
meaningful midpoint.

Averages the end-of-run Q-table stored in every RL run's trace under ``raw/`` --
100 runs per variant -- and draws the two as heat maps on a shared colour scale,
with the greedy action boxed in each state.  States where the two variants pick
the same greedy action are flagged, because that overlap (3 of 9) is the claim
section 6.4 rests on.

Unlike ``training_summary.py``, this reads only from this directory: the
``q_table`` array is saved inside each run's ``.npz``.

Run:  python visualization/extra/q_tables_deployed.py [output.png]
      (paths are anchored to the project's root directory, so any working directory works)
"""
from __future__ import annotations

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from optimization.rl_control import StateBins  # noqa: E402
from paths import DEFAULT_RESULTS_DIR, EXTRA_FIGURES_DIR  # noqa: E402

RAW = os.path.join(DEFAULT_RESULTS_DIR, "raw")
VARIANTS = ("rlga_elites", "rlga_offspring")
ACTIONS = ("mut_up", "mut_dn", "xo_up", "xo_dn", "ls_up", "ls_dn", "reset")

bins = StateBins()  # only its labelling is used; thresholds do not affect the plot
tables = {
    v: np.mean([np.load(f)["q_table"] for f in sorted(glob.glob(os.path.join(RAW, v, "*.npz")))],
               axis=0)
    for v in VARIANTS
}
greedy = {v: q.argmax(1) for v, q in tables.items()}
agree = greedy[VARIANTS[0]] == greedy[VARIANTS[1]]

low = min(q.min() for q in tables.values())
high = max(q.max() for q in tables.values())


def annotate(ax, values: np.ndarray, shade: np.ndarray) -> None:
    """Print each cell's Q value, in white on dark cells and black on bright ones.

    ``shade`` is the cell's position on the colour map in [0, 1]; viridis is dark
    at 0 and bright at 1, so the 0.55 cut keeps the text legible either way.
    """
    for state in range(values.shape[0]):
        for action in range(values.shape[1]):
            ax.text(action, state, f"{values[state, action]:.3f}",
                    ha="center", va="center", fontsize=7,
                    color="black" if shade[state, action] > 0.55 else "white")

fig, grid = plt.subplots(2, 2, figsize=(15, 10.5))
axes = grid[0]
for panel, (ax, v) in enumerate(zip(axes, VARIANTS)):
    q = tables[v]
    im = ax.imshow(q, cmap="viridis", vmin=low, vmax=high, aspect="auto")
    annotate(ax, q, (q - low) / (high - low))
    for state in range(q.shape[0]):
        for edge, width in (("white", 4.0), ("black", 1.6)):
            ax.add_patch(Rectangle((greedy[v][state] - 0.5, state - 0.5), 1, 1,
                                   fill=False, ec=edge, lw=width))
    ax.set_xticks(range(len(ACTIONS)))
    ax.set_xticklabels(ACTIONS, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(q.shape[0]))
    # Full state names on the left panel only; the right panel repeats the same
    # nine rows, so indices there are enough and long labels would collide with
    # the neighbouring heat map.
    ax.set_yticklabels([f"{s}  {bins.label(s)}" if panel == 0 else str(s)
                        for s in range(q.shape[0])], fontsize=8)
    for state, same in enumerate(agree):
        if same:
            ax.get_yticklabels()[state].set_color("#15803d")
            ax.get_yticklabels()[state].set_fontweight("bold")
    within = float(np.mean(q.max(1) - q.min(1)))
    between = float(q.mean(1).max() - q.mean(1).min())
    ax.set_title(f"{v}  (mean of 100 end-of-run tables)\n"
                 f"within-state spread {within:.3f}   vs   between-state range {between:.3f}",
                 fontsize=10)
fig.colorbar(im, ax=axes, label="Q value", fraction=0.025, pad=0.02)

# ---- second row: the same tables with each row scaled to its own range -----
# On the absolute scale above, the within-row differences that actually decide
# the greedy action are 7-8 % of the colour range and therefore invisible.
# Rescaling per row makes the choice visible -- and shows how little separates
# it from the runners-up.  The margin printed on the right is best minus
# second-best, in raw Q units.
for panel, (ax, v) in enumerate(zip(grid[1], VARIANTS)):
    q = tables[v]
    span = (q.max(1) - q.min(1))[:, None]
    scaled = (q - q.min(1)[:, None]) / np.where(span > 0, span, 1.0)
    ax.imshow(scaled, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    # The printed values are the raw Q values, not the rescaled ones -- the
    # rescaling drives colour only, so that the greedy cell is visible.
    annotate(ax, q, scaled)
    ordered = np.sort(q, axis=1)
    for state in range(q.shape[0]):
        for edge, width in (("white", 4.0), ("black", 1.6)):
            ax.add_patch(Rectangle((greedy[v][state] - 0.5, state - 0.5), 1, 1,
                                   fill=False, ec=edge, lw=width))
        ax.text(len(ACTIONS) - 0.35, state,
                f"margin {ordered[state, -1] - ordered[state, -2]:.4f}",
                va="center", fontsize=8, color="#334155")
    ax.set_xticks(range(len(ACTIONS)))
    ax.set_xticklabels(ACTIONS, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(q.shape[0]))
    ax.set_yticklabels([f"{s}  {bins.label(s)}" if panel == 0 else str(s)
                        for s in range(q.shape[0])], fontsize=8)
    ax.set_xlim(-0.5, len(ACTIONS) + 1.6)
    ax.set_title(f"{v} — each row rescaled to its own min/max", fontsize=10)

fig.suptitle("Deployed Q-tables.  Top: absolute scale — colour varies by row (state), not by "
             "column (action).\nBottom: the same rows rescaled individually, so the greedy "
             f"choice is visible along with the margin that decided it.\nBoxed = greedy "
             f"action; green state labels = the {int(agree.sum())} of 9 states where both "
             "variants agree.", fontsize=11)
os.makedirs(EXTRA_FIGURES_DIR, exist_ok=True)
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    EXTRA_FIGURES_DIR, "q_tables_deployed.png"
)
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}")

for v in VARIANTS:
    q = tables[v]
    srt = np.sort(q, 1)
    print(f"{v}: range [{q.min():.3f}, {q.max():.3f}] | within-state spread "
          f"{np.mean(q.max(1) - q.min(1)):.4f} | between-state range "
          f"{q.mean(1).max() - q.mean(1).min():.4f} | best-vs-2nd margin "
          f"{np.mean(srt[:, -1] - srt[:, -2]):.4f}")
    print(f"   greedy: {[ACTIONS[i] for i in greedy[v]]}")
print(f"agreement: {int(agree.sum())}/9 states -> {list(np.flatnonzero(agree))}")
