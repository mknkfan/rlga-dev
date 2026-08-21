"""The trained agents' own Q-tables, averaged over their 3 independent seeds.

Unlike ``q_tables_deployed.py`` (which averages the end-of-run Q-table
snapshot from all 100 evaluation runs under ``raw/``), this reads the Q-table
each agent was actually trained to, straight from ``results/agents/*.npz`` --
the same 3-seed mean the report and paper's Q-table discussion is about.

Colour map is viridis, deliberately: it is perceptually uniform and monotone
in luminance, so brighter always means a higher Q value.

Run:  python visualization/extra/q_tables_agent_mean.py [output.png]
      (paths are anchored to the project's root directory, so any working directory works)
"""
from __future__ import annotations

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
from paths import DEFAULT_AGENT_DIR  # noqa: E402

VARIANTS = ("rlga_elites", "rlga_offspring")
SEEDS = (0, 1, 2)
ACTIONS = ("mut_up", "mut_dn", "xo_up", "xo_dn", "ls_up", "ls_dn", "reset")

bins = StateBins()  # only its labelling is used; thresholds do not affect the plot
tables = {
    v: np.mean(
        [np.load(os.path.join(DEFAULT_AGENT_DIR, f"{v}_seed{s}.npz"))["q_table"] for s in SEEDS],
        axis=0,
    )
    for v in VARIANTS
}
greedy = {v: q.argmax(1) for v, q in tables.items()}
agree = greedy[VARIANTS[0]] == greedy[VARIANTS[1]]

low = min(q.min() for q in tables.values())
high = max(q.max() for q in tables.values())


def annotate(ax, values: np.ndarray, shade: np.ndarray) -> None:
    """Print each cell's Q value, in white on dark cells and black on bright ones."""
    for state in range(values.shape[0]):
        for action in range(values.shape[1]):
            ax.text(action, state, f"{values[state, action]:.3f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if shade[state, action] > 0.55 else "white")


fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), constrained_layout=True)
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
                        for s in range(q.shape[0])], fontsize=9)
    for state, same in enumerate(agree):
        if same:
            ax.get_yticklabels()[state].set_color("#15803d")
            ax.get_yticklabels()[state].set_fontweight("bold")
    within = float(np.mean(q.max(1) - q.min(1)))
    between = float(q.mean(1).max() - q.mean(1).min())
    ax.set_title(f"{v}\nwithin-state spread {within:.4f}   vs   between-state range {between:.3f}",
                 fontsize=10)
fig.colorbar(im, ax=axes, label="Q value", fraction=0.025, pad=0.02)
fig.suptitle("Trained Q-tables (mean over the 3 independently trained seeds per variant)\n"
             f"Boxed = greedy action; green state label = the {int(agree.sum())} of 9 states "
             "where both variants' greedy action agrees", fontsize=11)

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DEFAULT_AGENT_DIR, "q_tables_mean.png")
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
fig.savefig(out, dpi=130)
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
