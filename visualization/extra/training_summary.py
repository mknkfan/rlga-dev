"""Training-stage summary for the two agents used by the archived study.

Reads the agent files in ``results/agents/`` -- the ONLY data used anywhere in this
directory that does not live inside it, because episode-level training histories
are stored with the agents and nowhere else.  Each ``.npz`` carries
``episode_reward_mean``, ``episode_best_fitness``, ``episode_q_delta``,
``episode_epsilon``, ``episode_instance``, ``action_counts`` and the trained
``q_table``.

Writes a four-panel figure: reward evolution, instance-normalised solution
quality, Q-table movement per episode, and the trained Q-tables themselves.

Run:  python visualization/extra/training_summary.py [output.png]
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

from paths import DEFAULT_AGENT_DIR, EXTRA_FIGURES_DIR  # noqa: E402

AGENT_DIR = DEFAULT_AGENT_DIR
VARIANTS = ("rlga_elites", "rlga_offspring")
SEEDS = (0, 1, 2)
ACTIONS = ("mut_up", "mut_dn", "xo_up", "xo_dn", "ls_up", "ls_dn", "reset")
COLOUR = {"rlga_elites": "#1d4ed8", "rlga_offspring": "#b45309"}

data = {
    v: [np.load(os.path.join(AGENT_DIR, f"{v}_seed{s}.npz"), allow_pickle=True) for s in SEEDS]
    for v in VARIANTS
}


def stacked(variant: str, key: str) -> np.ndarray:
    return np.array([d[key] for d in data[variant]], dtype=float)


def normalised_fitness(variant: str) -> np.ndarray:
    """Best fitness per episode, divided by that instance's own mean.

    Training episodes visit instances in shuffled order and instance fitness
    scales differ by ~35 %, so the raw series is dominated by which instance an
    episode happened to draw.  Normalising per instance is what makes a training
    trend visible if one exists.
    """
    best, inst = stacked(variant, "episode_best_fitness"), stacked(variant, "episode_instance")
    out = np.empty_like(best)
    for k in range(best.shape[0]):
        for i in np.unique(inst[k]):
            mask = inst[k] == i
            out[k][mask] = best[k][mask] / best[k][mask].mean()
    return out


fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
episodes = np.arange(stacked(VARIANTS[0], "episode_reward_mean").shape[1])
eps = stacked(VARIANTS[0], "episode_epsilon")[0]
eps_floor = int(np.argmax(eps <= 0.0501))


def smooth(y: np.ndarray, window: int = 15) -> np.ndarray:
    return np.convolve(y, np.ones(window) / window, mode="valid")


# (a) reward evolution -------------------------------------------------
ax = axes[0, 0]
for v in VARIANTS:
    r = stacked(v, "episode_reward_mean")
    for series in r:
        ax.plot(episodes, series, color=COLOUR[v], alpha=0.18, lw=0.8)
    ax.plot(episodes[14:], smooth(r.mean(0)), color=COLOUR[v], lw=2.2, label=f"{v} (15-episode mean)")
ax.axvline(eps_floor, color="#64748b", ls="--", lw=1)
ax.annotate(f"epsilon reaches its floor\n(0.05) at episode {eps_floor}", (eps_floor + 3, -0.048),
            fontsize=8, color="#475569")
ax.set_xlabel("Training episode (one 300-generation GA run)")
ax.set_ylabel("Mean reward per generation")
ax.set_title("(a) Reward evolution — flat across 150 episodes")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (b) solution quality on the training instances -----------------------
ax = axes[0, 1]
for v in VARIANTS:
    n = normalised_fitness(v)
    ax.plot(episodes[14:], smooth(n.mean(0)), color=COLOUR[v], lw=2.2, label=v)
ax.axhline(1.0, color="#94a3b8", lw=1)
ax.set_xlabel("Training episode")
ax.set_ylabel("Best fitness / that instance's mean")
ax.set_title("(b) Solution quality on training instances — no trend")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (c) + (d) trained Q-tables -------------------------------------------
for ax, v in zip(axes[1], VARIANTS):
    q = np.mean([d["q_table"] for d in data[v]], axis=0)
    im = ax.imshow(q, cmap="RdYlGn", aspect="auto")
    for state in range(q.shape[0]):
        ax.add_patch(Rectangle((q[state].argmax() - 0.5, state - 0.5), 1, 1,
                               fill=False, ec="black", lw=1.8))
    ax.set_xticks(range(len(ACTIONS)))
    ax.set_xticklabels(ACTIONS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(q.shape[0]))
    ax.set_ylabel("State")
    spread = float(np.mean(q.max(1) - q.min(1)))
    between = float(q.mean(1).max() - q.mean(1).min())
    ax.set_title(f"trained Q-table: {v}\nwithin-state spread {spread:.3f} vs "
                 f"between-state range {between:.3f}", fontsize=10)
    fig.colorbar(im, ax=ax, label="Q value")

fig.suptitle("Training stage: what the agents actually learned (3 seeds x 150 episodes each)",
             fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96))
os.makedirs(EXTRA_FIGURES_DIR, exist_ok=True)
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    EXTRA_FIGURES_DIR, "training_summary.png"
)
fig.savefig(out, dpi=130)
print(f"wrote {out}")

# ---- the numbers quoted in section 6.7 -------------------------------
for v in VARIANTS:
    r = stacked(v, "episode_reward_mean")
    n = normalised_fitness(v)
    d = stacked(v, "episode_q_delta")
    q = [x["q_table"] for x in data[v]]
    post = episodes >= eps_floor
    greedy = [t.argmax(1) for t in q]
    agree = np.mean([np.mean(greedy[i] == greedy[j]) for i in range(3) for j in range(i + 1, 3)])
    print(f"\n{v}")
    print(f"  reward       first 50 {r[:, :50].mean():+.4f} -> last 50 {r[:, -50:].mean():+.4f}")
    print(f"  reward trend corr(episode, reward) after epsilon floor: "
          f"{np.round([np.corrcoef(episodes[post], a[post])[0, 1] for a in r], 3)}")
    print(f"  normalised best fitness first 50 {n[:, :50].mean():.4f} -> last 50 {n[:, -50:].mean():.4f}")
    print(f"  ||dQ|| per episode  first 10 {d[:, :10].mean():.4f} -> last 10 {d[:, -10:].mean():.4f}")
    print(f"  trained within-state spread {np.mean([(t.max(1) - t.min(1)).mean() for t in q]):.4f}"
          f" | greedy agreement across seeds {agree:.3f} (chance {1 / len(ACTIONS):.3f})")
