"""Single-instance test run reproducing the archived study's protocol, plus a layout figure.

Instance test_1 (index 0), run 0 -> run_seed = 42 + 100*0 + 0 = 42, agent_seed = 0 % 3 = 0.
Everything else (300 gens, pop 200, control_interval 1, the shipped agents in
results/agents, train-calibrated bins) matches results/runs_metadata.json.

Prints the fresh fitness beside the value stored in runs.csv for the same
(variant, instance, seed) -- they should agree exactly -- and writes a figure of
the final layouts.

Run:  python visualization/extra/test_run_layout.py [output.png]
      (paths are anchored to the project's root directory, so any working directory works)
"""
from __future__ import annotations

import csv
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from optimization.ga_core import ConfigurableGA, make_initial_population
from core.problems import build_instances
from optimization.rl_control import QLearningAgent, StateBins
from optimization.variants import resolve
from train.train_agents import agent_path
from run.paths import DEFAULT_AGENT_DIR, DEFAULT_BINS_PATH, DEFAULT_RESULTS_DIR, EXTRA_FIGURES_DIR

RESULTS = DEFAULT_RESULTS_DIR
AGENT_DIR = DEFAULT_AGENT_DIR
BINS = DEFAULT_BINS_PATH
INSTANCE_INDEX, RUN = 0, 0
SEED = 42 + 100 * INSTANCE_INDEX + RUN
AGENT_SEEDS = (0, 1, 2)
VARIANTS = ["ga_ls00", "ga_ls10_offspring", "ga_ls50_elites", "rlga_elites", "rlga_offspring"]

instance = build_instances("test")[INSTANCE_INDEX]
configs = resolve(VARIANTS, generations=300, population_size=200, control_interval=1)

stored = {}
for row in csv.DictReader(open(os.path.join(RESULTS, "runs.csv"))):
    if row["instance"] == instance.name and int(row["seed"]) == RUN:
        stored[row["variant"]] = row

print(f"instance {instance.name}: {instance.n_machines} machines, "
      f"bounds {instance.workspace_bounds}, run_seed {SEED}\n")
print(f"{'variant':22}{'fitness':>10}{'stored':>10}{'match':>7}{'evals':>9}{'tour':>8}{'sec':>7}")

results = {}
for key in VARIANTS:
    config = configs[key]
    agent = bins = None
    if config.ls_controller == "rl":
        bins = StateBins.load(BINS)
        agent = QLearningAgent.load(
            agent_path(key, AGENT_SEEDS[RUN % len(AGENT_SEEDS)], AGENT_DIR),
            epsilon=0.05, seed=SEED,
        )
        agent.lr = 0.05

    ga = ConfigurableGA(
        instance.machines, instance.sequence, instance.robot_position,
        instance.workspace_bounds, config=config, rl_agent=agent, state_bins=bins,
        seed=SEED, instance_name=instance.name,
    )
    population = make_initial_population(
        instance.machines, instance.workspace_bounds, config.population_size, seed=SEED
    )
    started = time.perf_counter()
    result = ga.optimize(initial_population=population)
    results[key] = result

    ref = float(stored[key]["best_fitness"])
    ok = "yes" if abs(result.best_fitness - ref) < 1e-9 else "NO"
    print(f"{config.label:22}{result.best_fitness:10.4f}{ref:10.4f}{ok:>7}"
          f"{result.total_evaluations:9d}{result.total_distance:8.2f}"
          f"{time.perf_counter() - started:7.1f}")

# ----------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------
min_x, max_x, min_y, max_y = instance.workspace_bounds
fig, axes = plt.subplots(1, len(VARIANTS), figsize=(4.4 * len(VARIANTS), 5.0))

for ax, key in zip(np.atleast_1d(axes), VARIANTS):
    result = results[key]
    machines = result.final_machines

    ax.add_patch(plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                               fill=False, ec="#94a3b8", lw=1.2, ls="--"))

    for machine in machines:
        corners = [(p.x, p.y) for p in machine.get_corners()]
        is_l = machine.shape == "l_shape"
        ax.add_patch(Polygon(corners, closed=True,
                             fc="#bfdbfe" if is_l else "#fde68a",
                             ec="#1e3a8a" if is_l else "#92400e", lw=1.3, alpha=0.85))
        ax.text(machine.position.x, machine.position.y, str(machine.id),
                ha="center", va="center", fontsize=9, fontweight="bold", color="#334155")

    # robot tour: origin -> access points in visiting order -> origin
    by_id = {m.id: m for m in machines}
    path = [(instance.robot_position.x, instance.robot_position.y)]
    for machine_id in instance.sequence:
        p = by_id[machine_id].get_access_point_world()
        path.append((p.x, p.y))
    path.append(path[0])
    xs, ys = zip(*path)
    ax.plot(xs, ys, "-", color="#dc2626", lw=1.4, alpha=0.85, zorder=3)
    ax.plot(xs[1:-1], ys[1:-1], "o", color="#dc2626", ms=5, zorder=4)
    ax.plot(0, 0, "*", color="#111827", ms=18, zorder=5)

    ax.set_xlim(min_x - 1, max_x + 1)
    ax.set_ylim(min_y - 1, max_y + 1)
    ax.set_aspect("equal")
    ax.set_title(f"{result.label}\nfitness {result.best_fitness:.2f} | tour "
                 f"{result.total_distance:.1f} | {result.total_evaluations:,} evals",
                 fontsize=10)
    ax.tick_params(labelsize=7)

fig.suptitle(f"{instance.name} (seed {SEED}) — final layouts. "
             "Star = robot at origin, red = tour through access points, "
             "blue = L-shape, amber = rectangle", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
os.makedirs(EXTRA_FIGURES_DIR, exist_ok=True)
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    EXTRA_FIGURES_DIR, "test_run_layouts.png"
)
fig.savefig(out, dpi=130)
print(f"\nwrote {out}")
