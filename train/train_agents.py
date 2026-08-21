"""
Train the Q-learning controllers, with independent repetitions.

The state discretisation comes from ``state_bins.json`` (calibrated on the
training split only). The complete training procedure is repeated with
several independent seeds (agent RNG, problem order, GA seeds); policy
stability across those repetitions is reported, and evaluation runs rotate
through the independently trained agents rather than relying on a single
one. The reward per training episode is recorded and plotted for judging
agent convergence.

Training uses the *training* split only (15 instances, seeds 2000-2014); the
test instances are never seen during training.

Run:  python -m train.train_agents
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.problems import build_instances
from optimization.ga_core import ConfigurableGA, make_initial_population
from optimization.rl_control import QLearningAgent, StateBins, policy_agreement
from optimization.variants import DEFAULT_CONTROL_INTERVAL, resolve
from paths import DEFAULT_AGENT_DIR
from train.calibrate_state_bins import DEFAULT_OUTPUT as DEFAULT_BINS_PATH

AGENT_DIR = DEFAULT_AGENT_DIR
DEFAULT_VARIANTS: Tuple[str, ...] = ("rlga_offspring", "rlga_elites")
DEFAULT_AGENT_SEEDS: Tuple[int, ...] = (0, 1, 2)


def agent_path(variant_key: str, agent_seed: int, directory: str = AGENT_DIR) -> str:
    return os.path.join(directory, f"{variant_key}_seed{agent_seed}.npz")


def build_training_schedule(
    n_instances: int, runs_per_instance: int, rng: random.Random
) -> List[int]:
    """``runs_per_instance`` shuffled passes over the instance set.

    Every instance is trained on exactly ``runs_per_instance`` times, never
    twice in a row (including across pass boundaries), so the learned policy
    is less dependent on problem order.
    """
    schedule: List[int] = []
    for _ in range(runs_per_instance):
        block = list(range(n_instances))
        rng.shuffle(block)
        if schedule and len(block) > 1 and block[0] == schedule[-1]:
            swap = rng.randrange(1, len(block))
            block[0], block[swap] = block[swap], block[0]
        schedule.extend(block)
    return schedule


def train_one_agent(
    variant_key: str,
    agent_seed: int,
    runs_per_instance: int = 10,
    split: str = "train",
    bins_path: str = DEFAULT_BINS_PATH,
    generations: int = 300,
    population_size: int = 200,
    control_interval: int = DEFAULT_CONTROL_INTERVAL,
    epsilon_start: float = 0.30,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.975,
    learning_rate: float = 0.05,
    directory: str = AGENT_DIR,
    quiet: bool = False,
) -> Dict:
    """Train a single agent end-to-end and persist it.

    ``epsilon`` decays once per GA run (episode), from ``epsilon_start`` to
    ``epsilon_min``; the Q-learning rule itself is unchanged from the original
    implementation.
    """
    config = resolve(
        [variant_key],
        generations=generations,
        population_size=population_size,
        control_interval=control_interval,
    )[variant_key]
    bins = StateBins.load(bins_path)
    instances = build_instances(split)
    action_set = config.action_set()

    agent = QLearningAgent(
        n_states=bins.n_states,
        n_actions=len(action_set),
        lr=learning_rate,
        epsilon=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        seed=10_000 + agent_seed,
    )

    rng = random.Random(20_000 + agent_seed)
    schedule = build_training_schedule(len(instances), runs_per_instance, rng)

    episode_instance: List[int] = []
    episode_reward_mean: List[float] = []
    episode_reward_sum: List[float] = []
    episode_best_fitness: List[float] = []
    episode_q_delta: List[float] = []
    episode_epsilon: List[float] = []
    episode_evaluations: List[int] = []
    episode_ls_rate_mean: List[float] = []
    action_counts = np.zeros((bins.n_states, len(action_set)), dtype=int)

    started = time.perf_counter()
    for episode, instance_index in enumerate(schedule):
        instance = instances[instance_index]
        run_seed = 100_000 + agent_seed * 1000 + episode

        q_before = agent.q_table.copy()
        ga = ConfigurableGA(
            instance.machines,
            instance.sequence,
            instance.robot_position,
            instance.workspace_bounds,
            config=config,
            rl_agent=agent,
            state_bins=bins,
            seed=run_seed,
            instance_name=instance.name,
        )
        population = make_initial_population(
            instance.machines, instance.workspace_bounds, population_size, seed=run_seed
        )
        result = ga.optimize(initial_population=population)

        # Rewards that actually drove a Q-update: generation g's reward is
        # credited to the action taken at g-1.
        credited = [
            float(result.rl_reward[g])
            for g in range(1, len(result.rl_reward))
            if result.rl_action[g - 1] >= 0
        ]
        for state, action in zip(result.rl_state, result.rl_action):
            if state >= 0 and action >= 0:
                action_counts[state, action] += 1

        episode_instance.append(instance_index)
        episode_reward_mean.append(float(np.mean(credited)) if credited else 0.0)
        episode_reward_sum.append(float(np.sum(credited)) if credited else 0.0)
        episode_best_fitness.append(float(result.best_fitness))
        episode_q_delta.append(float(np.linalg.norm(agent.q_table - q_before)))
        episode_epsilon.append(float(agent.epsilon))
        episode_evaluations.append(int(result.total_evaluations))
        episode_ls_rate_mean.append(float(np.nanmean(result.ls_rate)))

        agent.decay_epsilon()

        if not quiet and (episode + 1) % len(instances) == 0:
            done = episode + 1
            print(
                f"  [{variant_key} seed {agent_seed}] episode {done}/{len(schedule)} "
                f"mean reward {np.mean(episode_reward_mean[-len(instances):]):+.4f} "
                f"best fitness {np.mean(episode_best_fitness[-len(instances):]):.3f} "
                f"eps {agent.epsilon:.3f} "
                f"({time.perf_counter() - started:.0f}s)"
            )

    duration = time.perf_counter() - started

    os.makedirs(directory, exist_ok=True)
    out_path = agent_path(variant_key, agent_seed, directory)
    agent.save(
        out_path,
        extra={
            "variant_key": variant_key,
            "agent_seed": agent_seed,
            "split": split,
            "runs_per_instance": runs_per_instance,
            "control_interval": control_interval,
            "generations": generations,
            "population_size": population_size,
            # Two agents trained with different stall penalties are otherwise
            # indistinguishable once saved.
            "rl_stall_penalty": config.rl_stall_penalty,
            "state_bins": json.dumps(
                {
                    "diversity_edges": list(bins.diversity_edges),
                    "improvement_edge": bins.improvement_edge,
                    "source": bins.source,
                }
            ),
            "episode_instance": np.array(episode_instance),
            "episode_reward_mean": np.array(episode_reward_mean),
            "episode_reward_sum": np.array(episode_reward_sum),
            "episode_best_fitness": np.array(episode_best_fitness),
            "episode_q_delta": np.array(episode_q_delta),
            "episode_epsilon": np.array(episode_epsilon),
            "episode_evaluations": np.array(episode_evaluations),
            "episode_ls_rate_mean": np.array(episode_ls_rate_mean),
            "action_counts": action_counts,
            "action_names": np.array(action_set, dtype=object),
            "training_seconds": duration,
        },
    )

    if not quiet:
        print(f"  [{variant_key} seed {agent_seed}] saved -> {out_path} ({duration:.0f}s)")

    return {
        "variant_key": variant_key,
        "agent_seed": agent_seed,
        "path": out_path,
        "q_table": agent.q_table,
        "episode_reward_mean": np.array(episode_reward_mean),
        "episode_best_fitness": np.array(episode_best_fitness),
        "training_seconds": duration,
    }


def _job(kwargs: Dict) -> Dict:
    return train_one_agent(**kwargs)


def train_all(
    variant_keys: Sequence[str] = DEFAULT_VARIANTS,
    agent_seeds: Sequence[int] = DEFAULT_AGENT_SEEDS,
    runs_per_instance: int = 10,
    workers: int = 0,
    directory: str = AGENT_DIR,
    bins_path: str = DEFAULT_BINS_PATH,
    generations: int = 300,
    population_size: int = 200,
    control_interval: int = DEFAULT_CONTROL_INTERVAL,
    make_plots: bool = True,
) -> Dict[str, Dict]:
    """Train every (variant, seed) combination, then report policy stability."""
    jobs = [
        dict(
            variant_key=variant_key,
            agent_seed=agent_seed,
            runs_per_instance=runs_per_instance,
            directory=directory,
            bins_path=bins_path,
            generations=generations,
            population_size=population_size,
            control_interval=control_interval,
            quiet=True,
        )
        for variant_key in variant_keys
        for agent_seed in agent_seeds
    ]
    workers = workers or min(len(jobs), max(1, (os.cpu_count() or 2) - 2))

    print(
        f"Training {len(variant_keys)} variant(s) x {len(agent_seeds)} independent seeds "
        f"= {len(jobs)} agents, {runs_per_instance} runs per training instance "
        f"({workers} workers)"
    )
    started = time.perf_counter()

    results: List[Dict] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for k, outcome in enumerate(pool.map(_job, jobs), start=1):
                results.append(outcome)
                print(
                    f"  agent {k}/{len(jobs)} done: {outcome['variant_key']} "
                    f"seed {outcome['agent_seed']} ({outcome['training_seconds']:.0f}s)"
                )
    else:
        for job in jobs:
            results.append(train_one_agent(**{**job, "quiet": False}))

    print(f"Training finished in {time.perf_counter() - started:.0f}s")

    # ---- policy stability across independent repetitions ------------------
    stability: Dict[str, Dict] = {}
    for variant_key in variant_keys:
        tables = [r["q_table"] for r in results if r["variant_key"] == variant_key]
        stability[variant_key] = policy_agreement(tables)
        print(
            f"  {variant_key}: greedy-policy agreement across seeds "
            f"{stability[variant_key]['mean_policy_agreement']:.3f}, "
            f"Q-table correlation {stability[variant_key]['mean_q_correlation']:.3f}"
        )

    # Merge rather than overwrite: training the main and the ablation agents
    # are separate invocations that both report into this file.
    stability_path = os.path.join(directory, "policy_stability.json")
    merged: Dict[str, Dict] = {}
    if os.path.exists(stability_path):
        try:
            with open(stability_path, encoding="utf-8") as fh:
                merged = json.load(fh)
        except json.JSONDecodeError:
            merged = {}
    merged.update(stability)
    with open(stability_path, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)
    print(f"  policy stability written to {stability_path}")

    if make_plots:
        from visualization.plots import plot_training_summary

        for variant_key in variant_keys:
            paths = [agent_path(variant_key, s, directory) for s in agent_seeds]
            plot_training_summary(
                paths,
                save_path=os.path.join(directory, f"training_{variant_key}.png"),
                title=variant_key,
            )
        print(f"  training figures written to {directory}/")

    return {r["variant_key"] + f"_seed{r['agent_seed']}": r for r in results}


def load_agents(
    variant_key: str,
    agent_seeds: Sequence[int] = DEFAULT_AGENT_SEEDS,
    directory: str = AGENT_DIR,
    epsilon: float = 0.05,
) -> List[QLearningAgent]:
    """Load the independently trained agents of one variant, in seed order."""
    agents = []
    for agent_seed in agent_seeds:
        path = agent_path(variant_key, agent_seed, directory)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"trained agent {path} not found - run `python -m train.train_agents` first"
            )
        agents.append(QLearningAgent.load(path, epsilon=epsilon, seed=agent_seed))
    return agents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--agent-seeds", type=int, nargs="+", default=list(DEFAULT_AGENT_SEEDS))
    parser.add_argument("--runs-per-instance", type=int, default=10)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--population-size", type=int, default=200)
    parser.add_argument(
        "--control-interval",
        type=int,
        default=DEFAULT_CONTROL_INTERVAL,
        help="generations between RL control decisions during training",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--directory", default=AGENT_DIR)
    parser.add_argument("--bins", default=DEFAULT_BINS_PATH)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    train_all(
        variant_keys=args.variants,
        agent_seeds=args.agent_seeds,
        runs_per_instance=args.runs_per_instance,
        workers=args.workers,
        directory=args.directory,
        bins_path=args.bins,
        generations=args.generations,
        population_size=args.population_size,
        control_interval=args.control_interval,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
