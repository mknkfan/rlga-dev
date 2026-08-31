"""
The algorithm variants that are compared, in one place.

Defines the main grid as the cross product of ``{static, scheduled, RL}``
parameter control with ``{local search on offspring, local search on
elites, no local search}``, plus one extra registry: ``ablation_variants``
restricts the RL agent to controlling one parameter at a time. Each variant
resolves to a :class:`~optimization.ga_core.GAConfig`.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from optimization.ga_core import DEFAULT_LS_SCHEDULE, GAConfig

#: Human-readable names for the two local-search targets.
LS_TARGETS: Tuple[str, ...] = ("offspring", "elites")

#: How often the RL controller is queried, in generations.  Read from
#: :class:`GAConfig` so the two cannot drift apart; override per run with the
#: ``--control-interval`` flag on the training and experiment entry points.
DEFAULT_CONTROL_INTERVAL: int = GAConfig().control_interval


def _label(controller: str, target: str) -> str:
    target_suffix = {"offspring": "offspring", "elites": "elites", "none": "no LS"}[target]
    return f"{controller} [LS on {target_suffix}]"


def main_variants(
    generations: int = 300,
    population_size: int = 200,
    control_interval: int = DEFAULT_CONTROL_INTERVAL,
) -> Dict[str, GAConfig]:
    """The variants reported in the main comparison table."""
    shared = dict(
        generations=generations,
        population_size=population_size,
        control_interval=control_interval,
    )
    variants: Dict[str, GAConfig] = {}

    # Reference point: no local search at all.  Quantifies how much of the
    # performance is due to local search rather than to parameter control.
    variants["ga_ls00"] = GAConfig(
        label="GA no LS", ls_target="none", ls_controller="static", ls_rate=0.0, **shared
    )

    for target in LS_TARGETS:
        for rate, tag in ((0.1, "ls10"), (0.5, "ls50"), (1.0, "ls100")):
            key = f"ga_{tag}_{target}"
            variants[key] = GAConfig(
                label=_label(f"GA {int(rate * 100)}% LS", target),
                ls_target=target,
                ls_controller="static",
                ls_rate=rate,
                **shared,
            )

        variants[f"ga_sched_{target}"] = GAConfig(
            label=_label("GA scheduled LS", target),
            ls_target=target,
            ls_controller="schedule",
            ls_schedule=DEFAULT_LS_SCHEDULE,
            **shared,
        )

        variants[f"rlga_{target}"] = GAConfig(
            label=_label("RL-GA", target),
            ls_target=target,
            ls_controller="rl",
            ls_rate=0.1,  # the agent starts from the baseline rate
            **shared,
        )

    return variants


#: The ablation is run on the RL-GA's primary configuration only (local search
#: on elites). Running it on both targets doubles the number of agents that
#: have to be trained without changing what the ablation answers.
ABLATION_TARGETS: Tuple[str, ...] = ("elites",)


def ablation_variants(
    generations: int = 300,
    population_size: int = 200,
    targets: Sequence[str] = ABLATION_TARGETS,
    control_interval: int = DEFAULT_CONTROL_INTERVAL,
) -> Dict[str, GAConfig]:
    """Per-parameter ablations of the RL controller."""
    shared = dict(
        generations=generations,
        population_size=population_size,
        control_interval=control_interval,
    )
    subsets = {
        "lsonly": ("local_search_rate_up", "local_search_rate_down"),
        "mutonly": ("mutation_rate_up", "mutation_rate_down"),
        "xoveronly": ("crossover_rate_up", "crossover_rate_down"),
        "noLS": (
            "mutation_rate_up",
            "mutation_rate_down",
            "crossover_rate_up",
            "crossover_rate_down",
            "soft_reset",
        ),
    }
    pretty = {
        "lsonly": "RL-GA (LS rate only)",
        "mutonly": "RL-GA (mutation rate only)",
        "xoveronly": "RL-GA (crossover rate only)",
        "noLS": "RL-GA (no LS-rate control)",
    }

    variants: Dict[str, GAConfig] = {}
    for target in targets:
        for tag, actions in subsets.items():
            variants[f"rlga_{tag}_{target}"] = GAConfig(
                label=_label(pretty[tag], target),
                ls_target=target,
                ls_controller="rl",
                ls_rate=0.1,
                allowed_actions=actions,
                **shared,
            )
    return variants


def resolve(
    keys: List[str],
    generations: int = 300,
    population_size: int = 200,
    control_interval: int = DEFAULT_CONTROL_INTERVAL,
) -> Dict[str, GAConfig]:
    """Look up variant configurations by key, across every registry."""
    shared = dict(
        generations=generations,
        population_size=population_size,
        control_interval=control_interval,
    )
    registry = main_variants(**shared)
    registry.update(ablation_variants(**shared))
    missing = [k for k in keys if k not in registry]
    if missing:
        raise KeyError(f"unknown variant(s): {missing}. Available: {sorted(registry)}")
    return {k: registry[k] for k in keys}
