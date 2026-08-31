"""
Q-learning controller for GA parameter control, plus the state discretisation.

Covers the tabular Q-learning agent (7 actions over local-search rate,
mutation rate and crossover rate), its reward which may include a cost term
that charges for the extra evaluations local search consumes - and :class:`StateBins`, the
diversity/improvement discretisation whose thresholds are calibrated on the
training split through ``calibrate_state_bins.py``.

State space (9 states)
----------------------
``state = 3 * diversity_bin + improvement_bin`` where the diversity bin comes
from the calibrated tertiles of the population-diversity distribution and the
improvement bin from the sign / magnitude of the *relative* best-fitness
improvement of the previous generation.  Relative improvement is used instead
of the absolute one so the same thresholds transfer across instances whose
fitness scale differs.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# actions
# ----------------------------------------------------------------------

#: Actions shared by every RL-GA variant.  Index == action id.
BASE_ACTIONS: Tuple[str, ...] = (
    "mutation_rate_up",
    "mutation_rate_down",
    "crossover_rate_up",
    "crossover_rate_down",
    "local_search_rate_up",
    "local_search_rate_down",
    "soft_reset",
)

# ----------------------------------------------------------------------
# state discretisation
# ----------------------------------------------------------------------


@dataclass
class StateBins:
    """Discretisation thresholds for the (diversity, improvement) state.

    ``diversity_edges`` are the two cut points of the diversity tertiles and
    ``improvement_edge`` separates "small" from "large" relative improvement.
    ``source`` records how they were obtained, so a Q-table can never be used
    with thresholds of unknown provenance.
    """

    diversity_edges: Tuple[float, float] = (300.0, 400.0)
    improvement_edge: float = 0.01
    source: str = "legacy hard-coded values"

    n_diversity_bins: int = 3
    n_improvement_bins: int = 3

    @property
    def n_states(self) -> int:
        return self.n_diversity_bins * self.n_improvement_bins

    def state(self, diversity: float, relative_improvement: float) -> int:
        """Map a (diversity, relative improvement) pair to a state index."""
        lo, hi = self.diversity_edges
        if diversity < lo:
            diversity_bin = 0  # converging
        elif diversity < hi:
            diversity_bin = 1  # stable
        else:
            diversity_bin = 2  # exploring

        if not np.isfinite(relative_improvement) or relative_improvement <= 0.0:
            improvement_bin = 0  # stalled or worse
        elif relative_improvement < self.improvement_edge:
            improvement_bin = 1  # small gain
        else:
            improvement_bin = 2  # large gain

        return diversity_bin * self.n_improvement_bins + improvement_bin

    def label(self, state: int) -> str:
        div = ("div:low", "div:med", "div:high")[state // self.n_improvement_bins]
        imp = ("imp:none", "imp:small", "imp:large")[state % self.n_improvement_bins]
        return f"{div}/{imp}"

    # -- persistence ---------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "StateBins":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        data["diversity_edges"] = tuple(data["diversity_edges"])
        return cls(**data)


# ----------------------------------------------------------------------
# agent
# ----------------------------------------------------------------------


@dataclass
class QLearningAgent:
    """Tabular Q-learning agent with epsilon-greedy action selection.

    Identical learning rule to the original ``algo.agent.QLearningAgent``; the
    additions are a configurable action set, an optional epsilon decay for
    training, a private RNG (so parallel runs are reproducible) and metadata
    that travels with the Q-table.
    """

    n_states: int = 9
    n_actions: int = len(BASE_ACTIONS)
    lr: float = 0.01
    gamma: float = 0.95
    epsilon: float = 0.05
    epsilon_min: float = 0.01
    epsilon_decay: float = 1.0  # 1.0 == no decay (evaluation default)
    seed: Optional[int] = None
    q_table: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.q_table is None:
            self.q_table = np.zeros((self.n_states, self.n_actions))
        self.q_table = np.asarray(self.q_table, dtype=float)
        if self.q_table.shape != (self.n_states, self.n_actions):
            raise ValueError(
                f"Q-table shape {self.q_table.shape} does not match "
                f"({self.n_states}, {self.n_actions}). A Q-table trained with a "
                "different action set cannot be reused."
            )
        self._rng = np.random.default_rng(self.seed)
        self.n_updates = 0

    # -- policy --------------------------------------------------------

    def get_action(self, state: int) -> int:
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        # Ties are broken randomly rather than by np.argmax's first-index rule,
        # so an untouched all-zero row does not always return action 0.
        row = self.q_table[state]
        best = np.flatnonzero(row == row.max())
        return int(best[self._rng.integers(best.size)])

    def update(self, state: int, action: int, reward: float, next_state: int) -> float:
        current_q = self.q_table[state, action]
        max_next_q = float(np.max(self.q_table[next_state]))
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state, action] = new_q
        self.n_updates += 1
        return float(new_q - current_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def greedy_policy(self) -> np.ndarray:
        """Greedy action per state (used for the policy-stability report)."""
        return np.argmax(self.q_table, axis=1)

    def copy(self, seed: Optional[int] = None) -> "QLearningAgent":
        clone = QLearningAgent(
            n_states=self.n_states,
            n_actions=self.n_actions,
            lr=self.lr,
            gamma=self.gamma,
            epsilon=self.epsilon,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
            seed=seed,
            q_table=self.q_table.copy(),
        )
        return clone

    # -- persistence ---------------------------------------------------

    def save(self, path: str, extra: Optional[Dict] = None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "q_table": self.q_table,
            "n_states": self.n_states,
            "n_actions": self.n_actions,
            "lr": self.lr,
            "gamma": self.gamma,
            "action_names": np.array(BASE_ACTIONS, dtype=object),
        }
        if extra:
            payload.update(extra)
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: str, epsilon: float = 0.05, seed: Optional[int] = None) -> "QLearningAgent":
        data = np.load(path, allow_pickle=True)
        q_table = data["q_table"]
        return cls(
            n_states=int(q_table.shape[0]),
            n_actions=int(q_table.shape[1]),
            lr=float(data["lr"]) if "lr" in data.files else 0.01,
            gamma=float(data["gamma"]) if "gamma" in data.files else 0.95,
            epsilon=epsilon,
            seed=seed,
            q_table=q_table,
        )


# ----------------------------------------------------------------------
# Q-table / policy stability
# ----------------------------------------------------------------------


def policy_agreement(q_tables: Sequence[np.ndarray]) -> Dict[str, float]:
    """Stability of independently trained agents.

    Returns the mean pairwise fraction of states on which the greedy actions
    agree, and the mean pairwise Pearson correlation of the flattened
    Q-tables.  Both are 1.0 for identical agents.
    """
    tables = [np.asarray(q) for q in q_tables]
    if len(tables) < 2:
        return {"mean_policy_agreement": float("nan"), "mean_q_correlation": float("nan")}

    agreements: List[float] = []
    correlations: List[float] = []
    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            pi = np.argmax(tables[i], axis=1)
            pj = np.argmax(tables[j], axis=1)
            agreements.append(float(np.mean(pi == pj)))
            a = tables[i].ravel()
            b = tables[j].ravel()
            if a.std() > 0 and b.std() > 0:
                correlations.append(float(np.corrcoef(a, b)[0, 1]))
    return {
        "mean_policy_agreement": float(np.mean(agreements)),
        "min_policy_agreement": float(np.min(agreements)),
        "mean_q_correlation": float(np.mean(correlations)) if correlations else float("nan"),
        "n_pairs": float(len(agreements)),
    }
