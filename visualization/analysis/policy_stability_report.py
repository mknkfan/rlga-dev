"""
Report how stable the learned policies are across independent training repetitions.

The training script repeats the complete procedure with independent seeds
(agent RNG, problem order, GA seeds); this script reads the resulting
Q-tables back and quantifies their agreement:

* **greedy-policy agreement** - fraction of states on which two independently
  trained agents pick the same action (1.0 = identical policies);
* **Q-table correlation** - Pearson correlation of the flattened Q-tables;
* **per-state agreement** - which states are decided consistently and which
  are not, so an unstable state can be discussed rather than hidden.

It also rebuilds ``policy_stability.json`` from whatever agents are on disk,
which is what makes it safe to train different agent families in separate
(possibly concurrent) invocations.

Run:  python -m analysis.policy_stability_report
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, List

import numpy as np

from optimization.rl_control import policy_agreement
from train.train_agents import AGENT_DIR

_NAME_PATTERN = re.compile(r"^(?P<variant>.+)_seed(?P<seed>\d+)\.npz$")


def load_q_tables(directory: str = AGENT_DIR) -> Dict[str, Dict[int, np.ndarray]]:
    """Q-tables on disk, grouped by variant and keyed by agent seed."""
    tables: Dict[str, Dict[int, np.ndarray]] = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(directory, "*.npz"))):
        match = _NAME_PATTERN.match(os.path.basename(path))
        if not match:
            continue
        with np.load(path, allow_pickle=True) as data:
            if "q_table" not in data.files:
                continue
            tables[match.group("variant")][int(match.group("seed"))] = data["q_table"]
    return dict(tables)


def report(directory: str = AGENT_DIR, write: bool = True) -> Dict[str, Dict]:
    tables = load_q_tables(directory)
    if not tables:
        raise SystemExit(f"no trained agents found in {directory}")

    stability: Dict[str, Dict] = {}
    for variant, by_seed in sorted(tables.items()):
        seeds = sorted(by_seed)
        stacked = [by_seed[s] for s in seeds]
        summary = dict(policy_agreement(stacked))
        summary["agent_seeds"] = [float(s) for s in seeds]

        # Per-state agreement: on how many of the pairwise comparisons do the
        # independently trained agents choose the same greedy action?
        greedy = np.stack([np.argmax(q, axis=1) for q in stacked])
        per_state = []
        for state in range(greedy.shape[1]):
            actions = greedy[:, state]
            matches = sum(
                1
                for i in range(len(actions))
                for j in range(i + 1, len(actions))
                if actions[i] == actions[j]
            )
            pairs = len(actions) * (len(actions) - 1) / 2
            per_state.append(matches / pairs if pairs else float("nan"))
        summary["per_state_agreement"] = [float(v) for v in per_state]
        summary["greedy_actions_per_seed"] = greedy.tolist()
        stability[variant] = summary

    if write:
        path = os.path.join(directory, "policy_stability.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(stability, fh, indent=2)
        print(f"Written to {path}\n")

    for variant, summary in stability.items():
        print(
            f"{variant:<26} agents={len(summary['agent_seeds'])} "
            f"policy agreement {summary['mean_policy_agreement']:.3f} "
            f"(min pair {summary['min_policy_agreement']:.3f}), "
            f"Q correlation {summary['mean_q_correlation']:+.3f}"
        )
        consistent = sum(1 for v in summary["per_state_agreement"] if v == 1.0)
        print(
            f"{'':<26} states decided identically by every repetition: "
            f"{consistent}/{len(summary['per_state_agreement'])}"
        )
    return stability


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default=AGENT_DIR)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    report(directory=args.directory, write=not args.no_write)


if __name__ == "__main__":
    main()
