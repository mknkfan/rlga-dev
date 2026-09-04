"""
The equal-compute control for the placement-repair A/B.

    python -m arm_study.equal_cpu_check

``run_repair_ab`` gives both arms the same number of evaluations, and on that
budget the repair wins.  The budget is the problem: an infeasible layout is
thrown out by the placement test for about 0.36 ms, while a feasible one runs
inverse kinematics, the manipulability check and the swept-path collision test
over four branches for about 6.1 ms.  Repair works by turning the cheap
evaluations into expensive ones, so at equal evaluation count it quietly draws
about 1.7x the compute.  A win bought that way is a weaker claim than it looks.

So this re-runs the comparison with the handicap moved to the other side.  The
repair arm is truncated to the generation count whose CPU matches what the
baseline spent over its full 300 -- 174 generations, from the measured ratio --
and is then compared against the *full-length* baseline.  Truncation is the
conservative direction twice over: a GA's later generations are the expensive
ones, because the population is by then mostly feasible, so the first 174 of
300 cost rather less than 58 % of the total.  The repair arm is therefore
given, if anything, slightly less compute than the baseline rather than more,
and the realised CPU is reported so the reader can check that rather than take
it on trust.

Both studies must already be on disk::

    results_repair/raw            baseline and repair at 300 generations
    results_repair_equalcpu/raw   both arms at 174 generations

Pairing is by (instance, seed) exactly as before, and the seeds match across
the two studies because both use the same ``run_seed_for`` rule.
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, List, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_repair_ab import (  # noqa: E402
    bootstrap_ci, collect_records, mcnemar_exact, sign_flip_test,
    wilcoxon_signed_rank,
)

FULL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_repair")
EQUAL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results_repair_equalcpu"
)


def index_by_arm(records: Sequence[Dict], arm: str) -> Dict:
    return {(r["instance"], r["run"]): r for r in records if r["arm"] == arm}


def paired(baseline: Dict, repair: Dict, metric: str) -> List[float]:
    """Signed differences on one metric, positive where repair did better."""
    out = []
    for key, base in baseline.items():
        other = repair.get(key)
        if other is None:
            continue
        a, b = base.get(metric), other.get(metric)
        if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
            continue
        out.append(float(a) - float(b))
    return out


def describe(name: str, deltas: Sequence[float], unit: str, spec: str = ".4f") -> Dict:
    d = np.asarray(deltas, dtype=float)
    low, high = bootstrap_ci(d)
    p = sign_flip_test(d)
    _, wp = wilcoxon_signed_rank(d)
    row = {
        "metric": name, "n_pairs": int(d.size),
        "mean_gain": float(d.mean()), "median_gain": float(np.median(d)),
        "ci_low": low, "ci_high": high,
        "repair_wins": int((d > 0).sum()), "baseline_wins": int((d < 0).sum()),
        "permutation_p": p, "wilcoxon_p": wp,
    }
    verdict = ("repair better" if row["mean_gain"] > 0 else "repair worse")
    if p is not None and p > 0.05:
        verdict = "no detectable difference"
    print(f"  {name:<16} {row['mean_gain']:+{spec[1:]}} {unit:<6} "
          f"[{low:{spec}}, {high:{spec}}]  "
          f"wins {row['repair_wins']:>3}/{row['repair_wins'] + row['baseline_wins']:<3} "
          f"p={p:.4g}  -- {verdict}")
    return row


def main() -> None:
    for path in (FULL_DIR, EQUAL_DIR):
        if not os.path.isdir(os.path.join(path, "raw")):
            raise SystemExit(f"missing study: {path}")

    full = collect_records(FULL_DIR)
    equal = collect_records(EQUAL_DIR)
    baseline = index_by_arm(full, "baseline")          # 300 generations
    repair_full = index_by_arm(full, "repair")         # 300 generations
    repair_eq = index_by_arm(equal, "repair")          # 174 generations

    print(f"baseline @300 gens : {len(baseline)} runs")
    print(f"repair   @300 gens : {len(repair_full)} runs")
    print(f"repair   @174 gens : {len(repair_eq)} runs")

    def mean_of(index: Dict, key: str) -> float:
        return float(np.mean([r[key] for r in index.values()]))

    base_cpu = mean_of(baseline, "cpu_time")
    eq_cpu = mean_of(repair_eq, "cpu_time")
    print(f"\nRealised CPU per run: baseline {base_cpu:.1f} s, "
          f"truncated repair {eq_cpu:.1f} s "
          f"(repair is at {100 * eq_cpu / base_cpu:.1f} % of the baseline's compute)")
    if eq_cpu > base_cpu:
        print("  NOTE: the truncated repair still spent MORE cpu than the "
              "baseline, so the control is not yet conservative -- treat a "
              "win below as suggestive only.")
    print(f"Evaluations per run:  baseline {mean_of(baseline, 'evaluations'):.0f}, "
          f"truncated repair {mean_of(repair_eq, 'evaluations'):.0f}")

    print("\nEqual compute -- baseline @300 gens vs repair @174 gens")
    print("  (positive gain = repair better)")
    rows = [
        describe("cycle_time", paired(baseline, repair_eq, "cycle_time"), "s"),
        describe("mean_feasible",
                 [-d for d in paired(baseline, repair_eq, "mean_feasible")], ""),
        describe("cpu_time", paired(baseline, repair_eq, "cpu_time"), "s", ".2f"),
    ]

    print("\nFor reference -- equal evaluations, baseline vs repair @300 gens")
    reference = describe("cycle_time", paired(baseline, repair_full, "cycle_time"), "s")

    both = sum(1 for k, r in repair_eq.items()
               if k in baseline and r["feasible"] and baseline[k]["feasible"])
    only_b = sum(1 for k, r in repair_eq.items()
                 if k in baseline and baseline[k]["feasible"] and not r["feasible"])
    only_r = sum(1 for k, r in repair_eq.items()
                 if k in baseline and r["feasible"] and not baseline[k]["feasible"])
    print(f"\nFeasible at all: both {both}, baseline only {only_b}, "
          f"repair only {only_r}, McNemar p = {mcnemar_exact(only_b, only_r)}")

    out = {
        "baseline_generations": 300,
        "repair_generations": 174,
        "baseline_cpu_mean": base_cpu,
        "repair_cpu_mean": eq_cpu,
        "cpu_ratio": eq_cpu / base_cpu,
        "equal_compute": rows,
        "equal_evaluations_reference": reference,
        "feasible": {"both": both, "baseline_only": only_b, "repair_only": only_r},
    }
    path = os.path.join(EQUAL_DIR, "equal_cpu_check.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(f"\nwritten -> {path}")


if __name__ == "__main__":
    main()
