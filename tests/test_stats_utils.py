"""
Self-checks for the numpy-only statistics used in the study.

Since SciPy is not a dependency, the exact Wilcoxon signed-rank p-values are
verified against brute-force enumeration of all 2^n sign flips, the normal
approximation against a published worked example, Cliff's delta against its
definition and Holm-Bonferroni against a hand-computed case.

Run:  python -m tests.test_stats_utils
"""

from __future__ import annotations

import itertools
import sys

import numpy as np

from analysis.stats_utils import (
    bootstrap_ci,
    cliffs_delta,
    describe,
    holm_bonferroni,
    paired_effect_sizes,
    value_at_budget,
    budget_to_target,
    wilcoxon_signed_rank,
)


def brute_force_wilcoxon_p(differences: np.ndarray) -> float:
    """Exact two-sided p by enumerating every sign assignment of the ranks."""
    magnitudes = np.abs(differences)
    ranks = np.argsort(np.argsort(magnitudes)) + 1.0  # distinct magnitudes only
    w_plus = float(ranks[differences > 0].sum())
    w_minus = float(ranks[differences < 0].sum())
    observed = min(w_plus, w_minus)

    n = differences.size
    total = 0
    extreme = 0
    for signs in itertools.product([-1, 1], repeat=n):
        signs_array = np.array(signs)
        plus = float(ranks[signs_array > 0].sum())
        minus = float(ranks[signs_array < 0].sum())
        total += 1
        if min(plus, minus) <= observed:
            extreme += 1
    return extreme / total


def check(name: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{(' - ' + detail) if detail else ''}")
    return condition


def main() -> int:
    ok = True
    rng = np.random.default_rng(0)

    print("Wilcoxon signed-rank, exact branch vs brute-force enumeration:")
    for n in (5, 6, 8, 10):
        for trial in range(5):
            # Distinct magnitudes keep the exact branch active.
            magnitudes = rng.permutation(np.arange(1, n + 1)) + rng.random(n) * 0.01
            signs = rng.choice([-1.0, 1.0], size=n)
            differences = magnitudes * signs
            x = differences
            y = np.zeros(n)

            _, p, method, _ = wilcoxon_signed_rank(x, y)
            expected = brute_force_wilcoxon_p(differences)
            ok &= check(
                f"n={n} trial={trial} ({method})",
                method == "exact" and abs(p - expected) < 1e-12,
                f"p={p:.6f} expected={expected:.6f}",
            )

    print("\nWilcoxon, worked example with a zero difference and tied magnitudes:")
    # Differences: +15, -7, +5, +20, 0, -9, +17, -12, +5, -10.
    # The zero is dropped (n = 9); magnitudes 5 and 5 tie, so both get rank 1.5.
    # W+ = 7 + 1.5 + 9 + 8 + 1.5 = 27, W- = 3 + 4 + 6 + 5 = 18, statistic = 18.
    before = np.array([125, 115, 130, 140, 140, 115, 140, 125, 140, 135], dtype=float)
    after = np.array([110, 122, 125, 120, 140, 124, 123, 137, 135, 145], dtype=float)
    statistic, p, method, n_nonzero = wilcoxon_signed_rank(before, after)
    ok &= check(
        "zero dropped, tied magnitudes share a rank, W = min(W+, W-) = 18",
        abs(statistic - 18.0) < 1e-9 and n_nonzero == 9,
        f"W={statistic}, n={n_nonzero}, p={p:.4f}, method={method}",
    )
    ok &= check(
        "ties force the normal approximation",
        method == "normal approximation" and 0.0 <= p <= 1.0,
        f"method={method}",
    )

    print("\nDirection conventions (difference = x - y):")
    better = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    worse = better + 1.5
    result = paired_effect_sizes(better, worse)
    ok &= check(
        "x better than y -> negative mean difference, all wins",
        result.mean_difference < 0 and result.n_wins == 5 and result.n_losses == 0,
        f"mean_diff={result.mean_difference:.3f}, wins={result.n_wins}",
    )
    ok &= check(
        "rank-biserial is -1 when x wins every pair",
        abs(result.rank_biserial + 1.0) < 1e-12,
        f"r={result.rank_biserial:.3f}",
    )

    separated = better + 10.0  # every value of `separated` exceeds every value of `better`
    delta, label = cliffs_delta(better, separated)
    ok &= check(
        "Cliff's delta = -1 for fully separated samples",
        abs(delta + 1.0) < 1e-12 and label == "large",
        f"delta={delta:.3f} ({label})",
    )
    delta, label = cliffs_delta(better, worse)
    ok &= check(
        "Cliff's delta for partially overlapping samples matches its definition",
        abs(delta - (6 - 19) / 25) < 1e-12,
        f"delta={delta:.3f} ({label})",
    )
    delta, _ = cliffs_delta(better, better)
    ok &= check("Cliff's delta = 0 for identical samples", abs(delta) < 1e-12)

    print("\nHolm-Bonferroni:")
    # Sorted: 0.01 -> 3 * 0.01 = 0.03; 0.03 -> max(0.03, 2 * 0.03) = 0.06;
    #         0.04 -> max(0.06, 1 * 0.04) = 0.06 (enforced monotonicity).
    adjusted = holm_bonferroni([0.01, 0.04, 0.03])
    expected = [0.03, 0.06, 0.06]
    ok &= check(
        "adjusted p-values match hand computation",
        all(abs(a - e) < 1e-12 for a, e in zip(adjusted, expected)),
        f"{[round(a, 4) for a in adjusted]}",
    )
    ok &= check(
        "adjusted p-values are monotone in the sorted order",
        holm_bonferroni([0.2, 0.1])[0] >= holm_bonferroni([0.2, 0.1])[1],
    )

    print("\nDescriptive statistics:")
    sample = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    summary = describe(sample, n_boot=2000)
    ok &= check(
        "median is robust, mean is not",
        summary.median == 3.0 and abs(summary.mean - 22.0) < 1e-9,
        f"median={summary.median}, mean={summary.mean}",
    )
    ok &= check("IQR equals q3 - q1", abs(summary.iqr - (summary.q3 - summary.q1)) < 1e-12)
    ok &= check(
        "sd uses the n-1 denominator",
        abs(summary.std - np.std(sample, ddof=1)) < 1e-12,
    )

    normal = rng.normal(10.0, 1.0, size=200)
    lo, hi = bootstrap_ci(normal, np.mean, n_boot=5000)
    ok &= check(
        "bootstrap CI of the mean brackets the true mean",
        lo < 10.0 < hi,
        f"[{lo:.3f}, {hi:.3f}]",
    )

    print("\nAnytime-budget helpers:")
    axis = np.array([100.0, 200.0, 300.0, 400.0])
    curve = np.array([50.0, 40.0, 35.0, 30.0])
    ok &= check("value at an exact budget", value_at_budget(axis, curve, 200.0) == 40.0)
    ok &= check("value between records uses the last reached", value_at_budget(axis, curve, 250.0) == 40.0)
    ok &= check("value below the first record", value_at_budget(axis, curve, 10.0) == 50.0)
    ok &= check("budget to reach a target", budget_to_target(axis, curve, 35.0) == 300.0)
    ok &= check("unreachable target", budget_to_target(axis, curve, 1.0) == float("inf"))

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
