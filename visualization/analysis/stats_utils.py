"""
Statistics for comparing stochastic optimisers, implemented with numpy only.

Provides the descriptive stats, paired significance test and effect sizes
used to compare variants across matched seeds, favouring median/IQR and
paired tests with effect sizes over bare percentage differences or a bare
minimum-over-runs metric:

* :func:`describe` - n, mean, median, sd, IQR, min/max and bootstrap
  confidence intervals for both the mean and the median.
* :func:`wilcoxon_signed_rank` - two-sided paired test, exact for small
  samples without ties and normal-approximated (with tie and continuity
  correction) otherwise.
* :func:`paired_effect_sizes` - matched-pairs rank-biserial correlation,
  Cohen's dz and the median paired difference with a bootstrap CI.
* :func:`cliffs_delta` - non-parametric unpaired effect size.
* :func:`holm_bonferroni` - multiplicity correction for the family of
  comparisons against the RL-GA.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------
# descriptive statistics
# ----------------------------------------------------------------------


@dataclass
class Summary:
    n: int
    mean: float
    median: float
    std: float
    q1: float
    q3: float
    iqr: float
    minimum: float
    maximum: float
    mean_ci_low: float
    mean_ci_high: float
    median_ci_low: float
    median_ci_high: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def format_mean(self, digits: int = 2) -> str:
        return f"{self.mean:.{digits}f} ± {self.std:.{digits}f}"

    def format_median(self, digits: int = 2) -> str:
        return f"{self.median:.{digits}f} [{self.q1:.{digits}f}, {self.q3:.{digits}f}]"

    def format_mean_ci(self, digits: int = 2) -> str:
        return f"[{self.mean_ci_low:.{digits}f}, {self.mean_ci_high:.{digits}f}]"


def bootstrap_ci(
    values: Sequence[float],
    statistic=np.mean,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 12345,
) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval of ``statistic``."""
    data = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if data.size == 0:
        return (float("nan"), float("nan"))
    if data.size == 1:
        return (float(data[0]), float(data[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, data.size, size=(n_boot, data.size))
    stats = statistic(data[idx], axis=1)
    low, high = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(low), float(high))


def describe(
    values: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 12345
) -> Summary:
    """Full descriptive summary, ignoring non-finite entries."""
    data = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if data.size == 0:
        nan = float("nan")
        return Summary(0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan)

    q1, q3 = (float(v) for v in np.percentile(data, [25, 75]))
    mean_lo, mean_hi = bootstrap_ci(data, np.mean, n_boot, alpha, seed)
    med_lo, med_hi = bootstrap_ci(data, np.median, n_boot, alpha, seed + 1)

    return Summary(
        n=int(data.size),
        mean=float(np.mean(data)),
        median=float(np.median(data)),
        std=float(np.std(data, ddof=1)) if data.size > 1 else 0.0,
        q1=q1,
        q3=q3,
        iqr=q3 - q1,
        minimum=float(np.min(data)),
        maximum=float(np.max(data)),
        mean_ci_low=mean_lo,
        mean_ci_high=mean_hi,
        median_ci_low=med_lo,
        median_ci_high=med_hi,
    )


# ----------------------------------------------------------------------
# paired test
# ----------------------------------------------------------------------


def _normal_sf(z: float) -> float:
    """Upper-tail probability of the standard normal distribution."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks starting at 1, ties receiving their average rank."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        average = 0.5 * ((i + 1) + (j + 1))
        ranks[order[i : j + 1]] = average
        i = j + 1
    return ranks


def _exact_wilcoxon_p(w: float, n: int) -> float:
    """Two-sided exact p-value for the signed-rank statistic (no ties).

    Builds the null distribution of W+ by counting the subsets of {1..n}
    with each possible sum (a subset-sum dynamic program).
    """
    total = n * (n + 1) // 2
    counts = np.zeros(total + 1, dtype=float)
    counts[0] = 1.0
    for rank in range(1, n + 1):
        shifted = np.zeros_like(counts)
        shifted[rank:] = counts[:-rank]
        counts = counts + shifted
    distribution = counts / counts.sum()
    w_int = int(round(w))
    tail = float(distribution[: w_int + 1].sum())
    return float(min(1.0, 2.0 * tail))


@dataclass
class PairedTest:
    n_pairs: int
    n_nonzero: int
    statistic: float
    p_value: float
    method: str
    mean_difference: float
    median_difference: float
    difference_ci_low: float
    difference_ci_high: float
    rank_biserial: float
    cohens_dz: float
    n_wins: int
    n_losses: int
    n_ties: int

    def to_dict(self) -> Dict:
        return asdict(self)


def wilcoxon_signed_rank(
    x: Sequence[float], y: Sequence[float]
) -> Tuple[float, float, str, int]:
    """Two-sided Wilcoxon signed-rank test on paired samples.

    Returns ``(statistic, p_value, method, n_nonzero)`` where the statistic is
    ``min(W+, W-)``.  Zero differences are dropped (Wilcoxon's original
    treatment).  The exact null distribution is used when there are at most 30
    non-zero differences and no ties in their magnitudes; otherwise a normal
    approximation with tie and continuity correction is applied.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired samples must have the same length")

    mask = np.isfinite(a) & np.isfinite(b)
    differences = a[mask] - b[mask]
    differences = differences[differences != 0]
    n = differences.size
    if n == 0:
        return (float("nan"), 1.0, "no non-zero differences", 0)

    magnitudes = np.abs(differences)
    ranks = _average_ranks(magnitudes)
    w_plus = float(ranks[differences > 0].sum())
    w_minus = float(ranks[differences < 0].sum())
    statistic = min(w_plus, w_minus)

    unique_counts = np.unique(magnitudes, return_counts=True)[1]
    has_ties = bool(np.any(unique_counts > 1))

    if n <= 30 and not has_ties:
        return (statistic, _exact_wilcoxon_p(statistic, n), "exact", n)

    mean_w = n * (n + 1) / 4.0
    tie_term = float(np.sum(unique_counts**3 - unique_counts))
    var_w = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var_w <= 0:
        return (statistic, 1.0, "degenerate", n)
    z = (abs(statistic - mean_w) - 0.5) / math.sqrt(var_w)
    p = min(1.0, 2.0 * _normal_sf(max(z, 0.0)))
    return (statistic, p, "normal approximation", n)


def paired_effect_sizes(
    x: Sequence[float], y: Sequence[float], n_boot: int = 10_000, seed: int = 999
) -> PairedTest:
    """Paired comparison of ``x`` against ``y`` (difference = x - y).

    For minimisation, a negative mean/median difference means ``x`` is better.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    differences = a - b

    statistic, p_value, method, n_nonzero = wilcoxon_signed_rank(a, b)

    nonzero = differences[differences != 0]
    if nonzero.size:
        ranks = _average_ranks(np.abs(nonzero))
        w_plus = float(ranks[nonzero > 0].sum())
        w_minus = float(ranks[nonzero < 0].sum())
        rank_biserial = (w_plus - w_minus) / (nonzero.size * (nonzero.size + 1) / 2.0)
    else:
        rank_biserial = 0.0

    sd = float(np.std(differences, ddof=1)) if differences.size > 1 else 0.0
    dz = float(np.mean(differences) / sd) if sd > 0 else 0.0
    lo, hi = bootstrap_ci(differences, np.mean, n_boot=n_boot, seed=seed)

    return PairedTest(
        n_pairs=int(differences.size),
        n_nonzero=int(n_nonzero),
        statistic=float(statistic),
        p_value=float(p_value),
        method=method,
        mean_difference=float(np.mean(differences)) if differences.size else float("nan"),
        median_difference=float(np.median(differences)) if differences.size else float("nan"),
        difference_ci_low=lo,
        difference_ci_high=hi,
        rank_biserial=float(rank_biserial),
        cohens_dz=dz,
        n_wins=int(np.sum(differences < 0)),
        n_losses=int(np.sum(differences > 0)),
        n_ties=int(np.sum(differences == 0)),
    )


# ----------------------------------------------------------------------
# unpaired effect size and multiplicity correction
# ----------------------------------------------------------------------


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> Tuple[float, str]:
    """Cliff's delta and its conventional magnitude label.

    delta = P(x > y) - P(x < y).  For a minimisation problem a *negative*
    delta means ``x`` tends to produce smaller (better) values.
    """
    a = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    b = np.asarray([v for v in y if np.isfinite(v)], dtype=float)
    if a.size == 0 or b.size == 0:
        return (float("nan"), "undefined")

    greater = int(np.sum(a[:, None] > b[None, :]))
    less = int(np.sum(a[:, None] < b[None, :]))
    delta = (greater - less) / (a.size * b.size)

    magnitude = abs(delta)
    if magnitude < 0.147:
        label = "negligible"
    elif magnitude < 0.33:
        label = "small"
    elif magnitude < 0.474:
        label = "medium"
    else:
        label = "large"
    return (float(delta), label)


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    values = list(p_values)
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    adjusted = [0.0] * n
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (n - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def significance_marker(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# ----------------------------------------------------------------------
# anytime-performance helpers
# ----------------------------------------------------------------------


def value_at_budget(
    budget_axis: np.ndarray, best_so_far: np.ndarray, budget: float
) -> float:
    """Best-so-far value once ``budget`` units of effort have been spent.

    ``budget_axis`` is the cumulative evaluations (or seconds) recorded at each
    generation and ``best_so_far`` the incumbent fitness at that moment.  If
    the run never reached the budget, its final value is returned.
    """
    axis = np.asarray(budget_axis, dtype=float)
    values = np.asarray(best_so_far, dtype=float)
    reached = np.flatnonzero(axis <= budget)
    if reached.size == 0:
        return float(values[0])
    return float(values[reached[-1]])


def budget_to_target(
    budget_axis: np.ndarray, best_so_far: np.ndarray, target: float
) -> float:
    """Effort needed to first reach ``target``; ``inf`` if never reached."""
    axis = np.asarray(budget_axis, dtype=float)
    values = np.asarray(best_so_far, dtype=float)
    hit = np.flatnonzero(values <= target)
    if hit.size == 0:
        return float("inf")
    return float(axis[hit[0]])
