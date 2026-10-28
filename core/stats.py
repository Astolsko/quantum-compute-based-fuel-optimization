"""
core/stats.py  --  how sure we are about a benchmark number.

"Solver A was optimal on 10/10 instances, solver B on 8/10" reads like a win;
on 10 instances it is not one. These helpers put an interval or a p-value next
to every benchmark claim:

  wilson        95% interval for a success rate (k of n)
  bootstrap_ci  95% interval for a mean (e.g. the mean optimality gap)
  tts99         time-to-solution: expected time to hit the optimum at least
                once with 99% confidence by repeating a run (the standard
                annealing-benchmark metric, Ronnow et al. 2014)
  paired        two solvers on the SAME instances: Wilcoxon signed-rank on the
                gaps and an exact McNemar test on "hit the optimum"
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n, as fractions."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_ci(values, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95% percentile-bootstrap interval for the mean of `values` (finite only)."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    if np.ptp(v) == 0:                       # one value, or all identical
        return float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def tts99(p_success: float, seconds_per_run: float) -> float:
    """Seconds to reach the optimum at least once with 99% confidence by
    repeating independent runs (inf if no run ever succeeded)."""
    if p_success <= 0:
        return float("inf")
    if p_success >= 1:
        return seconds_per_run
    return seconds_per_run * math.log(0.01) / math.log(1.0 - p_success)


def paired(gaps_a, gaps_b, tol: float = 1e-6) -> dict:
    """Compare two solvers run on the same instances (None/inf = failed run).

    wilcoxon_p  on the gap differences, over instances where both succeeded
    mcnemar_p   exact two-sided test on "hit the optimum" (a failure counts as
                a miss); b = only A hit it, c = only B hit it
    """
    a = np.array([np.inf if g is None else g for g in gaps_a], dtype=float)
    b = np.array([np.inf if g is None else g for g in gaps_b], dtype=float)
    both = np.isfinite(a) & np.isfinite(b)
    diff = a[both] - b[both]
    w_p = (float(sps.wilcoxon(diff).pvalue) if np.count_nonzero(np.abs(diff) > tol) >= 1
           else 1.0)
    hit_a, hit_b = a < tol, b < tol
    only_a, only_b = int(np.sum(hit_a & ~hit_b)), int(np.sum(hit_b & ~hit_a))
    m_p = (float(sps.binomtest(only_a, only_a + only_b, 0.5).pvalue)
           if only_a + only_b else 1.0)
    return {"n_both": int(both.sum()), "mean_diff": (float(diff.mean()) if len(diff) else None),
            "wilcoxon_p": w_p, "only_a_opt": only_a, "only_b_opt": only_b,
            "mcnemar_p": m_p}
