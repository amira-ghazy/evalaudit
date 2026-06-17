"""Reliability: do raters (human or model) agree, and does a rater agree with itself?"""
from __future__ import annotations

import numpy as np


def weighted_kappa(a, b, k: int) -> float:
    """Quadratic-weighted Cohen's kappa for ordinal ratings in 1..k.

    Rewards near-agreement (a 4 vs a 5 is close) and corrects for chance.
    """
    a = np.asarray(a, dtype=int) - 1
    b = np.asarray(b, dtype=int) - 1
    o = np.zeros((k, k))
    for x, y in zip(a, b):
        o[x, y] += 1
    o /= o.sum()
    w = np.array([[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)])
    e = np.outer(o.sum(axis=1), o.sum(axis=0))
    denom = (w * e).sum()
    return float(1 - (w * o).sum() / denom) if denom else float("nan")


def icc_oneway(*raters) -> float:
    """One-way ICC(1) across two or more raters scoring the same items.

    Pass each rater's scores as a separate array of equal length.
    """
    data = np.vstack([np.asarray(r, dtype=float) for r in raters]).T  # items x raters
    n, m = data.shape
    grand = data.mean()
    row_means = data.mean(axis=1)
    ms_between = m * np.sum((row_means - grand) ** 2) / (n - 1)
    ms_within = np.sum((data - row_means[:, None]) ** 2) / (n * (m - 1))
    denom = ms_between + (m - 1) * ms_within
    return float((ms_between - ms_within) / denom) if denom else float("nan")


def krippendorff_alpha_ordinal(*raters, k: int) -> float:
    """Krippendorff's alpha (ordinal) for two complete rater columns."""
    m = np.vstack([np.asarray(r, dtype=int) for r in raters]).T - 1
    pairs = [(row[0], row[1]) for row in m]
    do = np.mean([(x - y) ** 2 for x, y in pairs])
    vals = m.flatten()
    de = np.mean([(x - y) ** 2 for x in vals for y in vals])
    return float(1 - do / de) if de else float("nan")
