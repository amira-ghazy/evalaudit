"""Bias: is the score swayed by things that aren't quality?"""
from __future__ import annotations

import numpy as np


def _resid(a, z):
    a = np.asarray(a, dtype=float)
    z = np.asarray(z, dtype=float)
    A = np.vstack([z, np.ones_like(z)]).T
    coef, *_ = np.linalg.lstsq(A, a, rcond=None)
    return a - A @ coef


def partial_corr(x, y, z) -> float:
    """Correlation of x and y after removing the linear effect of z from both."""
    rx, ry = _resid(x, z), _resid(y, z)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def verbosity_bias(score, length, truth) -> float:
    """How much answer length sways the score, controlling for true quality.

    Positive = longer answers score higher for reasons unrelated to quality.
    """
    return partial_corr(score, length, truth)


def group_gap(score, truth, group):
    """Mean (score - truth) per group, and the largest gap between groups.

    A non-zero gap means the instrument is not measurement-invariant: it scores
    one group differently at equal true quality.
    """
    score = np.asarray(score, dtype=float)
    truth = np.asarray(truth, dtype=float)
    group = np.asarray(group)
    tilts = {g: float((score[group == g] - truth[group == g]).mean()) for g in np.unique(group)}
    gap = max(tilts.values()) - min(tilts.values()) if len(tilts) > 1 else 0.0
    return {"per_group": tilts, "gap": float(gap)}


def position_bias(picked_first_order1, picked_first_order2):
    """Pairwise position bias from the same comparisons scored in both orders.

    Each array is boolean/0-1: did the judge pick the answer shown FIRST?
    Returns the rate the first-shown answer wins (0.5 = fair) and the flip rate.
    """
    o1 = np.asarray(picked_first_order1, dtype=float)
    o2 = np.asarray(picked_first_order2, dtype=float)
    first_wins = float((o1.mean() + o2.mean()) / 2)
    # winner flips when, after swapping, the same side is no longer chosen
    flips = float(np.mean(o1 != (1 - o2)))
    return {"first_shown_wins": first_wins, "flip_rate": flips}
