"""Measurement invariance: does the judge hold the same standard across groups?

A single mean gap (see bias.group_gap) only catches a constant offset. Real
measurement bias has two flavours, and they need different fixes:

  - uniform DIF      the judge is systematically harsher/kinder to a group at
                     EQUAL true quality (an intercept difference).
  - non-uniform DIF  the judge tracks quality differently across groups, e.g.
                     it separates good from bad answers well for one group and
                     compresses the other (a slope difference).

This module fits the regression model of test bias (Cleary, 1968), the same
hierarchical setup as logistic-regression DIF (Swaminathan & Rogers, 1990;
Zumbo, 1999), in linear form:

    score ~ quality                         (M0, baseline)
    score ~ quality + group                 (M1, adds uniform DIF)
    score ~ quality + group + quality:group (M2, adds non-uniform DIF)

Each flavour is reported as an incremental-R^2 effect size with a nested F.
Effect-size cut-offs follow common DIF practice (Zumbo & Thomas): R^2 gain
below ~0.035 is negligible, 0.035-0.070 moderate, above 0.070 large.
"""
from __future__ import annotations

import numpy as np

DELTA_R2_FLAG = 0.035  # moderate-or-larger DIF


def _ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, float(resid @ resid)


def measurement_invariance(score, truth, group, delta_r2_flag: float = DELTA_R2_FLAG,
                           shift_d_flag: float = 0.20):
    """Test whether a judge/rater is measurement-invariant across groups.

    score, truth, group : equal-length arrays. `truth` is the known/target
    quality used as the matching variable; `group` is the grouping label.

    Uniform DIF is flagged when EITHER its incremental R^2 reaches the
    conventional moderate cut-off (delta_r2_flag) OR the standardized score
    shift at equal quality reaches a small-effect cut-off (shift_d_flag, in
    score SD units) — the way adverse-impact analysis weighs a systematic
    offset that is practically meaningful even when its R^2 gain is modest.

    Returns a dict with per-group calibration (intercept, slope, mean residual),
    and `uniform` / `non_uniform` blocks each carrying the R^2 gain, nested F,
    and a flag. With <2 groups or no quality variance, everything is non-flagged.
    """
    score = np.asarray(score, dtype=float)
    truth = np.asarray(truth, dtype=float)
    group = np.asarray(group)
    n = len(score)
    groups = list(np.unique(group))

    out = {"n": n, "groups": groups, "per_group": {}}
    for g in groups:
        m = group == g
        yg, xg = score[m], truth[m]
        cal = {"n": int(m.sum()), "mean_residual": float((yg - xg).mean()) if m.sum() else float("nan")}
        if m.sum() >= 2 and xg.std() > 0:
            b, _ = _ols(np.column_stack([np.ones(m.sum()), xg]), yg)
            cal["intercept"], cal["slope"] = float(b[0]), float(b[1])
        else:
            cal["intercept"] = cal["slope"] = float("nan")
        out["per_group"][g] = cal

    sst = float(((score - score.mean()) ** 2).sum())
    if len(groups) < 2 or sst == 0:
        out["uniform"] = {"delta_r2": 0.0, "F": float("nan"), "max_shift": 0.0, "shift_vs_ref": {}}
        out["non_uniform"] = {"delta_r2": 0.0, "F": float("nan")}
        out["flag_uniform"] = out["flag_non_uniform"] = False
        return out

    tc = truth - truth.mean()                       # centre quality
    ones = np.ones(n)
    D = np.column_stack([(group == g).astype(float) for g in groups[1:]])  # k-1 dummies, ref = groups[0]
    DI = D * tc[:, None]

    X0 = np.column_stack([ones, tc])
    X1 = np.column_stack([ones, tc, D])
    X2 = np.column_stack([ones, tc, D, DI])
    b0, sse0 = _ols(X0, score)
    b1, sse1 = _ols(X1, score)
    b2, sse2 = _ols(X2, score)

    r2 = lambda sse: 1.0 - sse / sst
    d_uniform = r2(sse1) - r2(sse0)
    d_non = r2(sse2) - r2(sse1)

    def nested_F(sse_small, sse_big, df_add, p_big):
        den = sse_big / (n - p_big)
        return float(((sse_small - sse_big) / df_add) / den) if den > 0 and df_add > 0 else float("nan")

    kU, kN = D.shape[1], DI.shape[1]
    # uniform shifts (quality centred, so these are score gaps at mean quality, focal - reference)
    shifts = {groups[i + 1]: float(b1[2 + i]) for i in range(kU)}
    max_shift = max([0.0] + [abs(v) for v in shifts.values()])
    score_sd = float(np.std(score, ddof=1))
    std_shift = float(max_shift / score_sd) if score_sd > 0 else 0.0

    out["uniform"] = {
        "delta_r2": float(d_uniform),
        "F": nested_F(sse0, sse1, kU, X1.shape[1]),
        "reference": groups[0],
        "shift_vs_ref": shifts,
        "max_shift": float(max_shift),
        "std_shift": std_shift,
    }
    out["non_uniform"] = {
        "delta_r2": float(d_non),
        "F": nested_F(sse1, sse2, kN, X2.shape[1]),
    }
    out["flag_uniform"] = (d_uniform >= delta_r2_flag) or (std_shift >= shift_d_flag)
    out["flag_non_uniform"] = d_non >= delta_r2_flag
    return out
