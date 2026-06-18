"""evalaudit — psychometric audits for evaluation data.

Point it at a table of judge scores (and, where you have them, repeated passes,
human ratings, a group flag, and answer lengths) and get back reliability,
agreement, and bias in one call. Works the same on a real LLM judge's logged
scores or on human ratings.

    from evalaudit import audit
    report = audit(df, judge_cols=["judge_1", "judge_2"], human_cols=["h1", "h2"],
                   truth_col="true_quality", group_col="group", length_col="length")
    print(report)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bias import group_gap, verbosity_bias
from .invariance import measurement_invariance
from .reliability import weighted_kappa
from .report import AuditReport

__all__ = ["audit", "AuditReport", "measurement_invariance"]

# Default thresholds for flagging a concern. Override via the `thresholds` arg.
DEFAULTS = {
    "self_consistency_min": 0.80,
    "agreement_min": 0.60,
    "verbosity_abs_max": 0.15,
    "group_gap_max": 0.10,
    "dif_delta_r2_flag": 0.035,
}


def _consensus(df, cols):
    return np.round(df[cols].mean(axis=1)).astype(int)


def audit(df: pd.DataFrame, *, judge_cols, human_cols=None, truth_col=None,
          group_col=None, length_col=None, score_range=(1, 5), thresholds=None) -> AuditReport:
    """Run the psychometric battery on an evaluation table.

    judge_cols   one or more columns of the judge's scores. Two or more enables a
                 test-retest self-consistency check.
    human_cols   optional human rater columns; enables agreement (and, with two or
                 more, a human-reliability reference).
    truth_col    optional known/target quality; needed for verbosity and group checks.
    group_col    optional group flag for the invariance check.
    length_col   optional answer length for the verbosity check.
    """
    t = {**DEFAULTS, **(thresholds or {})}
    k = score_range[1] - score_range[0] + 1
    rep = AuditReport()
    judge = df[judge_cols[0]]

    # 1. self-consistency
    if len(judge_cols) >= 2:
        sc = weighted_kappa(df[judge_cols[0]], df[judge_cols[1]], k)
        rep.add("self-consistency (kappa)", sc, sc < t["self_consistency_min"],
                "test-retest across judge passes")
    else:
        rep.add("self-consistency (kappa)", None, False, "needs >=2 judge passes")

    # 2. agreement with humans
    if human_cols:
        consensus = _consensus(df, human_cols)
        ag = weighted_kappa(judge, consensus, k)
        rep.add("agreement w/ humans (kappa)", ag, ag < t["agreement_min"], "judge vs human consensus")

    # 3. verbosity bias
    if length_col is not None and truth_col is not None:
        vb = verbosity_bias(judge, df[length_col], df[truth_col])
        rep.add("verbosity bias (partial r)", vb, abs(vb) > t["verbosity_abs_max"],
                "score ~ length | true quality")

    # 4. measurement invariance across groups (uniform + non-uniform DIF)
    if group_col is not None and truth_col is not None:
        inv = measurement_invariance(judge, df[truth_col], df[group_col],
                                     delta_r2_flag=t["dif_delta_r2_flag"])
        u, nu = inv["uniform"], inv["non_uniform"]
        rep.add("uniform DIF (R2 gain)", u["delta_r2"], inv["flag_uniform"],
                "score shift %+.2f at equal quality" % u["max_shift"])
        rep.add("non-uniform DIF (R2 gain)", nu["delta_r2"], inv["flag_non_uniform"],
                "quality-tracking differs across groups")

    return rep
