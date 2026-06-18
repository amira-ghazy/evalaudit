"""Minimal tests: a clean instrument should pass, a biased one should flag."""
import numpy as np
import pandas as pd

from evalaudit import audit, measurement_invariance
from evalaudit.reliability import weighted_kappa


def _frame(verbosity=0.0, tilt=0.0, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    n = 300
    q = rng.integers(1, 6, n)
    length = (rng.normal(180, 70, n) + 12 * q).clip(40, 500)
    group = rng.choice(["A", "B"], n)
    clamp = lambda x: int(np.clip(round(x), 1, 5))
    def judge():
        return [clamp(q[i] + verbosity * (length[i] - 180) + (tilt if group[i] == "B" else 0) + rng.normal(0, noise)) for i in range(n)]
    return pd.DataFrame({"true_quality": q, "length": length.round().astype(int), "group": group,
                         "j1": judge(), "j2": judge(),
                         "h1": [clamp(q[i] + rng.normal(0, 0.4)) for i in range(n)]})


def test_weighted_kappa_identical_is_one():
    a = [1, 2, 3, 4, 5, 3, 2]
    assert abs(weighted_kappa(a, a, 5) - 1.0) < 1e-9


def test_clean_judge_passes():
    df = _frame(verbosity=0.0, tilt=0.0, noise=0.2)
    rep = audit(df, judge_cols=["j1", "j2"], human_cols=["h1"],
                truth_col="true_quality", group_col="group", length_col="length")
    assert rep.passed


def test_biased_judge_flags():
    df = _frame(verbosity=0.01, tilt=-0.5, noise=0.6)
    rep = audit(df, judge_cols=["j1", "j2"], human_cols=["h1"],
                truth_col="true_quality", group_col="group", length_col="length")
    assert not rep.passed


def test_invariance_clean_judge_no_dif():
    rng = np.random.default_rng(1)
    n = 500
    q = rng.integers(1, 6, n)
    g = rng.choice(["A", "B"], n)
    s = np.clip(np.round(q + rng.normal(0, 0.4, n)), 1, 5)
    inv = measurement_invariance(s, q, g)
    assert not inv["flag_uniform"] and not inv["flag_non_uniform"]


def test_invariance_catches_uniform_and_nonuniform():
    rng = np.random.default_rng(2)
    n = 600
    q = rng.integers(1, 6, n)
    g = rng.choice(["A", "B"], n)
    clamp = lambda a: np.clip(np.round(a), 1, 5)
    uniform = clamp(q - 0.7 * (g == "B") + rng.normal(0, 0.4, n))
    assert measurement_invariance(uniform, q, g)["flag_uniform"]
    nonuniform = clamp(np.where(g == "B", 3 + 0.5 * (q - 3), q) + rng.normal(0, 0.4, n))
    assert measurement_invariance(nonuniform, q, g)["flag_non_uniform"]


if __name__ == "__main__":
    test_weighted_kappa_identical_is_one()
    test_clean_judge_passes()
    test_biased_judge_flags()
    test_invariance_clean_judge_no_dif()
    test_invariance_catches_uniform_and_nonuniform()
    print("all tests passed")
