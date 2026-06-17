"""A runnable example: audit a (simulated) LLM judge in a few lines.

Replace the simulated frame with your own logged judge scores and the same call
runs unchanged.
"""
import numpy as np
import pandas as pd

from evalaudit import audit

rng = np.random.default_rng(7)
n = 300
true_q = rng.integers(1, 6, n)
length = (rng.normal(180, 70, n) + 12 * true_q).clip(40, 500)
group = rng.choice(["A", "B"], n)
clamp = lambda x: int(np.clip(round(x), 1, 5))

df = pd.DataFrame({
    "true_quality": true_q,
    "length": length.round().astype(int),
    "group": group,
    # a judge with verbosity bias + a group tilt + run-to-run noise
    "judge_1": [clamp(true_q[i] + 0.0045 * (length[i] - 180) + (-0.35 if group[i] == "B" else 0) + rng.normal(0, 0.6)) for i in range(n)],
    "judge_2": [clamp(true_q[i] + 0.0045 * (length[i] - 180) + (-0.35 if group[i] == "B" else 0) + rng.normal(0, 0.6)) for i in range(n)],
    "human_1": [clamp(true_q[i] + rng.normal(0, 0.5)) for i in range(n)],
    "human_2": [clamp(true_q[i] + rng.normal(0, 0.5)) for i in range(n)],
})

report = audit(
    df,
    judge_cols=["judge_1", "judge_2"],
    human_cols=["human_1", "human_2"],
    truth_col="true_quality",
    group_col="group",
    length_col="length",
)
print(report)
