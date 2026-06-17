# evalaudit

**Psychometric audits for evaluation data — reliability, agreement, and bias in one call.**

LLM-as-judge and human-rating pipelines are everywhere, and the people running them rarely
check the things a measurement specialist would insist on before trusting a score: is the
rater consistent with itself, does it agree with humans, and is it swayed by answer length,
group membership, or presentation order. There are good libraries for model fairness and good
harnesses for running evals, but no small, focused tool that takes evaluation data and hands
back a psychometric audit. `evalaudit` is that tool.

Point it at a table of scores and get a report.

```python
from evalaudit import audit

report = audit(
    df,
    judge_cols=["judge_1", "judge_2"],   # repeated passes -> self-consistency
    human_cols=["human_1", "human_2"],   # optional -> agreement
    truth_col="true_quality",            # optional -> bias checks
    group_col="group",                   # optional -> invariance
    length_col="length",                 # optional -> verbosity bias
)
print(report)
report.to_dict()   # machine-readable
```

```
EVALAUDIT REPORT
========================================================
[ ok ] self-consistency (kappa)     +0.861   test-retest across judge passes
[ ok ] agreement w/ humans (kappa)   +0.831   judge vs human consensus
[FLAG] verbosity bias (partial r)   +0.374   score ~ length | true quality
[FLAG] group gap                    +0.310   score offset at equal quality
--------------------------------------------------------
NEEDS REVIEW — 2 flag(s)
```

The same call runs unchanged on a real judge's logged scores or on human ratings — the
library doesn't care where the numbers came from.

---

## Install

```bash
pip install -e .        # from a clone
```

Dependencies are just `numpy` and `pandas`.

## What it checks

| Check | Needs | Method |
|-------|-------|--------|
| Self-consistency | ≥2 judge passes | quadratic-weighted Cohen's kappa (test-retest) |
| Agreement with humans | human rater columns | weighted kappa vs human consensus |
| Verbosity bias | length + truth columns | partial correlation of score with length, controlling for quality |
| Group invariance | group + truth columns | mean score offset by group at equal quality |

Each check is flagged against a sensible default threshold; pass your own via `thresholds=`.
A flag is a prompt to investigate, not a verdict.

The lower-level functions are public too, if you want a single metric:

```python
from evalaudit.reliability import weighted_kappa, icc_oneway, krippendorff_alpha_ordinal
from evalaudit.bias import verbosity_bias, group_gap, position_bias
```

## Run the example and tests

```bash
python example.py
python tests/test_basic.py
```

## Scope and limitations

- This is a focused audit toolkit, not a full IRT/SEM package. The invariance check is a mean
  offset, not a formal differential-item-functioning model — a deliberate "fast first look."
- Metrics assume ordinal integer scores over a fixed range (default 1–5).
- Position bias is computed from paired comparisons scored in both orders (see
  `bias.position_bias`); it isn't yet wired into the high-level `audit()` call.

## Why it exists

Built by an I/O psychologist working in AI evaluation, on the premise that an evaluator is a
measurement instrument and should be validated like one. Background reading:
[AI Evaluation Is a Measurement Problem](https://amira-ghazy.github.io/ai-evaluation.html).

## License

MIT.
