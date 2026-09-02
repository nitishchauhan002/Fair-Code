> *A model can pass Equal Opportunity - catching qualified people from every group at the same rate - while still falsely flagging one group's innocent members far more often than another's. Equal Opportunity only checks half of what Equalized Odds checks, and which half turns out to matter a lot.*

## The One-Sentence Definition

**Equal Opportunity** requires a model's true positive rate - among people who actually belong to the positive class, the fraction it correctly flags - to be equal across protected groups, and nothing more; it is the deliberately relaxed half of [Equalized Odds](equalized-odds.md), which requires that *and* an equal false positive rate.

## Why It Matters

Equal Opportunity was introduced by Hardt, Price, and Srebro (2016) as a practical compromise: full Equalized Odds can be hard to satisfy exactly, and in settings where missing a qualified person is the harm that matters most - a denied loan to someone who would have repaid it, a screened-out tenant who would have been a good renter - checking only the true positive rate targets that harm directly.

The trade-off is exactly what it sounds like: Equal Opportunity says nothing about false positives. A model can hit identical true positive rates across groups while flagging one group's *qualified-to-reject* members - people who genuinely wouldn't repay, wouldn't reoffend, wouldn't default - at a much higher rate than another's. Equal Opportunity would call that model fair. Equalized Odds would not.

Because Equalized Odds is defined as whichever gap (true positive rate or false positive rate) is larger, it is mathematically never smaller than Equal Opportunity's gap - it can only be equal to it or bigger. Equal Opportunity is a floor, not a ceiling: passing it tells you the true-positive-rate gap is small, but tells you nothing about how large the false-positive-rate gap might be hiding underneath.

## Concrete Example

Both real audits below use `equal_opportunity_diff` and `equalized_odds_diff` exactly as computed by this repo's frozen benchmark harness (`paper/results-frozen/results_fairness.csv`), baseline model, no mitigation applied.

### When They Coincide: COMPAS - Audit 01

For the logistic regression baseline on race (disadvantaged: African-American defendants, n=1,048; advantaged: Caucasian defendants, n=467):

| Metric | Value | 95% CI | p-value |
|---|---:|---|---:|
| Equal Opportunity Diff | 0.926 | [0.909, 0.942] | 0.0 |
| Equalized Odds Diff | 0.926 | [0.909, 0.942] | 0.0 |

The two numbers are identical here, and the frozen results record exactly why: `equalized_odds_diff`'s own definition is `max(|TPR gap|, |FPR gap|)`, and its `note` field for this row reads `driven_by_tpr_gap` - meaning the true-positive-rate gap (what Equal Opportunity measures) was the larger of the two, so Equalized Odds simply inherited it. Nothing was hidden in the false-positive-rate gap this time.

### When They Diverge: Tenant Screening - Audit 07

For the same baseline model on race (disadvantaged: Black applicants; advantaged: White applicants):

| Metric | Value | 95% CI | p-value | Driven by |
|---|---:|---|---:|---|
| Equal Opportunity Diff | 0.047 | [0.018, 0.077] | 0.0025 | - |
| Equalized Odds Diff | 0.068 | [0.023, 0.111] | 0.0025 | false positive rate |

Here they disagree. A screening process relying only on Equal Opportunity would report a modest 4.7-point gap and move on. The real, larger disparity - a 6.8-point false-positive-rate gap, meaning qualified-to-reject Black applicants are screened out at a meaningfully higher excess rate than qualified-to-reject White applicants - only shows up once Equalized Odds is checked. This is the exact failure mode Equal Opportunity's relaxation accepts: it was never designed to catch it.

## Detection Code

Computes both metrics side by side so a gap between them is visible immediately, rather than computing Equal Opportunity alone and never finding out what it left unchecked.

```python
import numpy as np
import pandas as pd


def equal_opportunity_and_equalized_odds(y_true, y_pred, group, disadvantaged, advantaged):
    """
    Computes the Equal Opportunity gap (true-positive-rate difference,
    among y_true == 1 rows only) and the Equalized Odds gap
    (max of the true-positive-rate and false-positive-rate differences)
    between two groups, so the two can be compared directly.

    Parameters:
        y_true: array-like of true binary labels (1 = positive class)
        y_pred: array-like of predicted binary labels
        group: array-like of group membership, same length as y_true
        disadvantaged, advantaged: the two group values to compare

    Returns a dict with tpr_gap (== Equal Opportunity), fpr_gap,
    equalized_odds_gap (whichever of the two is larger by magnitude),
    and which one drove it.
    """
    df = pd.DataFrame({"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred),
                        "group": np.asarray(group)})

    def rate(mask):
        sub = df[mask]
        return sub["y_pred"].mean() if len(sub) else float("nan")

    tpr_disadv = rate((df["group"] == disadvantaged) & (df["y_true"] == 1))
    tpr_adv = rate((df["group"] == advantaged) & (df["y_true"] == 1))
    fpr_disadv = rate((df["group"] == disadvantaged) & (df["y_true"] == 0))
    fpr_adv = rate((df["group"] == advantaged) & (df["y_true"] == 0))

    tpr_gap = tpr_disadv - tpr_adv
    fpr_gap = fpr_disadv - fpr_adv
    driven_by = "false_positive_rate" if abs(fpr_gap) > abs(tpr_gap) else "true_positive_rate"

    return {
        "equal_opportunity_gap": tpr_gap,
        "false_positive_rate_gap": fpr_gap,
        "equalized_odds_gap": fpr_gap if abs(fpr_gap) > abs(tpr_gap) else tpr_gap,
        "equalized_odds_driven_by": driven_by,
    }


# Usage example:
# result = equal_opportunity_and_equalized_odds(
#     y_true, y_pred, df["race"], disadvantaged="Black", advantaged="White",
# )
# print(result)
```

## Limitations

### 1. A close-to-zero Equal Opportunity gap proves nothing about false positives

As Tenant Screening shows above, a small true-positive-rate gap can sit right next to a meaningfully larger false-positive-rate gap. Report both numbers, or report Equalized Odds directly, rather than Equal Opportunity alone.

### 2. It still depends on a clean ground-truth label

Equal Opportunity, like Equalized Odds, is computed against `y_true` - if the label itself encodes historical bias (a past lending decision, a past arrest), the "true positive rate" it equalizes is a rate against a biased target, not against who genuinely deserved the outcome. See [What Is Label Bias?](label-bias.md).

### 3. Choosing Equal Opportunity over Equalized Odds is a values decision, not a technical one

Relaxing the false-positive-rate requirement is only defensible when false negatives are agreed to be the more serious harm for that specific system. That is not universally true - a wrongly denied benefits claim (false negative) and a wrongly approved fraudulent one (false positive) do not carry the same weight in every domain, and the choice of which fairness definition to enforce should be made explicitly, not by default.

### 4. Small subgroups make both gaps noisy

The true-positive-rate gap is estimated only from rows where `y_true == 1`; for a rare positive class in a small subgroup, that can be a handful of rows. Always report the underlying `n` per group alongside the gap, and bootstrap a confidence interval before treating a difference as real.

## Related Concepts

* [What Is Equalized Odds?](equalized-odds.md) - the stricter metric Equal Opportunity relaxes, and the one that catches what Equal Opportunity alone cannot.
* [False Positives vs. False Negatives in Medical Risk Models](false-positives-vs-false-negatives.md) - the per-threshold trade-off between the two error types Equal Opportunity and Equalized Odds split apart.
* [What Is Demographic Parity?](demographic-parity.md) - a fairness definition that ignores ground truth entirely, unlike Equal Opportunity and Equalized Odds which both condition on it.
* [What Is Label Bias?](label-bias.md) - why the "true" in true positive rate is only as trustworthy as the label it's measured against.

## Related Projects in This Repo

* [`COMPAS/`](../COMPAS/) - the audit where Equal Opportunity and Equalized Odds coincide, because the true-positive-rate gap dominates.
* [`Tenant Screening/`](../Tenant%20Screening/) - the audit where they diverge, because the false-positive-rate gap is actually the larger one.

## Further Reading

* [Hardt, M., Price, E., Srebro, N. (2016): Equality of Opportunity in Supervised Learning](https://arxiv.org/abs/1610.02413) - the paper that introduced Equal Opportunity as a relaxation of Equalized Odds.
* [Chouldechova, A. (2017): Fair Prediction with Disparate Impact](https://arxiv.org/abs/1703.00056) - the impossibility result underlying why relaxations like Equal Opportunity exist in the first place.
* [Barocas, S., Hardt, M., Narayanan, A. (2019): *Fairness and Machine Learning*](https://fairmlbook.org/classification.html) - situates Equal Opportunity among the full family of classification-based fairness criteria.

*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
