> *A model can be almost exactly as accurate for one group as another and still be dramatically unfair - overall accuracy hides which kind of mistake a model makes, and to whom.*

## The One-Sentence Definition

**Accuracy Equality** requires a model's overall accuracy - the fraction of predictions it gets right, correct or incorrect, in either direction - to be equal across protected groups; unlike [Equalized Odds](equalized-odds.md) or [Equal Opportunity](equal-opportunity.md), it never looks at *which* predictions were wrong, only how many.

## Why It Matters

Accuracy is the metric everyone already checks, so it's tempting to treat "equal accuracy across groups" as good enough evidence a model is fair. It isn't. Two groups can land on the same accuracy number through completely different error profiles: one group's mistakes can be almost entirely false positives, the other's almost entirely false negatives, and the two error rates can net out to the same overall score.

That distinction matters because false positives and false negatives are rarely equally costly. A false positive in a recidivism-risk tool means someone is wrongly flagged as high-risk; a false negative means someone who reoffends was scored as low-risk. Accuracy Equality would call a model "fair" even if it systematically makes the costlier error type against one specific group - which is exactly the blind spot [Equalized Odds](equalized-odds.md) and [Equal Opportunity](equal-opportunity.md) are built to catch, at the cost of a stricter, harder-to-satisfy requirement.

Accuracy Equality is one of the six fairness metrics this repo's benchmark harness actually computes (`accuracy_equality_diff` in `faircode/metrics.py`) - not a theoretical add-on, but a real number sitting next to the other five for every audit, strategy, and model in `paper/results-frozen/results_fairness.csv`.

## Concrete Example: COMPAS - Audit 01

The baseline logistic regression model on race (disadvantaged: African-American defendants; advantaged: Caucasian defendants) is a case where accuracy is nearly equal but the underlying errors are not, using the frozen numbers exactly as `faircode/benchmark.py` computed them:

| Metric | Value | 95% CI | p-value | n (disadv. / adv.) |
|---|---:|---|---:|---|
| Accuracy Equality Diff | -0.035 | [-0.070, -0.003] | 0.0395 | 1,788 / 1,466 |
| Equalized Odds Diff | 0.926 | [0.909, 0.942] | 0.0 | 1,048 / 467 |

A 3.5-point accuracy gap (African-American defendants scored slightly *less* accurately overall) looks like a minor, borderline-significant footnote. It sits right next to a 92.6-point Equalized Odds gap on the exact same model and dataset - `faircode/metrics.py`'s own `note` field for that row reads `driven_by_tpr_gap`, meaning the true-positive-rate gap between the two groups is enormous. Reporting only Accuracy Equality here would describe this model as nearly fair. It is not: African-American defendants who did not reoffend were flagged as high-risk at a far higher rate than Caucasian defendants who did not reoffend, a disparity Accuracy Equality's single blended number cannot see because the two error types are canceling each other out in the total.

## Detection Code

Computes Accuracy Equality alongside Equalized Odds so a hidden gap can't cancel out unnoticed.

```python
import numpy as np
import pandas as pd


def accuracy_equality_and_equalized_odds(y_true, y_pred, group, disadvantaged, advantaged):
    """
    Computes the Accuracy Equality gap (overall-accuracy difference) and the
    Equalized Odds gap (max of the true-positive-rate and false-positive-rate
    differences) between two groups, so a small accuracy gap can be checked
    against a possibly much larger error-type gap underneath it.

    Parameters:
        y_true: array-like of true binary labels
        y_pred: array-like of predicted binary labels
        group: array-like of group membership, same length as y_true
        disadvantaged, advantaged: the two group values to compare

    Returns a dict with accuracy_gap, tpr_gap, fpr_gap, and
    equalized_odds_gap (whichever of tpr_gap/fpr_gap is larger by magnitude).
    """
    df = pd.DataFrame({"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred),
                        "group": np.asarray(group)})
    df["correct"] = (df["y_true"] == df["y_pred"]).astype(int)

    def rate(mask, col="correct"):
        sub = df[mask]
        return sub[col].mean() if len(sub) else float("nan")

    acc_disadv = rate(df["group"] == disadvantaged)
    acc_adv = rate(df["group"] == advantaged)

    tpr_disadv = rate((df["group"] == disadvantaged) & (df["y_true"] == 1), "y_pred")
    tpr_adv = rate((df["group"] == advantaged) & (df["y_true"] == 1), "y_pred")
    fpr_disadv = rate((df["group"] == disadvantaged) & (df["y_true"] == 0), "y_pred")
    fpr_adv = rate((df["group"] == advantaged) & (df["y_true"] == 0), "y_pred")

    tpr_gap = tpr_disadv - tpr_adv
    fpr_gap = fpr_disadv - fpr_adv

    return {
        "accuracy_equality_gap": acc_disadv - acc_adv,
        "true_positive_rate_gap": tpr_gap,
        "false_positive_rate_gap": fpr_gap,
        "equalized_odds_gap": fpr_gap if abs(fpr_gap) > abs(tpr_gap) else tpr_gap,
    }


# Usage example:
# result = accuracy_equality_and_equalized_odds(
#     y_true, y_pred, df["race"], disadvantaged="African-American", advantaged="Caucasian",
# )
# print(result)
# A small accuracy_equality_gap next to a large equalized_odds_gap means the
# errors are canceling out in the total, not actually balanced.
```

## Limitations

### 1. A near-zero gap can hide a large, offsetting error-type gap

As COMPAS shows above, a small Accuracy Equality gap is fully compatible with a huge Equalized Odds gap on the same model. Never report Accuracy Equality as the only fairness check - pair it with a metric that conditions on the true label, like Equalized Odds or Equal Opportunity.

### 2. It's the metric most likely to be reported alone, by default

Because overall accuracy is already the standard model-evaluation metric, it's the easiest one to slice by group and call a fairness check, without anyone deciding to do that deliberately. That makes Accuracy Equality a metric that tends to get checked *instead of* the others, not *alongside* them, unless a team's fairness process explicitly requires more.

### 3. It still depends on a clean ground-truth label

Like every metric that uses `y_true`, Accuracy Equality is only as trustworthy as the label it's measured against. See [What Is Label Bias?](label-bias.md).

### 4. Small subgroups make the gap noisy

`faircode/significance.py`'s bootstrap CI on the COMPAS example above already spans from -7.0 to -0.3 points - a real range, not a single fixed number. See [What Is a Bootstrap Confidence Interval?](bootstrap-confidence-intervals.md) for why that range matters more than the point estimate alone.

## Related Concepts

* [What Is Equalized Odds?](equalized-odds.md) - the metric that catches the error-type imbalance Accuracy Equality's blended number can hide.
* [What Is Equal Opportunity?](equal-opportunity.md) - the relaxed half of Equalized Odds, still stricter than Accuracy Equality since it conditions on the true label.
* [What Is a Confusion Matrix?](confusion-matrix.md) - the four-way breakdown (true/false positive/negative) that Accuracy Equality collapses into one number.
* [False Positives vs. False Negatives in Medical Risk Models](false-positives-vs-false-negatives.md) - why the two error types Accuracy Equality can't distinguish are rarely equally costly.

## Related Projects in This Repo

* [`COMPAS/`](../COMPAS/) - the audit above, where a 3.5-point accuracy gap sits next to a 92.6-point Equalized Odds gap on the same model.
* [`faircode/metrics.py`](../faircode/metrics.py) - `accuracy_equality_diff`'s actual implementation, computed alongside the other five fairness metrics for every audit in this repo's benchmark harness.

## Further Reading

* [Hardt, M., Price, E., Srebro, N. (2016): Equality of Opportunity in Supervised Learning](https://arxiv.org/abs/1610.02413) - situates accuracy-based fairness against the true-positive/false-positive-rate criteria it doesn't capture.
* [Berk, R. et al. (2018): Fairness in Criminal Justice Risk Assessments](https://arxiv.org/abs/1703.09207) - surveys the family of statistical fairness criteria, including accuracy-based ones, and the trade-offs between them.
* [Barocas, S., Hardt, M., Narayanan, A. (2019): *Fairness and Machine Learning*](https://fairmlbook.org/classification.html) - the standard reference for how classification-based fairness criteria relate to and contradict each other.

*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
