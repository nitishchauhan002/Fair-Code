> *A hospital readmission model can be 88.7% accurate and still be almost useless - if only 11% of patients are ever readmitted, a model that flags nobody is already 89% "correct." Accuracy and AUC both grade on that curve. Precision and recall don't.*

## The One-Sentence Definition

**A precision-recall (PR) curve** plots a binary classifier's precision (of the cases it flagged positive, how many really were) against its recall (of the cases that really were positive, how many it caught) at every decision threshold, and **average precision (AP)** - the area under that curve - collapses it into one number, the same way AUC collapses a ROC curve; the difference is that PR curves ignore the true negatives ROC curves lean on, which is exactly what makes them the honest picture when positives are rare.

## Why It Matters

Most fairness audits live in exactly the regime PR curves were built for: rare positives, skewed base rates. A hospital readmission, a loan default, a flagged tenant, a denied claim - the outcome an audit cares about is usually the minority class, often by a wide margin. [ROC Curve and AUC](roc-curve-auc.md) already covers why an aggregate ranking score can hide a group-level gap. This explainer covers a second, compounding problem: ROC/AUC's own arithmetic is *lenient about rare positives* in a way that PR/AP is not.

The reason is mechanical, not statistical. A ROC curve's x-axis is the false positive rate - false positives divided by all true negatives. When true negatives vastly outnumber true positives (the readmission case above: 88.8% of patients are *not* readmitted), that denominator is huge, so even a large *number* of false positives barely moves the false positive rate, and AUC stays flat and reassuring. Precision's denominator is different: false positives divided by *all positive predictions*, a much smaller number under class imbalance - so the same false positives that AUC shrugs off can crater precision. A model can hold a "strong" 0.85 AUC while its precision at the threshold you'd actually deploy is close to a coin flip, because almost every case it flags positive turns out to be wrong.

This has a fairness edge, not just an accuracy one. If one subgroup has a rarer positive rate than another - a common, unremarkable fact about real populations, not itself a bias - a shared threshold tuned to the pooled data can land at defensible precision for the majority group and near-worthless precision for the minority one, while the *pooled* AUC never flags a problem, because AUC was never measuring the thing that broke.

## Reading a Precision-Recall Curve

Every point on a PR curve is one threshold, same as ROC - it just plots different axes:

- The **y-axis is precision** - of the cases flagged positive, what fraction really are.
- The **x-axis is recall** (the same quantity as the ROC curve's true positive rate) - of the cases that really are positive, what fraction the model caught.
- There is no fixed "coin flip" diagonal the way ROC has one at AUC 0.5. A random classifier's PR curve sits at a *horizontal* line equal to the population's base rate - the rarer the positive class, the lower that floor sits, and the harder it is to tell a genuinely good classifier from a lucky one by eye.
- The **top-right corner** is a perfect classifier: every positive caught, no false positives, precision and recall both 1.0.

| What AP/PR tells you | What AP/PR does not tell you |
|---|---|
| How well precision holds up as you demand more recall | Whether a "good" AP is good *relative to the base rate* - always compare against it |
| The trade-off you actually face when tuning a threshold under class imbalance | Anything about true negatives - it never rewards you for correctly ignoring the majority class |
| A picture that gets *harder* to look good on as positives get rarer | Group-level gaps - like AUC, it needs to be computed per group to see them |

| Base rate (rare positives) | ROC/AUC | PR/AP |
|---|---|---|
| What moves | False positive rate barely moves - true negatives dominate the denominator | Precision moves a lot - false positives are a large share of a small "flagged positive" pool |
| Random baseline | Always 0.5, regardless of base rate | Equal to the base rate - drops toward 0 as positives get rarer |
| What a "good" score hides | A ranker that's mediocre specifically among the flagged cases | Nothing extra beyond precision/recall themselves - which is the point |

## Concrete Example: Healthcare Readmission - Audit 06

Audit 06 predicts 30-day hospital readmission from the Diabetes 130-US Hospitals dataset (101,766 records). Readmission is rare: only **11.2%** of patients in the dataset were readmitted within 30 days - the other 88.8% were not.

The baseline logistic regression model's frozen performance numbers (`paper/results-frozen/results_performance.csv`) look reasonable at a glance: **88.7% accuracy** and an **AUC of 0.62**. Its **F1 score - the harmonic mean of precision and recall - is 0.039.** F1 stays low because AUC never had to reckon with the base rate: sweeping every threshold shows the model *can* rank cases somewhat sensibly (AUC 0.62, better than a coin flip), but at the threshold that actually gets deployed, the number of false positives it generates is large relative to the tiny pool of true positives, so precision collapses. The random forest and gradient boosting baselines show the same pattern (AUC 0.65 and 0.67, F1 of 0.021 and 0.024) - this isn't one unlucky model family, it's the base rate.

None of this is fairness-specific yet - it's the same trap for everyone in the dataset. The fairness question is whether it traps one group worse than another:

```python
gaps = per_group_average_precision(
    readmission_df, y_true_col="readmitted",
    y_score_col="risk_score", group_col="race",
)
print(gaps)
#           ap  base_rate      n
# White   0.14      0.108  86527
# Black   0.09      0.142  10091
# gap     0.05        NaN    NaN
```

This is illustrative output, not a published result of Audit 06, but it shows the shape of the problem: the Black subgroup has a *higher* true readmission rate (0.142 vs. 0.108) yet a *lower* average precision (0.09 vs. 0.14) - the model's flags are less trustworthy for exactly the group that needs the flag more often. A pooled AUC, computed the way [ROC Curve and AUC](roc-curve-auc.md) describes, would not surface this: it would need a per-group breakdown to catch a ranking gap, and even then it wouldn't show how badly precision degrades once class imbalance is factored back in per group.

## Detection Code

Computes average precision within each group (never just pooled, for the same reason [per-group AUC](roc-curve-auc.md) matters), alongside each group's own base rate - the number every AP score needs to be read against, since AP is not comparable across groups with different base rates the way a 0-to-1 accuracy scale might tempt you to assume.

```python
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


def per_group_average_precision(df, y_true_col, y_score_col, group_col):
    """
    Computes average precision (AP) within each group, plus each group's
    base rate (share of true positives) and the AP gap between the
    highest- and lowest-scoring group.

    AP is only meaningful next to the base rate it was computed against -
    a "low" AP for a group with a rare positive class may still be well
    above that group's own random-baseline floor, and a "high" AP for a
    group with a common positive class may be barely above its floor.

    Parameters:
        df: DataFrame with true labels, model scores, and group membership
        y_true_col: column of the ground-truth binary label (1 = positive)
        y_score_col: column of the model's continuous score or probability
        group_col: column of the protected attribute or group label

    Returns a DataFrame indexed by group (plus a "gap" row) with ap,
    base_rate, n.
    """
    rows = []
    for group, sub in df.groupby(group_col):
        if sub[y_true_col].nunique() < 2:
            ap = float("nan")  # undefined with only one class present
        else:
            ap = average_precision_score(sub[y_true_col], sub[y_score_col])
        rows.append({
            "group": group,
            "ap": ap,
            "base_rate": sub[y_true_col].mean(),
            "n": len(sub),
        })

    result = pd.DataFrame(rows).set_index("group")
    result.loc["gap"] = [
        result["ap"].max() - result["ap"].min(), float("nan"), float("nan")
    ]
    return result


def group_pr_points(df, y_true_col, y_score_col, group_col):
    """
    Returns, per group, the (precision, recall, thresholds) arrays for
    precision_recall_curve. Overlay these against each group's own
    base-rate floor (a flat line at that group's positive share) to see
    how much each curve actually rises above chance.
    """
    curves = {}
    for group, sub in df.groupby(group_col):
        if sub[y_true_col].nunique() < 2:
            continue
        precision, recall, thresholds = precision_recall_curve(
            sub[y_true_col], sub[y_score_col]
        )
        curves[group] = {
            "precision": precision, "recall": recall, "thresholds": thresholds,
            "base_rate": sub[y_true_col].mean(),
        }
    return curves


# Usage example
# gaps = per_group_average_precision(readmission_df, "readmitted", "risk_score", "race")
# curves = group_pr_points(readmission_df, "readmitted", "risk_score", "race")
```

## Limitations

### 1. AP is not comparable across groups with different base rates

An AP of 0.3 is excellent for a group with a 5% base rate and mediocre for a group with a 40% base rate - always report each group's base rate alongside its AP, never AP alone.

### 2. Small groups make AP noisy, and rare-positive groups make it worse

AP is computed over the positive cases specifically, so a subgroup that is both small *and* has a rare positive class can end up with an AP built on a handful of true positives. Report the count of positive cases per group, not just total n, and bootstrap a confidence interval before treating a gap as real.

### 3. A single summary number still hides the threshold you deploy

Same limitation as AUC: AP integrates over every threshold, but only one threshold ever ships. Report precision and recall *at the operating threshold*, by group, alongside any AP summary - see [False Positives vs. False Negatives](false-positives-vs-false-negatives.md) for that per-threshold view.

### 4. Equal AP across groups is not equal treatment

Two groups can have identical AP while one group's curve reaches that area through high precision at low recall and the other's through the reverse - meaning the two groups experience very different trade-offs at any shared threshold, even though the summary number matches.

## Related Concepts

* [ROC Curve and AUC](roc-curve-auc.md) - the other threshold-free ranking summary, and why its own arithmetic is lenient about rare positives in exactly the way this explainer describes.
* [False Positives vs. False Negatives in Medical Risk Models](false-positives-vs-false-negatives.md) - the per-threshold error breakdown that both AUC and AP average over.
* [What Is Class Imbalance?](class-imbalance.md) - the underlying condition (skewed positive/negative ratios) that makes the ROC-vs-PR distinction matter in the first place.
* [What Is Predictive Parity?](predictive-parity.md) - a fairness metric built directly from precision, and the COMPAS dispute over whether equalizing it is enough.

## Related Projects in This Repo

* [`Healthcare Readmission/`](../Healthcare%20Readmission/) - the audit behind the concrete example above, where an 11.2% base rate turns a "reasonable" AUC into a near-worthless F1.
* [`Benefits Denial/`](../Benefits%20Denial/) - a second audit with a skewed target (24% positive rate) where the same ROC-vs-PR gap is worth checking.

## Further Reading

* [Davis, J., Goadrich, M. (2006): The Relationship Between Precision-Recall and ROC Curves](https://dl.acm.org/doi/10.1145/1143844.1143874) - the paper that formally connects the two curve families and proves a curve dominating in ROC space also dominates in PR space, while showing why their *visual* impression of "how good" a model looks can differ sharply under imbalance.
* [Saito, T., Rehmsmeier, M. (2015): The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) - a direct empirical demonstration of the exact mechanism this explainer describes: ROC curves that look consistently strong across imbalance levels while the matching PR curves reveal deteriorating precision.
* [Barocas, S., Hardt, M., Narayanan, A. (2019): *Fairness and Machine Learning*](https://fairmlbook.org/classification.html) - the classification chapter's discussion of base-rate differences across groups applies directly here: a shared threshold cannot equalize precision across groups with different base rates without giving something else up.

*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
