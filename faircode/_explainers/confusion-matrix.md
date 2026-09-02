> *A model can boast 90% accuracy while hiding the fact that its 10% error rate consists entirely of falsely flagging innocent people from a minority group. A confusion matrix splits a single accuracy number into the four outcomes that actually matter, revealing exactly who pays for a model's mistakes.*

## The One-Sentence Definition

**A confusion matrix** is a table that summarizes a classification model's performance by breaking its predictions down into true positives, true negatives, false positives, and false negatives - the four foundational numbers that drive almost every algorithmic fairness metric.

## Why It Matters

In high-stakes systems, the type of mistake a model makes is just as important as how often it makes it. A false positive (a false alarm) and a false negative (a missed case) rarely carry the same cost. 

Accuracy alone is a dangerous metric for fairness because it aggregates all correct predictions and all errors into a single percentage. A model that perfectly predicts outcomes for a 90% majority group but misclassifies everyone in a 10% minority group will still score 90% overall accuracy. By calculating a confusion matrix separately for each demographic group, you stop asking "is the model accurate?" and start asking "how does the model fail, and who absorbs the cost of those failures?"

## The Core Concept

A binary classification model makes a prediction (Positive or Negative), and the real world provides the actual truth (Positive or Negative). This creates a 2x2 grid of possibilities:

| | Predicted Positive (1) | Predicted Negative (0) |
|---|---|---|
| **Actual Positive (1)** | **True Positive (TP)**: Correctly flagged | **False Negative (FN)**: Incorrectly missed |
| **Actual Negative (0)** | **False Positive (FP)**: Incorrectly flagged | **True Negative (TN)**: Correctly cleared |

From these four quadrants, we derive the rates that define most fairness metrics:

*   **True Positive Rate (TPR / Recall / Sensitivity)**: TP / (TP + FN). Of all the people who actually experienced the event, what fraction did the model catch?
*   **False Positive Rate (FPR)**: FP / (FP + TN). Of all the people who did not experience the event, what fraction did the model incorrectly flag?
*   **False Negative Rate (FNR)**: FN / (FN + TP). Of all the people who actually experienced the event, what fraction did the model miss?
*   **Positive Predictive Value (PPV / Precision)**: TP / (TP + FP). Of all the people the model flagged, what fraction actually experienced the event?

## Concrete Example: COMPAS - Audit 01

The COMPAS recidivism algorithm perfectly illustrates why confusion matrices matter for fairness. Northpointe, the creator of COMPAS, pointed out that their model had equal **Positive Predictive Value** for Black and White defendants - a risk score of 7 meant roughly the same likelihood of reoffending for both groups.

However, ProPublica looked at the confusion matrix and calculated the **False Positive Rate** and **False Negative Rate**. They found that Black defendants who did not reoffend were falsely labeled as high-risk (False Positives) at nearly twice the rate of White defendants who did not reoffend. Conversely, White defendants who did reoffend were incorrectly labeled as low-risk (False Negatives) much more often than Black defendants.

Because the underlying base rates of arrest were different for the two groups, it was mathematically impossible for the model to have both equal Positive Predictive Value and equal False Positive/Negative Rates. The confusion matrix made this trade-off visible, sparking the modern debate on algorithmic fairness.

## Detection Code

This code computes the four core values (TP, TN, FP, FN) and the key derived rates for each demographic group, allowing you to spot disparities in how errors are distributed.

```python
import pandas as pd
from sklearn.metrics import confusion_matrix

def group_confusion_metrics(df, y_true_col, y_pred_col, group_col):
    """
    Computes a confusion matrix and derived rates for each group.
    
    Parameters:
        df: DataFrame with true labels, model predictions, and group membership
        y_true_col: column name of the ground-truth binary label (1 = positive)
        y_pred_col: column name of the model's binary prediction (1 = positive)
        group_col: column name of the protected attribute or group label
        
    Returns a DataFrame indexed by group with TP, TN, FP, FN, TPR, FPR, FNR, and PPV.
    """
    rows = []
    for group, sub in df.groupby(group_col):
        # Ensure labels are binary [0, 1] for reliable unraveling
        tn, fp, fn, tp = confusion_matrix(
            sub[y_true_col], sub[y_pred_col], labels=[0, 1]
        ).ravel()
        
        # Calculate rates with zero-division protection
        tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
        fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
        ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        
        rows.append({
            "group": group,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "TPR (Recall)": tpr, 
            "FPR": fpr, 
            "FNR": fnr, 
            "PPV (Precision)": ppv,
            "n": len(sub)
        })
        
    return pd.DataFrame(rows).set_index("group")

# Usage example
# metrics_df = group_confusion_metrics(compas_df, "two_year_recid", "high_risk_prediction", "race")
# print(metrics_df[["FPR", "FNR", "PPV (Precision)"]])
```

## Limitations

### 1. Requires a single decision threshold

A confusion matrix evaluates a model at one specific threshold (e.g., probability > 0.5 means positive). If you change the threshold, all four numbers in the matrix change. To evaluate how a model performs across all possible thresholds, you need a [ROC Curve](roc-curve-auc.md).

### 2. Counts all errors equally

The standard confusion matrix treats every False Positive as identical to every other False Positive. It does not know if a model barely missed the threshold (0.49) or was wildly wrong (0.01). It also does not inherently weigh the human cost of a False Positive against a False Negative.

### 3. The Base Rate Problem

When two demographic groups have different base rates (prevalence) of the target variable in the dataset, it is mathematically impossible to equalize all metrics derived from the confusion matrix at the same time. You often have to choose between equalizing predictive value (calibration) and equalizing error rates (Equalized Odds).

## Related Concepts

* [False Positives vs. False Negatives in Medical Risk Models](false-positives-vs-false-negatives.md) - a deep dive into why the two error types carry vastly different real-world costs.
* [Equalized Odds](equalized-odds.md) - the fairness metric that demands equal False Positive and True Positive Rates across groups.
* [Predictive Parity](predictive-parity.md) - the fairness metric that focuses on equal Positive Predictive Value (Precision).
* [What Is a ROC Curve and AUC?](roc-curve-auc.md) - how to evaluate a model's ranking ability across all possible thresholds, independent of a single confusion matrix.

## Related Projects in This Repo

* [`COMPAS/`](../COMPAS/) - the recidivism audit that highlights the conflict between equalizing Positive Predictive Value and equalizing False Positive Rates.
* [`Healthcare Readmission/`](../Healthcare%20Readmission/) - an audit demonstrating how a model can maintain high overall accuracy while concentrating its False Negatives in a specific demographic group.

## Further Reading

* [Chouldechova, A. (2017): Fair Prediction with Disparate Impact](https://arxiv.org/abs/1703.00056) - the mathematical proof showing why a confusion matrix cannot simultaneously satisfy predictive parity and equal error rates when base rates differ.
* [Kleinberg, J., Mullainathan, S., Raghavan, M. (2016): Inherent Trade-Offs in the Fair Determination of Risk Scores](https://arxiv.org/abs/1609.05807) - a foundational paper formalizing the impossibilities of satisfying all fairness metrics derived from the confusion matrix.

*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
