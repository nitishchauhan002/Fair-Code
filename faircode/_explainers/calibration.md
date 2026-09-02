# Explainer: What is Calibration?

> *The reason a model can be "equally accurate" for everyone - and still treat them unequally.*

---

## The One-Sentence Definition

A model is **calibrated** for a group if its predicted probabilities actually match real-world outcomes - so when it says "70% risk," roughly 70% of people in that group actually experience the outcome.

---

## Why It Matters

Most people evaluate AI models on accuracy. Calibration is a different question: *do the scores mean the same thing across demographic groups?*

If a risk-scoring system assigns a "70% recidivism risk" score to Black defendants and a "70% risk" score to White defendants, calibration asks: does 70% actually mean 70% for *both* groups? If the model is miscalibrated for one group - say, 70% translates to 50% actual recidivism for White defendants but 80% for Black defendants - then identical scores are producing fundamentally different real-world stakes depending on who receives them.

This is called **differential calibration**, and it's one of the central problems exposed by the ProPublica COMPAS investigation. It's also why calibration is often in direct conflict with other fairness metrics like equalized odds - you frequently cannot satisfy both at the same time.

---

## Concrete Example: COMPAS Recidivism Scores

The COMPAS risk tool, used in U.S. courts to inform bail and sentencing decisions, became the centre of a landmark fairness debate in 2016. ProPublica found that the tool made systematically different kinds of errors for Black and White defendants.

Northpointe (the tool's developer) responded that COMPAS *was* calibrated: among defendants scored "high risk," roughly the same proportion reoffended regardless of race. That claim is true by the numbers. But calibration alone masked a different problem - the tool reached that calibration through asymmetric error rates. Black defendants were far more likely to be falsely labelled high risk (false positives), while White defendants were more likely to be falsely labelled low risk (false negatives).

This is the [Chouldechova (2017)](https://arxiv.org/abs/1703.00056) result in practice: **when base rates differ between groups, you cannot simultaneously achieve calibration and equal false positive/false negative rates.** You have to choose. COMPAS chose calibration. ProPublica measured the error rates. Both were right about what they measured.

We can verify the calibration gap directly using the ProPublica dataset:

```python
import pandas as pd

df = pd.read_csv('compas-scores-two-years.csv')
df = df[df['score_text'].isin(['Low', 'Medium', 'High'])]
df = df[df['race'].isin(['African-American', 'Caucasian'])]

# Map score bands to numeric risk tiers
score_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['score_tier'] = df['score_text'].map(score_map)

# Actual recidivism rate per score band per race
calibration_table = df.groupby(['race', 'score_text'])['two_year_recid'].mean().round(3)
print(calibration_table)
```

**Sample output:**

| Race | Score Band | Actual Recidivism Rate |
|---|---|---|
| African-American | Low | 0.33 |
| African-American | Medium | 0.53 |
| African-American | High | 0.67 |
| Caucasian | Low | 0.21 |
| Caucasian | Medium | 0.45 |
| Caucasian | High | 0.63 |

The High band converges (0.67 vs 0.63) - that's the calibration Northpointe pointed to. But the Low band reveals the asymmetry: a "Low risk" score means something very different for a Black defendant (33% actual recidivism) than for a White defendant (21%). The same label, different stakes.

---

## How to Measure Calibration

Calibration is measured by comparing predicted probabilities to actual outcome rates within prediction bins. The standard approach is a **calibration curve** (also called a reliability diagram).

```python
import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def measure_calibration_by_group(df, feature_cols, target_col, group_col, n_bins=5):
    """
    Train a classifier and plot calibration curves separately per group.
    A well-calibrated model produces a curve close to the diagonal.
    Divergence between groups indicates differential calibration.
    """
    X = pd.get_dummies(df[feature_cols])
    y = df[target_col]
    groups = df[group_col]

    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, groups, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    results = {}
    for group in g_test.unique():
        mask = g_test == group
        if mask.sum() < 20:
            continue
        fraction_pos, mean_pred = calibration_curve(
            y_test[mask], probs[mask], n_bins=n_bins
        )
        results[group] = {
            'mean_predicted': mean_pred,
            'fraction_positive': fraction_pos
        }

    return results


def calibration_gap(results):
    """
    Compute the mean absolute calibration error (MACE) per group.
    Higher = worse calibration for that group.
    """
    for group, data in results.items():
        error = np.mean(np.abs(data['fraction_positive'] - data['mean_predicted']))
        print(f"{group}: Mean Absolute Calibration Error = {error:.4f}")


# Example usage with COMPAS
feature_cols = ['sex', 'age_cat', 'priors_count', 'c_charge_degree']
results = measure_calibration_by_group(
    df, feature_cols, 'two_year_recid', 'race'
)
calibration_gap(results)
```

A perfectly calibrated model produces a diagonal line from (0, 0) to (1, 1). A model that is calibrated for one group but not another will show one group's curve hugging the diagonal while the other curves away - the gap between those curves is the differential calibration.

---

## Limitations and Trade-offs

Calibration is not a complete picture of fairness, and it has well-documented limitations.

**1. Calibration can coexist with discriminatory error patterns.**
A model can be calibrated across groups while still generating far more false positives for one group than another. Northpointe's defence of COMPAS is the canonical example. Calibration and equalised false positive/false negative rates are mathematically incompatible when base rates differ between groups - this is the Chouldechova (2017) impossibility result. See the [Fairness Metric Conflicts explainer](fairness-metric-conflicts.md) for the full proof.

**2. Calibration is sensitive to base rate differences.**
If Group A has a 20% base rate and Group B has a 40% base rate for the outcome, a model can be well-calibrated for both groups and still assign systematically higher scores to Group B - scores that reflect historical data patterns, not individual risk.

**3. Calibration says nothing about whether the outcome itself is fair.**
A model perfectly calibrated on arrest rates is calibrated on a measure that reflects over-policing, not actual crime. The fairness of the calibration target matters as much as the calibration itself.

**4. Small group sizes inflate calibration error estimates.**
With fewer samples, calibration curves are noisier. Always check sample size per bin before drawing conclusions about a subgroup's calibration.

---

## Related Concepts

- [Proxy Variables](proxy-variables.md) - features that encode protected attributes and corrupt training data before calibration is even measured
- [Equalized Odds](equalized-odds.md) - the metric that measures false positive and false negative rates across groups; often in direct conflict with calibration
- [Fairness Metric Conflicts](fairness-metric-conflicts.md) - the mathematical proof that calibration, demographic parity, and equalized odds cannot all be satisfied simultaneously when base rates differ
- [COMPAS/](../COMPAS/) - the audit where calibration vs. equalized odds plays out in full with real data

---

## Further Reading

- [ProPublica: Machine Bias (2016)](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing) - the original investigation that triggered the calibration vs. error rate debate
- [Chouldechova, A.: Fair Prediction with Disparate Impact (2017)](https://arxiv.org/abs/1703.00056) - the paper that proved the mathematical incompatibility between calibration and equal error rates
- [Barocas, Hardt & Narayanan: Fairness and Machine Learning (2023)](https://fairmlbook.org/) - Chapter 3 covers calibration in depth with formal definitions

---

*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
