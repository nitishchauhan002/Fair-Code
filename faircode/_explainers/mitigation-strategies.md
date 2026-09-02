> *Where in the machine learning pipeline you intervene - before training, during training, or after training - changes what a fairness fix can and can't do, and the three approaches don't produce interchangeable results even when they target the exact same gap.*

## The One-Sentence Definition

**Pre-, in-, and post-processing** are the three points in a model pipeline where a fairness intervention can happen: pre-processing changes the training data before the model ever sees it, in-processing bakes a fairness constraint into the training process itself, and post-processing adjusts the model's outputs after it has already been trained.

## Why It Matters

"Mitigate the bias" is not one technique - it's a choice of *when* to intervene, and that choice has real consequences independent of which specific method you pick within it. Pre-processing is the simplest to apply and needs no special training infrastructure, but it can only remove what the data explicitly encodes, not what a model learns to reconstruct from what's left. In-processing can enforce a fairness constraint directly during optimization, but it requires retraining access to the model - not always available with a third-party or already-deployed system. Post-processing needs no retraining at all, but it requires knowing each individual's group membership at prediction time, which is sometimes exactly the information that's illegal or impractical to collect.

This repo's own benchmark harness is built entirely around this taxonomy - it applies the same five-strategy ladder to every audit, so "does this fix work" always means "compared to which stage of the pipeline."

## Core Concept: This Repo's S0-S4 Ladder

`faircode/strategies.py` implements five strategies in strictly increasing order of intervention:

| Strategy | Stage | What it does |
|---|---|---|
| S0 `baseline` | none | Every feature, including the protected attribute, used as-is. |
| S1 `unawareness` | pre-processing | Drop the protected attribute column only. |
| S2 `unawareness_proxy_removal` | pre-processing | Drop the protected attribute *and* its known proxies. |
| S3 `in_processing` | in-processing | Fairlearn `ExponentiatedGradient` (Agarwal et al. 2018) - trains under a Demographic Parity constraint, on the same reduced feature set as S2. |
| S4 `post_processing` | post-processing | Fairlearn `ThresholdOptimizer` (Hardt, Price & Srebro 2016 style) - fits per-group decision thresholds after training, on the same reduced feature set as S2. |

S1 alone is [fairness through unawareness](fairness-through-unawareness.md) - the intuition that removing the protected attribute is sufficient - and this repo's own results are the direct rebuttal to that intuition, shown below. S3 and S4 deliberately train on the *same* feature set as S2, not the full S0 feature set, so any further improvement they show over S2 can be attributed to the constraint or the threshold adjustment itself, not to seeing different data.

## Concrete Example: COMPAS - Audit 01

The demographic parity gap for the baseline logistic regression model on race, across all five strategies, using the frozen numbers exactly as `faircode/benchmark.py` computed them (`paper/results-frozen/results_fairness.csv`):

| Strategy | Demographic Parity Diff | 95% CI | p-value |
|---|---:|---|---:|
| S0 baseline | 0.855 | [0.839, 0.871] | 0.0 |
| S1 unawareness | 0.172 | [0.139, 0.206] | 0.0 |
| S2 unawareness_proxy_removal | 0.115 | [0.088, 0.144] | 0.0 |
| S3 in_processing | -0.014 | [-0.040, 0.011] | 0.3305 |
| S4 post_processing | 0.023 | [-0.009, 0.056] | 0.1485 |

Three things this progression shows that a single before/after comparison would miss:

**Dropping the protected attribute alone (S0 → S1) does most of the work, but leaves a real gap.** The 85.5-point baseline gap falls to 17.2 points - a huge improvement - but 17.2 points is still a large, clearly significant disparity. This is exactly the failure mode [Fairness Through Unawareness](fairness-through-unawareness.md) warns about: "we removed race" does not mean "we removed race's effect."

**Removing proxies too (S1 → S2) helps further, but not proportionally as much.** Another 5.7 points close, to 11.5 - a real but much smaller gain than the first step, since not every remaining correlate of race was captured by the defined proxy list.

**The constraint-based methods (S3, S4) land at or below S2, and stop being statistically significant.** S3's gap is actually slightly negative (-1.4 points) and S4's is 2.3 points - both with confidence intervals that cross zero and p-values well above 0.05. Both strategies converge to roughly the same residual gap that simple proxy removal already reached, not a dramatically better one. That convergence is the basis for treating this residual as a floor characteristic of the data and task, rather than a limitation of any one specific mitigation method - three different approaches (drop proxies, constrain training, adjust thresholds) independently landing in the same place is stronger evidence than any one of them alone.

## Detection Code

Compares a metric across mitigation stages and flags whether later strategies actually helped, using the same held-out predictions pattern the other explainers in this repo use.

```python
import pandas as pd


def summarize_mitigation_ladder(results_df, audit, metric, protected_attribute, model="logistic_regression"):
    """
    Extracts one metric's progression across the five strategies for a
    single audit/model/protected_attribute combination, in the fixed
    S0-S4 order, and flags whether each step actually reduced the gap.

    Parameters:
        results_df: a DataFrame shaped like paper/results-frozen/results_fairness.csv
        audit, metric, protected_attribute, model: filter values

    Returns a DataFrame with one row per strategy, in S0-S4 order, plus an
    `improved_since_previous` column (None for the first row).
    """
    order = ["baseline", "unawareness", "unawareness_proxy_removal",
             "in_processing", "post_processing"]

    subset = results_df[
        (results_df["audit"] == audit)
        & (results_df["metric"] == metric)
        & (results_df["protected_attribute"] == protected_attribute)
        & (results_df["model"] == model)
    ].set_index("strategy").reindex(order)

    improved = [None]
    for prev, curr in zip(subset["value"], subset["value"].iloc[1:]):
        improved.append(abs(curr) < abs(prev))

    subset = subset.reset_index()
    subset["improved_since_previous"] = improved
    return subset[["strategy", "value", "ci_low", "ci_high", "p_value", "improved_since_previous"]]


# Usage example:
# import pandas as pd
# results = pd.read_csv("paper/results-frozen/results_fairness.csv")
# print(summarize_mitigation_ladder(results, "compas", "demographic_parity_diff", "race"))
```

## Limitations

### 1. In-processing needs retraining access, which isn't always available

`ExponentiatedGradient` requires fitting the model under the constraint from scratch. A team working with a third-party model, a legacy system, or a model that's already deployed and expensive to retrain cannot apply this strategy at all - only pre- or post-processing remain options.

### 2. Post-processing needs group membership at prediction time

`ThresholdOptimizer` must know which group each individual belongs to *at inference*, to apply the right per-group threshold. In many regulated domains (credit, employment), collecting or using protected-attribute data at decision time is restricted or illegal even when it was available during training - making post-processing legally unavailable in exactly the settings where it's most tempting.

### 3. None of the three approaches monotonically improves every metric

The COMPAS example shows real progress on demographic parity, but a strategy tuned to close one metric's gap can leave another metric's gap unchanged or worse - this repo's benchmark harness computes all six metrics for every strategy for exactly this reason, since a mitigation report using only one metric can miss that trade-off entirely.

### 4. A near-zero gap after mitigation isn't the same as a fair model

S3 and S4 close the *measured* demographic parity gap, but neither one touches the underlying label quality, feature construction, or the fact that the model is being used for this decision at all. See [What Is Label Bias?](label-bias.md).

## Related Concepts

* [What Is Fairness Through Unawareness?](fairness-through-unawareness.md) - the S1 strategy on its own, and why the 17.2-point residual gap above is the direct evidence against relying on it alone.
* [What Is Demographic Parity?](demographic-parity.md) - the metric `ExponentiatedGradient`'s constraint directly targets in this repo's S3 strategy.
* [Proxy Variables](proxy-variables.md) - what S2's proxy-removal step is actually trying to remove, and why it can never remove all of it.

## Related Projects in This Repo

* [`faircode/strategies.py`](../faircode/strategies.py) - the actual S0-S4 implementation this explainer describes.
* [`COMPAS/`](../COMPAS/) - the audit behind the five-strategy progression quoted above, and the one already cited on the project homepage.

## Further Reading

* [Agarwal, A. et al. (2018): A Reductions Approach to Fair Classification](https://arxiv.org/abs/1803.02453) - the paper behind `ExponentiatedGradient`, this repo's in-processing strategy.
* [Hardt, M., Price, E., Srebro, N. (2016): Equality of Opportunity in Supervised Learning](https://arxiv.org/abs/1610.02413) - the paper behind the threshold-adjustment approach `ThresholdOptimizer` implements.
* [Friedler, S. et al. (2019): A Comparative Study of Fairness-Enhancing Interventions in Machine Learning](https://arxiv.org/abs/1802.04422) - an empirical comparison across the same three intervention families, on different datasets.

*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
