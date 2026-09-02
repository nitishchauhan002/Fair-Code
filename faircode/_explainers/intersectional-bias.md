> *A model can look nearly fair on sex alone and nearly fair on national origin alone, while still denying benefits to foreign-born women at a rate neither single-attribute check would predict. Auditing one protected attribute at a time cannot see this - it quietly averages the intersection back into each attribute's own marginal.*

## The One-Sentence Definition

**Intersectional bias** is a fairness gap that shows up only when two or more protected attributes are checked together - at the group that carries both disadvantages at once - and can be larger than either attribute's own single-axis gap, or even larger than the two gaps added together, a compounding effect Kimberlé Crenshaw (1989) first named in the legal context this framework borrows from.

## Why It Matters

Every fairness metric in this repo - Demographic Parity, Equalized Odds, [Equal Opportunity](equal-opportunity.md), Predictive Parity - is normally computed one protected attribute at a time: sex on its own, race on its own, age on its own. That is also how most real-world audits are run, because it is simpler and matches how protected attributes are usually declared in law and in data.

The problem is mechanical, not a matter of trying harder. A single-axis audit for sex necessarily averages together every national-origin group within "female," and a single-axis audit for national origin averages together every sex within each origin group. If the real harm is concentrated specifically on the group at the intersection - foreign-born women, older Black applicants, young minority men - that averaging can dilute the signal below what either axis alone would flag as a problem, while the group that's actually affected keeps experiencing it in full.

This is not a hypothetical concern. [Buolamwini and Gebru's Gender Shades (2018)](http://gendershades.org/) found commercial facial-analysis systems with near-perfect accuracy on lighter-skinned men and on men overall, and near-perfect accuracy on lighter-skinned subjects overall - yet with error rates for darker-skinned women that neither the race-alone nor the gender-alone numbers predicted. Checking race and checking gender separately would have missed exactly the group actually being failed.

## Concrete Example: Benefits Denial - Audit 05

Audit 05 predicts benefits eligibility from the Adult/Census dataset. Checked separately, sex and national origin (whether an applicant was born in the United States) are each declared protected attributes with their own single-axis Demographic Parity gap - the kind [What Is Demographic Parity?](demographic-parity.md) describes.

The frozen benchmark harness also checks the intersection directly (`paper/results-frozen/results_fairness.csv`, `intersectional_demographic_parity_diff`, baseline gradient boosting model, `sex_x_national_origin`): applicants who are both female *and* not US-native (n=235) are approved at a rate **19.7 percentage points** lower than applicants who are neither (n=3,907) - a 95% CI of [-0.225, -0.166], p < 0.001. That row carries a `superadditive` flag in the frozen results: the gap at the intersection exceeds what the sex-alone gap and the national-origin-alone gap add up to on their own - the compounding effect this explainer describes, not an invented one.

Across every strategy and model the frozen results cover, five of Benefits Denial's six declared attribute-pairs carry that same `superadditive` flag at least once - `sex_x_national_origin`, `sex_x_age`, `national_origin_x_age`, `national_origin_x_race`, and `age_x_race`. Only `sex_x_race` never does. At the baseline stage specifically, it's exactly two rows: `national_origin_x_age` under logistic regression and `sex_x_national_origin` under gradient boosting, the one quoted above - a single-axis audit of either attribute alone would have reported a smaller number for both.

## Detection Code

Splits the population into the four quadrants of two protected attributes - both disadvantaged, both advantaged, and each one alone - so the doubly-disadvantaged cell is visible on its own instead of being averaged into either marginal.

```python
import numpy as np
import pandas as pd


def intersectional_gap(df, outcome_col, attr_a_col, attr_b_col,
                        disadvantaged_a, disadvantaged_b):
    """
    Compares the outcome rate for the group disadvantaged on BOTH attributes
    against the group disadvantaged on NEITHER, alongside each attribute's
    own marginal (single-axis) gap - so a marginal-only audit's blind spot
    is visible directly.

    Parameters:
        df: DataFrame with the outcome and both attribute columns
        outcome_col: column of the binary outcome (1 = positive prediction)
        attr_a_col, attr_b_col: the two protected-attribute columns
        disadvantaged_a, disadvantaged_b: the disadvantaged value for each

    Returns a dict with the intersectional gap, each marginal gap, whether
    the intersectional gap is superadditive (exceeds the marginals' sum),
    and the size of each of the four quadrants.
    """
    a = df[attr_a_col] == disadvantaged_a
    b = df[attr_b_col] == disadvantaged_b

    both, neither = a & b, ~a & ~b
    a_only, b_only = a & ~b, ~a & b

    def rate(mask):
        sub = df.loc[mask, outcome_col]
        return float(sub.mean()) if len(sub) else float("nan")

    intersectional_gap = rate(both) - rate(neither)
    gap_a_alone = rate(a) - rate(~a)
    gap_b_alone = rate(b) - rate(~b)
    superadditive = abs(intersectional_gap) > abs(gap_a_alone) + abs(gap_b_alone)

    return {
        "intersectional_gap": intersectional_gap,
        "gap_a_alone": gap_a_alone,
        "gap_b_alone": gap_b_alone,
        "superadditive": bool(superadditive),
        "cell_sizes": {
            "both": int(both.sum()), "neither": int(neither.sum()),
            "a_only": int(a_only.sum()), "b_only": int(b_only.sum()),
        },
    }


# Usage example:
# result = intersectional_gap(
#     df, outcome_col="approved", attr_a_col="sex", attr_b_col="native_country",
#     disadvantaged_a="Female", disadvantaged_b="foreign_born",
# )
```

## Limitations

### 1. Every extra attribute crossed shrinks the cell it's checking

Two attributes already narrows the population to one quadrant of four; three attributes narrows it to one of eight. Always report the cell size (`n`) next to the gap - a striking number from 12 people is not evidence of anything.

### 2. Superadditive is a description, not a cause

Flagging a gap as superadditive says the intersection is worse than the marginals predict; it does not say why. The mechanism could be a genuine compounding social effect, a proxy variable that happens to correlate with exactly that combination, or a small-sample artifact - each needs different follow-up.

### 3. Which pairs to check is a modeling choice, not something the data decides for you

Checking every pair of declared protected attributes (what this repo's benchmark harness does) still won't surface a triple intersection, and checking pairs at all requires first declaring the individual attributes - an intersectional check inherits every limitation of the single-axis attributes it's built from.

### 4. A non-significant intersectional gap does not clear the model

With a small doubly-disadvantaged cell, a wide confidence interval crossing zero often just means there isn't enough data to tell, not that there's no effect - see [What Is the Base Rate Fallacy?](base-rate-fallacy.md) for the same small-subgroup volatility problem from a different angle.

## Related Concepts

* [What Is a Protected Attribute?](protected-attribute.md) - intersectional bias is what single-axis protected-attribute audits, checked one at a time, cannot see.
* [What Is Demographic Parity?](demographic-parity.md) - the single-axis metric this explainer's concrete example crosses two of.
* [What Is Equal Opportunity (and How It Differs From Equalized Odds)?](equal-opportunity.md) - another case where checking less than the full picture can hide a real disparity.
* [What Is the Base Rate Fallacy?](base-rate-fallacy.md) - the small-subgroup statistics problem that gets sharper once you're down to an intersectional cell.

## Related Projects in This Repo

* [`Benefits Denial/`](../Benefits%20Denial/) - the audit behind the concrete example above, where sex and national origin combine into a superadditive gap.
* [`faircode profile --cross COLA,COLB`](../faircode/cli.py) / the [Open Dataset Profiler](../profiler.html) - checks *representation*, not outcomes: whether an intersectional cell has enough rows to trust any conclusion drawn from it at all, before any outcome gap is even computed. Use it first; `intersectional_gap()` above (and `faircode.significance.intersectional_report`, which the frozen benchmark results in this repo are computed from) checks the outcome gap once the cell is large enough to trust.

## Further Reading

* [Crenshaw, K. (1989): Demarginalizing the Intersection of Race and Sex](https://chicagounbound.uchicago.edu/uclf/vol1989/iss1/8/) - the paper that introduced intersectionality as a framework, in the legal context this statistical version borrows its name and its core claim from: the disadvantage compounds rather than adding up.
* [Buolamwini, J., Gebru, T. (2018): Gender Shades](http://gendershades.org/) - the canonical empirical demonstration in ML: commercial facial-analysis error rates for darker-skinned women that neither the race-alone nor gender-alone numbers predicted.
* [Barocas, S., Hardt, M., Narayanan, A. (2019): *Fairness and Machine Learning*](https://fairmlbook.org/classification.html) - situates intersectional checks within the broader classification-fairness framework this repo's other metrics explainers draw from.

*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
