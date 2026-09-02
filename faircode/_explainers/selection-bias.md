# What Is Selection Bias?

> *A dataset does not remember the people who were turned away before it was collected.*

---

## The One-Sentence Definition

**Selection bias** occurs when the process that decides whether a unit ends up in a dataset at all is correlated with the outcome being studied, so the sample a model is trained and audited on is not a fair stand-in for the full population, even before any protected attribute or proxy variable is considered.

---

## Why It Matters

Every audit in this repo starts from a CSV file. `unfair.py` loads it, `fair.py` drops a few columns, and both report a fairness gap on the rows in front of them. That workflow, and almost every fairness technique built on top of it, quietly assumes the rows in front of you are the population you care about, minus some random noise.

Selection bias breaks that assumption before the first line of code runs. If getting into the dataset at all - getting hospitalized, getting arrested, getting approved for a loan, surviving to follow-up - depends on the same outcome the model is later trained to predict, then the sample is shaped by that outcome from the start. No amount of dropping protected attributes or proxy features fixes this, because the distortion is not in a column. It is in which rows exist.

This repo's [`sampling-bias.md`](sampling-bias.md) explainer already lists "selection bias" as one row in its table of representation problems, alongside underrepresentation and survivorship bias, framed as data collected from a non-random subset of the population. That framing is correct as far as it goes. This explainer goes one level deeper: it treats selection bias as a specific causal mechanism - conditioning on a downstream gate that itself depends on the outcome - and shows why that mechanism can produce a biased model even when the sample looks perfectly balanced by every demographic check you can run on the data you have, because the bias lives in the applicants, patients, or defendants who never made it into any row at all.

---

## How It Works

### Conditioning on a Gate

Call the full population `U`. Some selection indicator `S` determines whether a unit becomes a visible row: `S = 1` if it is observed, `S = 0` if it is not. Every dataset in existence is really `U` filtered down to `{S = 1}`. That filtering is harmless for a model as long as `S` is independent of the outcome `Y` and the features `X`. It stops being harmless the moment `S` depends on `Y`, directly or through some factor correlated with `Y`, because then:

`P(Y | X, S = 1) ≠ P(Y | X)`

The relationship the model learns from the observed rows is no longer the relationship that holds in the full population. Nothing about the features themselves needs to be wrong for this to happen. The distortion survives even a perfectly clean feature set, because it was never a feature problem.

### Berkson's Paradox

The classic illustration is a hospital admissions study. Two diseases are truly independent in the general population. But if either disease alone can be severe enough to trigger hospital admission, then among *hospitalized* patients the two diseases become negatively correlated: a patient sick enough to be admitted for one disease needed less help from the second one to clear the admission bar. The correlation is not in the population. It is manufactured entirely by conditioning on who got admitted. This is Berkson's paradox, first described in exactly this hospital-data context, and it is the same mechanism at work any time a dataset is built from "people who cleared some gate" rather than "people, period."

### Reject Inference in Credit Scoring

Credit scoring has its own long-documented name for this: **reject inference**. A bank's historical loan file only contains people the bank already chose to fund. Applicants who were turned down never get a repayment outcome recorded, because they were never given a loan to repay or default on. A scorecard trained on that file learns what "good credit" looks like exclusively among people someone already trusted enough to approve - and has no information whatsoever about how the rejected applicants would have performed, because that population was excluded by construction, not by chance.

---

## Concrete Example: German Credit Lending - Audit 03

[`German Credit Lending/credit_customers.csv`](../German%20Credit%20Lending/) is the dataset behind Audit 03 in this repo. Its `class` column takes exactly two values across all 1,000 rows: `good` (700 rows) and `bad` (300 rows). There is no third value for "denied," "never funded," or "rejected before underwriting," because an applicant who was turned away before receiving a loan generates no repayment history and so was never entered as a row in the first place. The dataset is, by construction, a reject-inference sample: everyone in it already cleared the original approval gate.

`unfair.py` reports a 7.16 percentage point good-credit gap between older and younger applicants, traced in that audit to `employment` (tenure) acting as an age proxy, closing to 1.89 points (a 73.6% reduction) once `age` and `employment` are dropped in `fair.py`. That result is real and it is exactly what a proxy-variable audit is supposed to find. But it describes bias **among the 1,000 people who were already approved for credit at some point in the past**. It says nothing about whether the original human loan officers who decided who got into this dataset at all applied a different approval bar to younger applicants before Audit 03's data collection ever began. If they did, that gate is invisible to both `unfair.py` and `fair.py` - it is not a column either script can drop, because it operated on the applicants that never became rows.

Even the base rate is a clue worth naming honestly rather than a proof: 70% of this dataset is labeled `good` credit, and 37.1% of applicants are under 30. A pool built by re-surveying the same people who applied for credit, rather than the people a bank already chose to fund, would not necessarily reproduce either number. Neither figure by itself proves an upstream gate distorted this sample - that is genuinely unknowable from these 1,000 rows alone - but both are exactly the pattern reject-inference researchers have flagged as a standing limitation of scorecards trained purely on booked-loan data.

---

## Detection Code

Selection bias cannot be detected by inspecting the rows you have more closely - that is the entire problem. The two things you *can* do are (1) prove the mechanism is real with a controlled simulation, and (2) build a habit of checking your sample's outcome rate against an independent external reference whenever one exists.

```python
import numpy as np
import pandas as pd


def simulate_selection_bias(n=20000, selection_strength=2.5, seed=42):
    """
    Demonstrates Berkson's paradox / collider bias directly: two features
    are generated fully independent of each other in the full population,
    but a downstream selection gate that depends on both of them is applied
    before the "dataset" is assembled - exactly like a loan-approval or
    hospital-admission decision. Compares the correlation in the full
    population against the correlation in the selected-only sample.

    Parameters:
        n: size of the full (unobserved) population.
        selection_strength: how strongly the gate depends on the two
            features (0 reproduces a random sample, larger values reproduce
            a stricter, more outcome-dependent gate).
        seed: random seed for reproducibility.

    Returns:
        dict with the population correlation, the selected-sample
        correlation, and the fraction of the population that was retained.
    """
    rng = np.random.default_rng(seed)

    feature_a = rng.normal(0, 1, n)
    feature_b = rng.normal(0, 1, n)  # independent of feature_a by construction

    gate_score = selection_strength * (feature_a + feature_b) + rng.normal(0, 1, n)
    selected = gate_score > np.quantile(gate_score, 0.7)  # only the top 30% clear the gate

    population_corr = np.corrcoef(feature_a, feature_b)[0, 1]
    selected_corr = np.corrcoef(feature_a[selected], feature_b[selected])[0, 1]

    return {
        "population_correlation": round(float(population_corr), 4),
        "selected_sample_correlation": round(float(selected_corr), 4),
        "fraction_retained": round(float(selected.mean()), 4),
    }


def check_outcome_rate_against_reference(df, outcome_col, positive_value, reference_rate, tolerance=0.05):
    """
    Flags when a dataset's observed positive-outcome rate diverges sharply
    from an independent external reference rate (e.g. a regulator's
    published approval rate, a prior audit, or a census figure). A large
    gap does not prove selection bias occurred, but it is exactly the
    signature an invisible upstream gate would leave behind, and it is the
    only check available when the excluded population left no row to
    inspect directly.

    Parameters:
        df: DataFrame containing the outcome column.
        outcome_col: name of the outcome column.
        positive_value: the value counted as the "positive" outcome.
        reference_rate: an independently sourced expected positive rate.
        tolerance: acceptable absolute deviation before flagging (default 0.05).

    Returns:
        dict with the observed rate, the reference rate, the gap, and a flag.
    """
    observed_rate = (df[outcome_col] == positive_value).mean()
    gap = observed_rate - reference_rate

    return {
        "observed_rate": round(float(observed_rate), 4),
        "reference_rate": round(float(reference_rate), 4),
        "gap": round(float(gap), 4),
        "flag": "POSSIBLE SELECTION GATE" if abs(gap) > tolerance else "WITHIN TOLERANCE",
    }


# Usage example
# bias_demo = simulate_selection_bias()
# print(bias_demo)
# # Independent features (population_correlation ~ 0) become correlated
# # (selected_sample_correlation << 0) purely from conditioning on the gate.
#
# df = pd.read_csv("German Credit Lending/credit_customers.csv")
# result = check_outcome_rate_against_reference(
#     df, outcome_col="class", positive_value="good", reference_rate=0.55
# )
# print(result)
```

Running `simulate_selection_bias()` with the defaults produces a population correlation near 0.00 and a selected-sample correlation in the range of -0.5 to -0.6: two features built to be fully independent become strongly, artificially anti-correlated purely because both were inputs to the same selection gate. That is Berkson's paradox reproduced on demand, and it is the mechanism, not a coincidence of this particular random seed.

---

## Limitations and Trade-offs

### 1. Selection Bias Is Invisible From Inside the Sample

By definition, the excluded units leave no row to inspect. Every check above either simulates the mechanism on synthetic data or compares the sample against an external reference - there is no query you can run against the 1,000 rows in `credit_customers.csv` alone that proves or disproves whether an upstream gate distorted them, because the evidence would have to come from the applicants who are not there.

### 2. The Line Between Selection Bias and Sampling Bias Is Blurry in Practice

Both failure modes can look identical from inside a single dataset: a group is underrepresented, or a base rate looks too clean. Treat this explainer's collider/gate framing and [`sampling-bias.md`](sampling-bias.md)'s representation framing as two complementary lenses on the same family of problems, not as competing diagnoses that need to be resolved before an audit can proceed.

### 3. Statistical Corrections Lean on Unverifiable Assumptions

Reject inference and Heckman-style two-step corrections exist precisely because the missing population cannot simply be collected after the fact. Both approaches work by assuming a specific statistical relationship between the selection gate and the outcome for the excluded units - an assumption that is inherently untestable with the data on hand, because it is a claim about the very rows that were never observed.

### 4. A Skewed Base Rate Is a Hint, Not a Verdict

A 70/30 good/bad split, or any other base rate that looks unusually clean, can come from selection bias, from a genuinely careful original screening process, or from ordinary chance in a 1,000-row sample. `check_outcome_rate_against_reference()` above is a prompt to go find an independent reference population, not a standalone proof that a gate exists.

### 5. Fixing the Upstream Gate Is Usually Outside the Model Builder's Control

A model trained on `credit_customers.csv` cannot retroactively learn how a rejected applicant would have repaid. The realistic response is rarely "fix the data" - it is disclosing the limitation explicitly, seeking an external reference population where one exists, and treating any fairness gap measured on a selection-biased sample as a floor on the true gap, not the whole of it.

---

## Related Concepts

- [What Is Sampling Bias?](sampling-bias.md) - the representation-focused sibling of this concept: who is over- or under-represented in a dataset you already have, versus who was excluded from it entirely before collection began.
- [What Is Label Bias?](label-bias.md) - covers what happens when the labels themselves are distorted by biased human decisions. Selection bias is the earlier-stage version of the same problem: the row is missing before any label could even be attached.
- [What Is a Confounding Variable?](confounding-variable.md) - a hidden variable that causes two observed features to correlate. Selection bias is the special case where the hidden cause is the act of being included in the dataset itself.
- [What Is Distribution Shift?](distribution-shift.md) - covers a population that changes between training and deployment. Selection bias is a distortion that can exist on day one, before any drift has had a chance to occur.

---

## Further Reading

- [Berkson, J. (1946): Limitations of the Application of Fourfold Table Analysis to Hospital Data, Biometrics 2(3), 47-53](https://doi.org/10.2307/3002000) - the original paper describing how conditioning on hospital admission manufactures a spurious correlation between independent conditions.
- [Hand, D.J. & Henley, W.E. (1997): Statistical Classification Methods in Consumer Credit Scoring: A Review, Journal of the Royal Statistical Society Series A 160(3), 523-541](https://doi.org/10.1111/j.1467-985X.1997.00078.x) - the standard review covering reject inference and the reject-approval selection problem in credit scoring datasets.
- [Heckman, J.J. (1979): Sample Selection Bias as a Specification Error, Econometrica 47(1), 153-161](https://doi.org/10.2307/1912352) - the foundational econometric treatment of correcting for a non-randomly selected sample.

---

*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
