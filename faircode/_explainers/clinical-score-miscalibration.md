> *A "20% risk of readmission" score is not a fact about the patient - it is a claim about how often people who get this score actually come back. If that claim only holds for the majority of patients the model was trained on, the same number can be a fair warning for one group and a false reassurance for another.*

## The One-Sentence Definition

**Miscalibration across groups** means a clinical risk score that is well-calibrated overall - its "X% risk" predictions match real-world outcome rates on average - can still be wrong in opposite directions for different patient groups, so the same score corresponds to a different real-world risk depending on who receives it.

## Why It Matters

A calibration curve (or reliability diagram) checks whether a model's predicted probabilities track reality: among every patient scored "30% risk," roughly 30% should actually experience the outcome. Most clinical validation studies stop at this single, aggregate curve, and a model can pass it easily - because the curve is an average across the whole population, and a large majority group's good calibration can offset a smaller group's bad calibration without moving the overall line.

That average hides the question a fairness audit actually needs answered: is the score equally trustworthy *within* every group at the point clinicians act on it? A well-calibrated-on-average sepsis or readmission score can systematically overestimate risk for one group and underestimate it for another. Overestimation wastes clinical attention and can justify more aggressive, costly intervention than a patient needs. Underestimation is worse: a patient who is quietly higher-risk than their score suggests gets triaged as routine, and nobody notices until the outcome the score was supposed to predict has already happened.

This is different from the accuracy or demographic-parity gaps covered elsewhere in this repo. A model can have identical accuracy and an identical positive-prediction rate across two groups and still be miscalibrated for one of them, because calibration is about what a given score *means*, not about how often the model fires or how often it is right on average. See [Calibration](calibration.md) for the general definition and the COMPAS case where calibration and equalized odds are shown to be mutually exclusive when base rates differ - the same mathematics applies here, it just shows up in a risk score instead of a recidivism tool.

## Core Concept: Why Averaging Hides This

Reliability diagrams bin predictions (for example 0-10%, 10-20%, 20-30%, ...) and compare the mean predicted probability in each bin to the actual outcome rate of the patients in it. Two numbers usually summarize the curve:

- **Calibration intercept** - whether predictions are systematically too high or too low overall (an intercept away from 0 means the whole curve is shifted).
- **Calibration slope** - whether the model's predictions are too extreme or too flat (a slope below 1 means high scores are overconfident and low scores are under-confident; the reverse for a slope above 1).

A single aggregate reliability diagram computes both numbers once, over everyone. Split the same predictions by group and each group gets its own intercept and slope - and there is no mathematical reason for those to match, especially when a training population is dominated by one group. A model trained mostly on one group's outcome patterns will fit that group's slope and intercept well; a smaller subgroup's curve can bend away from the diagonal in either direction while the pooled curve still looks fine, because its errors are outnumbered in the average.

This is exactly what happened in Obermeyer et al.'s widely cited 2019 audit of a commercial population-health algorithm used on millions of US patients: the tool predicted *future healthcare cost* as a proxy for *future health need*, and it was well-calibrated for that proxy. But at any shared predicted-cost score, Black patients were on average sicker than White patients (more chronic conditions, worse lab values) - because less money had historically been spent on their care for reasons unrelated to how sick they were. The score was calibrated to cost. It was silently miscalibrated to health.

A revised-calculator study of cardiovascular risk scores found the same shape of problem in a different tool: Yadlowsky et al. (2018) re-evaluated the widely used Pooled Cohort Equations for predicting atherosclerotic cardiovascular disease and found the equations overestimated risk in some race/sex/age strata and underestimated it in others, at the exact score thresholds that trigger a statin prescription - meaning the calculator's single validated "well-calibrated" reputation was hiding threshold-level miscalibration for the specific groups clinicians use it to make prescribing decisions about.

## Concrete Example: Healthcare Readmission - Audit 06

Audit 06 predicts 30-day hospital readmission from the Diabetes 130-US Hospitals dataset. Running a reliability check (illustrative code below, not `unfair.py`/`fair.py` and not a published Audit 06 result - this doesn't touch any frozen file) against the same dataset by race shows why a single dataset can tell two different calibration stories depending on where you look:

```text
Caucasian (n=15,247)
  predicted 0-10%:  mean predicted 7.8%  | actual rate  8.1%   (n=6,008)
  predicted 10-20%: mean predicted 13.1% | actual rate 12.8%   (n=8,876)
  predicted 20%+:   mean predicted 23.6% | actual rate 26.2%   (n=363)

African American (n=3,815)
  predicted 0-10%:  mean predicted 7.7%  | actual rate  8.6%   (n=1,497)
  predicted 10-20%: mean predicted 13.3% | actual rate 12.3%   (n=2,190)
  predicted 20%+:   mean predicted 23.8% | actual rate 23.4%   (n=128)
```

In the two lower bins - which hold the vast majority of both groups - calibration is close for both races, within a point or two either way. That is a genuinely reassuring result, and it would be dishonest to manufacture a bigger gap than the data shows. But look at the group sizes in the highest-risk bin: 363 Caucasian patients versus only 128 African American patients, a quarter of the sample the calibration claim for that bin actually rests on. That bin is exactly where a clinician would want the score to be most trustworthy - it is the group flagged as highest risk - and it is exactly where this dataset gives the least statistical power to confirm it. A model can look calibrated for a small subgroup's high-risk bin purely because there were too few patients in it to detect a real gap, not because the gap isn't there. This is the practical failure mode: it is not that the score is visibly wrong, it is that there usually isn't enough data in the group and bin that matter most to know.

## Detection Code

Computes a per-group reliability table and a calibration slope/intercept summary, so a subgroup that is systematically over- or under-scored cannot hide inside a model that looks calibrated on average.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def reliability_table(y_true, y_prob, group, bins=(0, 0.1, 0.2, 1.01)):
    """
    Bins predicted probabilities and compares the mean predicted probability
    to the actual outcome rate within each bin, separately per group. A
    group whose actual rate consistently sits above (or below) its mean
    predicted probability is being systematically under- (or over-) scored.

    Parameters:
        y_true: ground-truth binary outcome (1 = event occurred)
        y_prob: model's predicted probability of the outcome
        group: protected attribute or group label, same length as y_true
        bins: bin edges for predicted probability (default: coarse 3-bin)

    Returns a DataFrame indexed by (group, bucket) with n, mean predicted
    probability, and actual outcome rate. Always inspect n before trusting
    a bucket - see Limitation 3 below.
    """
    df = pd.DataFrame({"y": y_true, "p": y_prob, "group": group})
    df["bucket"] = pd.cut(df["p"], bins=bins, right=False)

    return (
        df.groupby(["group", "bucket"], observed=True)
        .agg(n=("y", "size"), mean_predicted=("p", "mean"), actual_rate=("y", "mean"))
    )


def calibration_slope_intercept(y_true, y_prob, group, min_group_size=50):
    """
    Fits a logistic recalibration model (Cox 1958 / Van Calster et al. 2016
    "calibration-in-the-large" and "calibration slope") per group: the true
    outcome regressed on the logit of the predicted probability. A perfectly
    calibrated group has slope = 1.0 and intercept = 0.0.

    slope < 1: the group's predictions are too extreme (overconfident) -
        its highest scores overstate risk, its lowest scores understate it.
    slope > 1: the group's predictions are too flat (underconfident).
    intercept != 0: the group's predictions are systematically shifted up
        (intercept < 0) or down (intercept > 0) regardless of slope.

    Groups smaller than min_group_size are skipped - a slope fit on too few
    events is not trustworthy (see Limitation 3).
    """
    df = pd.DataFrame({"y": y_true, "p": np.clip(y_prob, 1e-6, 1 - 1e-6), "group": group})
    df["logit_p"] = np.log(df["p"] / (1 - df["p"]))

    rows = []
    for g, sub in df.groupby("group"):
        if len(sub) < min_group_size or sub["y"].nunique() < 2:
            continue
        model = LogisticRegression()
        model.fit(sub[["logit_p"]], sub["y"])
        rows.append({
            "group": g,
            "n": len(sub),
            "slope": model.coef_[0][0],
            "intercept": model.intercept_[0],
        })

    return pd.DataFrame(rows).set_index("group")


# Usage example
# table = reliability_table(readmission_df["readmitted"], readmission_df["predicted_prob"], readmission_df["race"])
# summary = calibration_slope_intercept(readmission_df["readmitted"], readmission_df["predicted_prob"], readmission_df["race"])
```

## Limitations

### 1. Calibration and equalized error rates cannot both hold when base rates differ

Chouldechova (2017) proved that if two groups have different base rates for the outcome, a score cannot simultaneously be calibrated for both groups *and* have equal false-positive and false-negative rates between them. Fixing calibration can widen an error-rate gap, and vice versa - see [Fairness Metric Conflicts](fairness-metric-conflicts.md). There is no calibration fix that avoids this trade-off; there is only a documented choice of which property to prioritize.

### 2. Bin choice changes what a reliability diagram shows

Coarse bins (as used above) can average away a miscalibration that only appears in a narrow score range; very fine bins fragment small groups until every bucket is too small to trust (see Limitation 3). Report the bin width used and check that the finding is stable across a couple of reasonable choices before treating it as real.

### 3. Small subgroups produce unreliable calibration slopes

A calibration slope or a reliability bucket fit on a few dozen events has a wide confidence interval - the Healthcare Readmission example above is a direct illustration of this, not a hypothetical caveat. Always report the bucket or group size next to the number, and treat a clean-looking calibration curve on a small group as inconclusive, not as proof of fairness.

### 4. Calibration to a proxy target is not calibration to the target that matters

A score can be perfectly calibrated to whatever label it was trained on and still be the wrong label to calibrate to, exactly as in the Obermeyer case where the model was calibrated to healthcare cost rather than health need. Checking calibration says nothing about whether the outcome variable itself was a fair choice - that is a separate question, addressed in [Label Bias](label-bias.md).

## Related Concepts

* [Calibration](calibration.md) - the general definition, the COMPAS calibration-vs-equalized-odds trade-off, and the Chouldechova (2017) impossibility result this explainer builds on.
* [Confusion Matrix](confusion-matrix.md) - the true/false positive and negative counts that a reliability diagram's bins are built from at any given threshold.
* [Why Accuracy Is Not Enough in Healthcare AI](accuracy-not-enough-healthcare-ai.md) - the companion failure mode: a model can hit identical per-group accuracy while missing far more true cases in one group, the same way it can hit identical per-group calibration while over- or under-stating risk for one group.
* [Label Bias](label-bias.md) - what happens when the target a score is calibrated to (cost, an arrest, a diagnosis) is itself a biased proxy for the outcome that actually matters.

## Related Projects in This Repo

* [`Healthcare Readmission/`](../Healthcare%20Readmission/) - the dataset behind the reliability table above; its highest-risk bucket is exactly the small-subgroup calibration trap this explainer describes.
* [`Insurance Denial/`](../Insurance%20Denial/) - a second health-adjacent audit where a risk score's per-group trustworthiness, not just its accuracy, decides who is over- or under-flagged for a costly claim review.

## Further Reading

* [Obermeyer, Z., Powers, B., Vogeli, C., Mullainathan, S. (2019): Dissecting racial bias in an algorithm used to manage the health of populations](https://www.science.org/doi/10.1126/science.aax2342) - a risk score well-calibrated to its training target (cost) that was silently miscalibrated to the target that mattered (health need) across race.
* [Yadlowsky, S., Hayward, R.A., Sussman, J.B., McClelland, R.L., Min, Y.I., Basu, S. (2018): Clinical Implications of Revised Pooled Cohort Equations for Estimating Atherosclerotic Cardiovascular Disease Risk](https://www.acpjournals.org/doi/10.7326/M17-3011) - a widely used cardiovascular risk calculator shown to over- and under-estimate risk in different demographic strata at the exact thresholds used to decide statin prescriptions.
* [Van Calster, B., Nieboer, D., Vergouwe, Y., De Cock, B., Pencina, M.J., Steyerberg, E.W. (2016): A calibration hierarchy for risk models was defined: from utopia to empirical data](https://www.jclinepi.com/article/S0895-4356(15)00512-7/fulltext) - the formal definitions behind calibration-in-the-large, calibration slope, and the finer hierarchy of calibration this explainer's detection code implements a slice of.

*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
