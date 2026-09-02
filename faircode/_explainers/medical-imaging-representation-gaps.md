> *A dermatology model trained mostly on light skin is not "slightly less accurate" on dark skin - it is being tested on a population it has barely seen. And a chest X-ray model that learns to spot the hospital instead of the disease will look excellent on the hospital it trained in and quietly fail everywhere else.*

## The One-Sentence Definition

**Medical imaging representation gaps** are what happens when a diagnostic imaging model - dermatology, radiology, retinal, pathology - is trained on a dataset where some skin tones, body types, imaging equipment, or acquisition sites are thin or absent, so the model either performs worse on the underrepresented group or, worse, learns to key off a confounder like scanner type or hospital site that happens to correlate with the diagnosis in the training data but means nothing clinically.

## Why It Matters

Imaging models are usually judged on one aggregate accuracy or AUC number computed across the whole test set. That number can look excellent while the model has effectively never learned to recognize the disease's presentation on the patients it saw the fewest of. Dermatology datasets are a well-documented example: the images used to train and validate many skin-lesion classifiers are overwhelmingly of lighter Fitzpatrick skin types, because that is what was available in the dermatology literature and public datasets. A model trained on that distribution has no reason to have learned what melanoma or a benign lesion actually looks like on darker skin, and it will not announce that gap on its own; it will simply be less accurate for those patients while the headline metric, dominated by the majority group, stays high.

The second failure mode is more insidious than under-representation alone: **shortcut learning**. A convolutional network optimizing for training accuracy will happily exploit *any* pattern that predicts the label, including ones with nothing to do with the disease - a portable scanner's image signature, a hospital's specific equipment, a marker or ruler placed next to a lesion during specialist photography. If a particular device or site was disproportionately used to image patients with (or without) a condition, the model can learn to detect the device instead of the disease, and it will look flawless on data from that device and fail sharply on any other. Because a chest X-ray or a dermatology photo carries no obvious label saying "this pixel pattern is a scanner artifact, not anatomy," this kind of confounding is invisible to a normal accuracy check and only shows up when the model is tested outside the site it trained on.

Both failure modes share the same root cause as every other explainer in this repo: a model finds whatever correlates with the label in its training data, whether or not that correlation is the one a clinician would consider real, and an aggregate metric computed on data drawn from the same skewed distribution cannot tell the difference.

## Core Concept: Representation Gap vs. Shortcut Confounding

These are two distinct mechanisms that often compound each other:

- **Representation gap** - a group is present in the training data, but in small enough numbers that the model has too few examples to learn its presentation of the disease well. This shows up as a straightforward per-group accuracy or AUC gap that widens as the underrepresented group's sample size shrinks.
- **Shortcut confounding** - a variable unrelated to the pathology (scanner model, hospital site, image compression artifact, a physical marker placed during acquisition) happens to correlate with the label in the training data, and the model learns that shortcut instead of the intended signal. This can produce a model that looks *extremely* accurate in training and internal validation - because the shortcut is real and present in that data - while failing on any external dataset where the shortcut and the label are no longer correlated.

A representation gap degrades gracefully: performance drops for the thin group but the model is still trying to detect the disease. Shortcut confounding does not degrade gracefully: the model may never have been detecting the disease for anyone, and internal test-set accuracy is not evidence otherwise, because the shortcut is present in the internal test set too.

## Concrete Example: Two Documented Imaging Failures

**Chest X-rays learning the hospital, not the pneumonia.** Zech et al. (2018) trained a CNN to detect pneumonia from chest radiographs across multiple hospital systems and found it partly learned to recognize *which hospital* took the image - because one hospital's radiographs were disproportionately portable films taken of sicker, bedridden inpatients, portable-scanner image characteristics became a shortcut for "sicker patient" and therefore for the label. The model's performance on data from a hospital it trained on was strong; its performance dropped sharply when tested on a hospital it had not seen, because the site-specific shortcut it had learned did not transfer. No single-hospital accuracy number would have revealed this - it only appeared under external, cross-site validation.

**Gender imbalance producing a biased chest X-ray classifier.** Larrazabal et al. (2020) trained identical chest X-ray diagnosis models on datasets with deliberately varied male/female sample ratios and showed that a model trained predominantly on one sex's images performed measurably worse on the other sex - even though sex was not a diagnostic label the model was asked to predict, and nothing about the pathologies studied should have differed by sex in principle. The representation ratio in the training set alone was enough to produce the gap, with no confounding device or site involved.

Both papers make the same point from different mechanisms: an aggregate accuracy or AUC computed on a test set drawn from the same skewed source as the training data cannot detect either failure. Detecting them requires a per-group breakdown and, for the shortcut case, testing on data collected somewhere the shortcut wasn't present.

## Detection Code

Two checks that a single aggregate AUC hides: a per-group performance gap, and a proxy check for whether a non-clinical variable (site, device) is entangled with the label closely enough to be a plausible shortcut.

```python
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import chi2_contingency


def group_auc_gap(df, y_true_col, y_score_col, group_col, min_group_size=30):
    """
    Computes AUC per group from a model's predicted scores, so a
    representation gap hiding inside a strong aggregate AUC becomes visible.
    Groups smaller than min_group_size are flagged rather than trusted -
    AUC on a small sample is a noisy estimate.

    Parameters:
        df: DataFrame with one row per image/patient
        y_true_col: ground-truth binary diagnosis label
        y_score_col: model's predicted probability or score
        group_col: skin tone bucket, sex, age band, acquisition site, etc.

    Returns a DataFrame indexed by group with auc, n, and a
    small_sample flag; adds a "gap" row (max AUC - min AUC among
    groups that clear min_group_size).
    """
    rows = []
    for group, sub in df.groupby(group_col):
        small = len(sub) < min_group_size or sub[y_true_col].nunique() < 2
        auc = float("nan") if small else roc_auc_score(sub[y_true_col], sub[y_score_col])
        rows.append({"group": group, "auc": auc, "n": len(sub), "small_sample": small})

    result = pd.DataFrame(rows).set_index("group")
    trustworthy = result[~result["small_sample"]]
    if len(trustworthy) >= 2:
        result.loc["gap"] = [trustworthy["auc"].max() - trustworthy["auc"].min(), float("nan"), False]
    return result


def shortcut_confounder_check(df, label_col, confounder_col):
    """
    Chi-squared test for whether a non-clinical variable (imaging device,
    site, hospital) is statistically entangled with the diagnostic label in
    the training data. A significant result does not prove the model is
    using it as a shortcut, but it means the shortcut is *available* for
    the model to learn - exactly the setup Zech et al. (2018) found. Follows
    the same proxy-detection pattern used throughout this repo
    (see CONTRIBUTING.md's proxy-variable section).

    Returns the chi-squared statistic and p-value; p < 0.05 means the
    confounder and the label are unlikely to be independent by chance.
    """
    table = pd.crosstab(df[confounder_col], df[label_col])
    chi2, p_value, _, _ = chi2_contingency(table)
    return {"chi2": chi2, "p_value": p_value, "crosstab": table}


# Usage example
# gaps = group_auc_gap(predictions_df, "has_condition", "predicted_score", "skin_tone_bucket")
# confound = shortcut_confounder_check(training_df, "has_condition", "acquisition_site")
```

## Limitations

### 1. Ground-truth group labels are often estimates, not facts

Most public imaging datasets do not record skin tone directly; researchers estimate it from the image itself (e.g. via Fitzpatrick scale or individual typology angle), which is itself an imperfect, sometimes contested measurement. A per-group AUC gap is only as trustworthy as the group label it's computed on.

### 2. A representation fix does not fix a shortcut, and vice versa

Oversampling or reweighting the underrepresented group addresses a pure representation gap, but if a shortcut confounder is present, adding more images from the same skewed sites can reinforce the shortcut rather than remove it. Diagnosing which mechanism is at play (per-group gap alone vs. gap plus a confounder check) should come before choosing a fix.

### 3. Internal validation cannot rule out shortcut confounding

A model can pass every per-group and aggregate check computed on data from the same sites it trained on and still be relying on a site-level shortcut, because the shortcut is present in that entire dataset, training and test split alike. Only external validation on genuinely different acquisition sites or devices can surface this - which is also why the Zech et al. finding only appeared under cross-hospital testing.

### 4. Small subgroup sample sizes make per-group AUC noisy

The same caution that applies to per-group accuracy or recall in tabular audits applies here: an AUC computed on a few dozen images has a wide confidence interval. Report group sizes next to every per-group AUC, and treat a gap on a small subgroup as a reason to gather more data before concluding the model is fine (or biased).

## Related Concepts

* [Sampling Bias](sampling-bias.md) - the general version of the representation-gap problem this explainer specializes to imaging datasets.
* [Distribution Shift](distribution-shift.md) - why a model validated on one hospital's imaging distribution can fail once deployed against a different one, the same mechanism behind the Zech et al. cross-site failure.
* [Why Accuracy Is Not Enough in Healthcare AI](accuracy-not-enough-healthcare-ai.md) - the tabular-data version of the same lesson: an aggregate metric averages over the group that matters most.
* [How Does AI Detect Patterns?](how-ai-detects-patterns.md) - why a model has no built-in way to distinguish a causal, clinical pattern from a shortcut correlation like scanner type or hospital site.

## Related Projects in This Repo

* [`Healthcare Readmission/`](../Healthcare%20Readmission/) - this repo's own healthcare audit demonstrates the tabular-data analogue: proxy features (`payer_code`, `discharge_disposition_id`) standing in for structural access gaps the same way an imaging shortcut stands in for a device or site.

## Further Reading

* [Zech, J.R., Badgeley, M.A., Liu, M., Costa, A.B., Titano, J.J., Oermann, E.K. (2018): Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs across multiple institutions](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002683) - the study behind the hospital-as-shortcut example, showing strong internal accuracy that did not transfer across sites.
* [Larrazabal, A.J., Nieto, N., Peterson, V., Milone, D.H., Ferrante, E. (2020): Gender imbalance in medical imaging datasets produces biased classifiers for computer-aided diagnosis](https://www.pnas.org/doi/10.1073/pnas.1919012117) - a controlled demonstration that training-set sex ratio alone produces a per-group performance gap in chest X-ray diagnosis.
* [Adamson, A.S., Smith, A. (2018): Machine Learning and Health Care Disparities in Dermatology](https://jamanetwork.com/journals/jamadermatology/fullarticle/2688587) - a direct analysis of skin-tone representation gaps in the public dermatology image datasets used to train and validate skin-lesion classifiers.

*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
