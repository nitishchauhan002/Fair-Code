> *A commercial risk-prediction algorithm used on over 200 million people annually assigned White and Black patients the same risk score when they generated the same healthcare costs. But because less money is historically spent on Black patients at the same level of illness, Black patients at that shared score were dramatically sicker. When algorithms mistake medical spending for medical need, systemic inequality becomes automated discrimination.*

## The One-Sentence Definition

**"The Obermeyer Case"** refers to the canonical real-world proxy-label failure identified by Obermeyer et al. (2019), where a commercial healthcare risk algorithm predicted healthcare *cost* as a stand-in for health *need* - systematically under-referring sicker Black patients to high-risk care management programs because historical spending on Black patients was lower at every level of illness.

## Why It Matters

Supervised machine learning models do not optimize for what developers *intend* them to measure; they optimize strictly for the target label (`Y`) specified in the training dataset. When developers select a target proxy that is corrupted by systemic disparities - such as medical expenditures, arrest records, or past manager evaluations - the model learns to reproduce those disparities even if all explicit demographic attributes are removed from the feature set.

In healthcare population management, high-risk care management programs provide extra resources (specialized primary care, dedicated nurse check-ins, and monitoring) to complex patients to prevent emergency hospitalizations. Because financial billing data is clean, standardized, and readily available across electronic health records (EHRs), developers frequently train algorithms to predict future total medical spending as a proxy for future health need.

However, medical spending is not medical need. Spending reflects health need **filtered through access to care**, insurance coverage, socioeconomic barriers, geographic proximity to health systems, and physician referral patterns. When an algorithm predicts spending, it learns that a patient with fewer recorded medical bills is "lower risk," mistaking under-utilization and barriers to care for good health.

## The Core Concept: How Spending Corrupts Health Risk Scores

To understand why proxy label choice corrupts fairness, compare the true clinical target with the proxy target:

* **True Target (`Y*`):** Actual health need (e.g., severity of chronic diseases, organ dysfunction, uncontrolled hypertension, risk of emergency complications).
* **Proxy Target (`Y_cost`):** Total annual healthcare expenditures in dollars.

In a fair system without structural barriers, healthcare spending would be directly proportional to health need (`Y_cost` proportional to `Y*`) across all demographic groups. In reality, historical healthcare expenditures exhibit severe racial disparity at equal levels of illness:

```text
Expected_Cost(Race = Black, Illness = k) < Expected_Cost(Race = White, Illness = k)
```

Because less money is spent caring for Black patients at any given illness level `k`, an algorithm trained to predict `Y_cost` learns a biased spending score. When the algorithm ranks patients by predicted risk to enroll the top 3% (or 5%) into specialized care programs:

| Metric at Shared Enrollment Score Threshold | White Patients | Black Patients | Structural Disparity |
|---|---|---|---|
| **Predicted Healthcare Cost** | Equal | Equal | Algorithm appears calibrated on cost |
| **Actual Chronic Conditions Count** | Baseline | **~28% Higher** | Black patients are substantially sicker |
| **Biomedical Biomarkers (e.g., HbA1c, BP)** | Baseline | **Significantly Worse** | Black patients have worse physiological health |
| **Care Program Auto-Enrollment Rate** | Baseline | **Substantially Reduced** | Sicker Black patients are systematically bypassed |

The algorithm is not broken in a mathematical sense - it predicts future spending with high accuracy for both groups. The failure lies in the **semantic gap** between the proxy label (`Y_cost`) and the human goal (`Y*`).

## The Real-World Impact: The 2019 Obermeyer Findings

In 2019, Ziad Obermeyer, Brian Powers, Christine Vogeli, and Sendhil Mullainathan published their landmark study in *Science*, auditing a commercial risk-prediction algorithm applied to over 200 million patients annually across major US health systems.

Key quantitative findings from the study include:

1. **Illness Disparity at the Threshold:** At the 97th percentile risk threshold - where patients were automatically enrolled in specialized care management - Black patients generated the same predicted cost as White patients, but had **26.3% to 28% more chronic conditions** (such as hypertension, diabetes complications, and heart failure).
2. **The Re-allocation Effect:** If the algorithm had been retrained to predict actual health status (measured by un-met health needs and active chronic conditions) rather than spending, the proportion of Black patients automatically enrolled in the high-risk care management program would have **more than doubled**, increasing from **17.7% to 46.5%**.
3. **Disparity Across All Biomarkers:** The disparity persisted across independent physiological measurements not used in the algorithm's target, including blood pressure, cholesterol, renal function indicators, and hemoglobin A1c.
4. **The "Fairness Through Unawareness" Trap:** The algorithm did not use race as an input feature. Removing race did nothing to prevent the bias, because racial disparities in healthcare access were baked directly into the target variable itself.

## Concrete Example: Healthcare Readmission Audit

The Obermeyer case study directly mirrors the structural challenges in Fair Code's [`Healthcare Readmission/`](../Healthcare%20Readmission/) audit (based on the Diabetes 130-US Hospitals dataset with 101,766 records).

In clinical risk modeling, target labels such as 30-day hospital readmission (`readmitted = 1`) or total inpatient visit counts can suffer from proxy distortion:
* A patient who lives near a tertiary care center and has comprehensive insurance may be readmitted quickly when symptoms recur.
* A patient with severe care access barriers, transportation deficits, or lack of insurance may delay returning to the hospital until emergency status, or may present at a different non-reporting facility.

In the frozen benchmark results for Audit 06 (`paper/results-frozen/summary.csv`), baseline models for `healthcare_readmission` evaluate fairness across race and age:

```csv
audit,strategy,protected_attribute,metric,mean_value
healthcare_readmission,baseline,race,demographic_parity_diff,-0.0000858
healthcare_readmission,baseline,race,equalized_odds_diff,0.0017434
healthcare_readmission,baseline,race,predictive_parity_diff,0.0336660
```

While on-paper demographic parity and equalized odds gaps for race appear small in aggregate baseline benchmarks, aggregate metrics cannot detect whether the target variable itself under-counts true health need in under-resourced subgroups. If the target label only records encounters that resulted in a hospital admission, unrecorded out-of-hospital deterioration creates silent proxy label bias.

## Detection Code

The following Python function audits a dataset for Obermeyer-style proxy-label disparity. It evaluates whether patients from different demographic groups at the same predicted risk threshold possess unequal levels of true health need, and calculates the population re-allocation percentage if the target is switched from cost to health status.

```python
import numpy as np
import pandas as pd


def audit_proxy_label_disparity(
    df: pd.DataFrame,
    proxy_col: str,
    true_health_col: str,
    group_col: str,
    percentile_threshold: float = 0.97,
) -> pd.DataFrame:
    """
    Audits a clinical dataset for proxy label disparity by checking whether
    patients at the same predicted risk or cost threshold have equal true
    health needs across demographic groups.

    Parameters:
        df: DataFrame containing predictions/proxy scores, ground-truth health status,
            and group membership.
        proxy_col: Column name of the proxy target or model score (e.g. predicted spending).
        true_health_col: Column name of true health status (e.g. chronic condition count).
        group_col: Column name of the protected attribute (e.g. race or age).
        percentile_threshold: Top percentile used for care program enrollment (default 0.97).

    Returns:
        DataFrame summarizing mean proxy score, mean true illness at threshold,
        and enrollment percentage shifts per group.
    """
    df = df.copy()
    cutoff_proxy = df[proxy_col].quantile(percentile_threshold)
    enrolled_proxy = df[df[proxy_col] >= cutoff_proxy]

    cutoff_true = df[true_health_col].quantile(percentile_threshold)
    enrolled_true = df[df[true_health_col] >= cutoff_true]

    total_n = len(df)
    results = []

    for group_name, group_df in df.groupby(group_col):
        n_group = len(group_df)
        proxy_enrolled_sub = enrolled_proxy[enrolled_proxy[group_col] == group_name]
        true_enrolled_sub = enrolled_true[enrolled_true[group_col] == group_name]

        mean_illness_at_proxy_cutoff = (
            proxy_enrolled_sub[true_health_col].mean()
            if len(proxy_enrolled_sub) > 0
            else np.nan
        )

        proxy_enrollment_share = (len(proxy_enrolled_sub) / len(enrolled_proxy)) * 100
        true_enrollment_share = (len(true_enrolled_sub) / len(enrolled_true)) * 100

        results.append(
            {
                "group": group_name,
                "n_patients": n_group,
                "mean_proxy_score": group_df[proxy_col].mean(),
                "mean_illness_at_threshold": mean_illness_at_proxy_cutoff,
                "proxy_enrollment_share_pct": proxy_enrollment_share,
                "true_health_enrollment_share_pct": true_enrollment_share,
                "reallocation_shift_pct": true_enrollment_share - proxy_enrollment_share,
            }
        )

    summary_df = pd.DataFrame(results).set_index("group")
    return summary_df


# Usage Example:
# audit_results = audit_proxy_label_disparity(
#     df=patient_data,
#     proxy_col="predicted_annual_cost",
#     true_health_col="active_chronic_conditions_count",
#     group_col="race",
#     percentile_threshold=0.97
# )
# print(audit_results)
```

## Limitations

### 1. "True Health Need" Is Difficult to Measure Without Spending
Finding a completely un-biased ground truth `Y*` in medical records is non-trivial. While chronic condition counts and lab biomarkers are far superior to spending, lab testing frequency itself can be subject to access disparities (patients with fewer medical visits have fewer lab records).

### 2. Financial Constraints vs. Clinical Governance
Healthcare organizations often operate under strict fixed budgets. Finance teams prefer cost-based targets because they directly map to short-term budgetary exposure. Overcoming proxy label bias requires aligning clinical leadership and financial decision-makers on the long-term ROI of preventive health equity.

### 3. Care Program Outreach Barriers
Simply fixing the algorithm's target label to auto-enroll sicker Black patients does not guarantee improved health outcomes if structural barriers (lack of transportation, hourly work inflexibility, or clinical mistrust) prevent enrolled patients from utilizing the care management program. Algorithmic fairness must be paired with operational equity.

## Related Concepts

* [Label Bias](label-bias.md) - how historical discrimination in ground-truth target labels corrupts supervised learning models before training starts.
* [Proxy Variables](proxy-variables.md) - why removing race from input features does not remove demographic bias when input variables correlate with protected attributes.
* [Why Accuracy Is Not Enough in Healthcare AI](accuracy-not-enough-healthcare-ai.md) - how aggregate performance metrics mask severe subgroup failures in clinical decision support.
* [False Positives vs. False Negatives in Medical Risk Models](false-positives-vs-false-negatives.md) - understanding the asymmetric clinical costs of missing high-risk patients versus false alarms.
* [Miscalibration in Clinical Risk Scores Across Groups](clinical-score-miscalibration.md) - why a risk score calibrated to cost produces miscalibrated illness predictions across demographic groups.
* [Missing Data as Bias in Electronic Health Records](missing-data-bias-ehr.md) - how unobserved lab values and clinical encounters reflect care access rather than low patient risk.

## Related Projects in This Repo

* [`Healthcare Readmission/`](../Healthcare%20Readmission/) - Fair Code's primary clinical audit analyzing readmission risk predictions, feature importance, and fairness metrics across age, gender, and race.
* [`Insurance Denial/`](../Insurance%20Denial/) - examining how financial decisions and claims approval algorithms interact with patient risk categories.
* [`Benefits Denial/`](../Benefits%20Denial/) - auditing public assistance algorithms where automated eligibility criteria mirror access disparities.

## Further Reading

* [Obermeyer, Z., Powers, B., Vogeli, C., Mullainathan, S. (2019): Dissecting racial bias in an algorithm used to manage the health of populations](https://www.science.org/doi/10.1126/science.aax2342) - the seminal *Science* paper establishing the canonical case study of proxy label bias in commercial health algorithms.
* [Rambachan, A., Kleinberg, J., Ludwig, J., Mullainathan, S. (2020): An Economic Approach to Regulating Algorithms](https://www.nber.org/papers/w27111) - NBER Working Paper detailing economic and statistical frameworks for algorithmic bias and proxy targets.
* [Benjamin, R. (2019): Assessing risk, automating racism](https://www.science.org/doi/10.1126/science.aaz3873) - *Science* commentary discussing the societal implications of automating historical resource allocation patterns in public health.

*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
