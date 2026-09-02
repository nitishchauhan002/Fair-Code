# Explainer: What Is a Protected Attribute?

> *Why removing race from a dataset does not remove the bias - and why the "fairness through unawareness" approach fails.*

## The One-Sentence Definition

**A protected attribute** is a demographic or personal characteristic - such as race, gender, age, or religion - that the law, policy, or a specific fairness standard shields from being used as the basis for adverse decisions like denying a loan, a job, or healthcare.

## Why It Matters

When developers first try to build a "fair" model, their instinct is usually to delete the protected attribute from the dataset before training. This is called **fairness through unawareness**. The logic seems sound: if the model cannot see race, it cannot discriminate by race.

This fails completely in practice. 

Machine learning models do not care what a column is named; they care what it predicts. If you remove the `sex` column but keep `marital.status`, `relationship`, and `hours.per.week`, the model will learn to reconstruct the applicant's sex from those correlated features (known as proxy variables). The model is still penalizing the protected group, but now it is doing so indirectly.

More importantly, **if you do not collect the protected attribute, you cannot audit the model**. You cannot measure a fairness gap if you do not know who is in the dataset. Dropping the protected attribute early in the pipeline guarantees that any bias the model learns will be invisible to you.

## Legal and Domain Variations

What counts as a "protected attribute" changes depending on where you are and what you are building:

* **In the US:** The Civil Rights Act of 1964 protects race, color, religion, sex, and national origin in employment. The Equal Credit Opportunity Act (ECOA) adds age and marital status for lending.
* **In the EU:** GDPR Article 9 defines "special categories of personal data," adding genetic data, biometric data, and trade union membership to the standard list of protected characteristics.
* **By domain:** Age is highly protected in employment and lending, but it is a standard, medically necessary input in clinical risk models.

## Concrete Example: Benefits Denial - Audit 05

Audit 05 in this repo evaluates an automated means-test model (based on the UCI Adult dataset) that predicts whether an applicant earns above a $50K threshold to determine welfare eligibility. The protected attributes are sex, race, national origin, and age.

If we look at the baseline model trained with all features, the model flags male applicants for ineligibility at a rate **18.00 percentage points higher** than female applicants. 

If we apply "fairness through unawareness" and drop `sex` (along with race, age, and national origin), the model still flags male applicants at a massively elevated rate. Why? Because the dataset still contains `relationship` (Husband/Wife) and `marital.status` - features that perfectly encode sex in historical data. The model just shifted its penalty from the explicit `sex` column to the proxy columns.

To actually fix the bias, you have to measure it first, identify the proxies, and remove the protected attribute *and* the proxies together. When Audit 05 drops the proxies as well, the sex gap drops by 53%.

## Detection Code

This snippet demonstrates why "fairness through unawareness" fails. It drops the protected attribute, trains a model on the remaining "neutral" features, and then asks that model to predict the dropped attribute. If the model succeeds, the protected signal is still there.

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def test_fairness_through_unawareness(df, protected_col):
    """
    Tests if the remaining features can still predict the protected attribute.
    If AUC is high, 'fairness through unawareness' has failed - the proxies
    are keeping the protected signal alive.
    """
    # 1. Drop the target variable (assuming it's 'target' for this example)
    # In practice, you drop the actual outcome variable here.
    if 'target' in df.columns:
        X = df.drop(columns=['target', protected_col])
    else:
        X = df.drop(columns=[protected_col])
        
    y = df[protected_col]
    
    # 2. Convert categorical features
    X = pd.get_dummies(X)
    
    # 3. Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # 4. Predict the protected attribute
    preds = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    
    print(f"--- Unawareness Test ---")
    print(f"Ability to predict '{protected_col}' from remaining features:")
    print(f"AUC: {auc:.3f} (0.5 = random guessing, 1.0 = perfect reconstruction)")
    
    return auc
```

## Limitations

### 1. The Data Collection Paradox

To prove a model is fair, you must collect protected attributes. But privacy laws (like GDPR) and corporate risk teams often forbid collecting this exact data. This paradox forces teams to either estimate demographics (which introduces new biases) or ship models without auditing them.

### 2. Intersectional Harms

Treating protected attributes as independent columns (e.g., checking race, then checking gender) hides intersectional bias. A model might look fair for Black applicants overall and fair for women overall, while severely penalizing Black women.

### 3. "Neutral" is not always Fair

Just because a feature is not a legally protected attribute does not mean it is fair to use. Zip code is not a protected attribute, but using it to deny loans in a historically redlined city produces the exact same disparate impact as explicitly using race.

## Related Concepts

* [What is a Proxy Variable?](proxy-variables.md) - the correlated features that reconstruct protected attributes after they are removed.
* [Disparate Treatment](disparate-treatment.md) - intentional discrimination via direct use of a protected attribute.
* [Proxy Entanglement](proxy-entanglement.md) - why removing proxies one at a time fails when multiple features encode the same protected signal.

## Further Reading

* Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012): [Fairness Through Awareness](https://arxiv.org/abs/1104.3913) - the foundational paper explaining why the unawareness approach is insufficient.
* [Title VII of the Civil Rights Act of 1964](https://www.eeoc.gov/statutes/title-vii-civil-rights-act-1964) - the US legal foundation for protected classes in employment.
* [GDPR Article 9](https://gdpr-info.eu/art-9-gdpr/) - the EU standard for processing special categories of personal data.

*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
