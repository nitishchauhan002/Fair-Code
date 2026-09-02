> *Nobody told the algorithm to sort people by sex. It found the split on its own, because the split was already sitting in the data.*

## The One-Sentence Definition

**Unsupervised learning** finds structure, clusters, patterns, or compressed representations, in unlabelled data, letting the algorithm organise it without being told what to look for.

## Why It Matters

Every audit elsewhere in this repo starts from a label: hired or not, good credit or bad, high-risk or low-risk. Unsupervised learning removes that label entirely. There is no ground truth to check the output against, no `y` to hold out and score against. That absence is exactly what makes it risky in a fairness context: a clustering algorithm cannot discriminate against a group "on purpose," because it has no concept of groups to begin with, yet it can still sort people along demographic lines as a side effect of sorting them by whatever features it was given. The insurance industry's use of ZIP-code clustering for pricing is the standard real-world case: nobody fed the model race, but ZIP code carries enough historical residential-segregation signal that the resulting clusters tracked race anyway, a pattern examined in ProPublica's investigative reporting and by state insurance regulators.

The non-obvious part is that this happens even when the fields most obviously tied to a protected attribute are excluded. Clustering only needs *some* subset of features to correlate with a protected group, not all of them, for the resulting structure to double as a demographic split.

## How It Works

### Structure Without a Target

A supervised model is handed `(features, label)` pairs and learns a function from one to the other. An unsupervised model is handed features only. K-means, the clustering algorithm used below, works by placing `k` centroids in feature space and iteratively assigning each row to its nearest centroid, then moving each centroid to the mean of the rows assigned to it, repeating until assignments stop changing. Nothing in that loop references sex, race, or any other protected attribute. It only minimises distance in whatever feature space it was given.

### Walking Through the Clustering Code

The Benefits Denial audit's `unfair.py` already identified `relationship` and `marital.status` as near-perfect proxies for sex (`Husband` is 0% female, `Wife` is 0% male; married applicants skew heavily male). Running k-means on a feature set that includes those columns, but excludes `sex`, `race`, and `native.country` outright, shows the same proxy effect appearing without supervision:

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

feature_cols = ['age', 'workclass', 'education.num', 'marital.status',
                'occupation', 'relationship', 'capital.gain',
                'capital.loss', 'hours.per.week']
X = pd.get_dummies(df[feature_cols])
Xs = StandardScaler().fit_transform(X)

km = KMeans(n_clusters=2, random_state=42, n_init=10)
df['cluster'] = km.fit_predict(Xs)
```

`sex`, `race`, and `native.country` never appear in `feature_cols`. The clustering step has no way to "see" them. Whatever split it produces has to come entirely from the other nine columns.

## Concrete Example: Benefits Denial - Audit 04

Running the code above on the UCI Adult Census Income dataset (32,561 records, the same file the Benefits Denial audit trains on) produces two clusters of size 17,680 and 14,881. Neither `sex`, `race`, nor `native.country` was part of the clustering input, so the composition of each cluster along those attributes was checked after the fact, not fed in beforehand:

| | Cluster 0 (n=17,680) | Cluster 1 (n=14,881) |
|---|:---:|:---:|
| Female | 51.9% | 10.7% |
| Male | 48.1% | 89.3% |
| Black | 13.0% | 5.6% |
| White | 81.9% | 89.6% |
| Foreign-born | 10.2% | 10.7% |

Cluster 1 is 89.3% male against cluster 0's near-even split, and Black applicants appear at more than double the rate in cluster 0 versus cluster 1. Nobody asked the algorithm to separate applicants by sex or race. It separated them by `relationship`, `marital.status`, `occupation`, and `hours.per.week`, and a sex and race split came out the other side as a side effect, exactly the mechanism [What Is a Proxy Variable?](proxy-variables.md) describes for supervised models, now showing up with no label in sight.

National origin is the counterexample worth keeping: foreign-born applicants land at essentially the same rate in both clusters (10.2% vs 10.7%). Not every protected attribute reappears in every clustering. Whether one does depends on how strongly it correlates with the specific features chosen and the number of clusters requested, which is precisely the fragility this explainer's limitations section covers next.

## Detection Code

```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_without_protected_attributes(df, feature_cols, n_clusters=2, random_state=42):
    """
    Runs k-means on a feature set that deliberately excludes protected
    attributes, so any demographic split found afterward is a side effect
    of the remaining features, not something the algorithm was given.

    Parameters:
        df: DataFrame containing feature_cols.
        feature_cols: columns to cluster on. Should NOT include the
            protected attributes you intend to check against.
        n_clusters: number of k-means clusters (k).
        random_state: seed for reproducible centroid initialisation.

    Returns:
        Series of cluster labels, indexed the same as df.
    """
    X = pd.get_dummies(df[feature_cols])
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return pd.Series(km.fit_predict(Xs), index=df.index, name="cluster")


def check_cluster_demographic_skew(df, cluster_labels, protected_col):
    """
    Cross-tabulates cluster assignment against a protected attribute that
    was NOT part of the clustering input, to check whether the clusters
    line up with group membership anyway.

    Parameters:
        df: DataFrame containing protected_col.
        cluster_labels: Series returned by cluster_without_protected_attributes.
        protected_col: name of the protected attribute to check, e.g. 'sex'.

    Returns:
        DataFrame of per-cluster proportions for each value of protected_col.
    """
    working = df[[protected_col]].copy()
    working["cluster"] = cluster_labels
    return pd.crosstab(working["cluster"], working[protected_col], normalize="index").round(3)


# Usage example
# clusters = cluster_without_protected_attributes(
#     df,
#     ['age', 'workclass', 'education.num', 'marital.status', 'occupation',
#      'relationship', 'capital.gain', 'capital.loss', 'hours.per.week'],
#     n_clusters=2,
# )
# check_cluster_demographic_skew(df, clusters, 'sex')
```

## Limitations and Trade-offs

### 1. There Is No Ground Truth to Evaluate Against

A supervised model's error rate can be checked against held-out labels. A clustering has no equivalent: there is no "correct" set of clusters to compare to, only internal measures like silhouette score that describe how tight and separated the clusters are, not whether they mean anything. Whether a clustering is "good" is a judgment call made by whoever interprets it, which is a much softer check than a held-out accuracy score.

### 2. `k` and the Distance Metric Are Assumptions, Not Discoveries

Nothing about the data tells you the "right" number of clusters. Choosing `k=2` versus `k=4` above changes which features dominate the split and can change whether a demographic pattern shows up strongly, weakly, or not at all, before a single row is clustered. The same is true of the distance metric: Euclidean distance on standardised features treats every dimension as equally important, which is itself a modelling choice with fairness consequences.

### 3. Disparate Impact Is Harder to Audit Without Labels

A supervised model's fairness gap is a single number computed against a known outcome. A clustering that visibly sorts people along demographic lines still has to be judged case by case: is this clustering being used to set insurance prices, in which case the ZIP-code precedent applies directly, or to generate exploratory customer segments that never touch a real decision? The same statistical pattern carries very different real-world risk depending on what happens downstream of it.

### 4. Dimensionality Reduction Can Erase the Signal You're Trying to Detect

Real clustering pipelines often run PCA or UMAP before k-means to cut down feature count. Both techniques preserve the directions of highest variance, not the directions most correlated with a protected attribute, so a preprocessing step aimed purely at compression can silently discard the exact signal a fairness audit would need to catch downstream.

## Related Concepts

* [What Is a Proxy Variable?](proxy-variables.md) - the mechanism (a feature correlating with a protected attribute) that this explainer shows operating without any label or supervision at all.
* [What Is Proxy Entanglement?](proxy-entanglement.md) - why dropping one proxy at a time fails when several correlated features jointly encode the same protected signal, visible above in how much of the sex split rode on `relationship` and `marital.status` together.
* [What Is Machine Learning Bias?](ml-bias.md) - places training data as one of the four entry points bias uses to reach a model; this explainer is what that entry point looks like with no label attached.
* [What Is Supervised Learning?](supervised-learning.md) - the labeled counterpart to this explainer, where the model is explicitly shown the outcome it is reproducing rather than discovering structure on its own.

## Related Projects in This Repo

* [`Benefits Denial/`](../Benefits%20Denial/) - the anchor for this explainer: the same `adult.csv` file the audit's supervised model trains on, clustered here with no label and no protected attribute in the feature set, and still recovering a strong sex split and a real, smaller race split.

## Further Reading

* [ProPublica: The Secret Bias Hidden in Mortgage-Approval Algorithms](https://www.propublica.org/article/the-secret-bias-hidden-in-mortgage-approval-algorithms) - documents ZIP-code and similar geographic clustering recovering protected-class structure in real lending and insurance pricing systems.
* [Chierichetti, Kumar, Lattanzi & Vassilvitskii (2017): Fair Clustering Through Fairlets, NeurIPS](https://papers.nips.cc/paper/2017/hash/978fce5bcc4eccc88ad48ce3914124a-Abstract.html) - the foundational paper formalising what it means for a clustering algorithm itself to be fair, and a concrete algorithm for enforcing it.
* [Barocas, Hardt & Narayanan: Fairness and Machine Learning](https://fairmlbook.org/) - covers unsupervised and representation-learning settings alongside the supervised case this repo's audits focus on.

*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
