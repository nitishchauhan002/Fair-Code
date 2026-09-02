# Explainer: Why AI Hallucinates

> *When training data is sparse, models optimize for plausibility over truth - leading to high-confidence fabrications.*

---

## The One-Sentence Definition

A model **"hallucinates"** when it outputs fluent, confident text or predictions that are factually wrong or entirely fabricated, because it is optimising for plausible outputs rather than verified facts.

---

## Why It Matters (Domain Dependence & High Stakes)

Hallucination is highly domain-dependent and becomes exponentially more dangerous in high-stakes fields. As explored in ["Cognitive Mirage: A Review of Hallucinations in Large Language Models" (Ye et al., 2023)](https://arxiv.org/abs/2309.06794), a hallucination in casual conversation is merely an annoyance, but in legal tech, healthcare, or finance, it can have severe real-world consequences.

Most users assume that if a model outputs a prediction or text with high confidence, it must have strong evidence to support it. This is a fundamental misunderstanding of machine learning.

A model does not possess a concept of "truth" or "facts." Instead, it is a statistical pattern matcher that estimates probability distributions based on its training data. When a model is queried in a region of the feature space where training data is sparse or non-existent, it does not stop or output a warning. Instead, it extrapolates from the patterns it learned in dense regions. 

Because the model's loss function penalises implausible or poorly formatted outputs, the model prioritises generating a *plausible-sounding* output. In sparse regions, this optimization force creates **hallucination**: highly confident, syntactically perfect predictions that are completely unsupported by the evidence. This failure mode behaves identically whether the model is a Large Language Model (LLM) fabricating legal precedents or a tabular random forest classifier confidently denying health insurance claims.

---

## Real-World Case: Mata v. Avianca, Inc. (678 F.Supp.3d 443)

The primary source and most famous, legally documented illustration of AI hallucination is the federal court case [*Mata v. Avianca, Inc.* (2023)](https://www.courtlistener.com/docket/67104031/mata-v-avianca-inc/). 

An attorney representing a plaintiff suing Avianca Airlines used ChatGPT to research legal precedents. The attorney asked ChatGPT to find cases supporting his argument. ChatGPT responded with a list of highly relevant, detailed cases, including *Martinez v. Delta Air Lines* and *Varghese v. China Southern Airlines Co.*, complete with docket numbers, dates, and elaborate internal quotes.

The problem: **none of these cases existed.** ChatGPT had entirely fabricated them. 

Because the model was trained on the language of legal briefs, it knew exactly what a valid court citation *should* look like (fluent, structured, citing specific airline cases). When queried about a specific legal scenario where actual training evidence was thin, ChatGPT optimized for *plausible legal syntax* over historical facts, generating fictional precedents that looked identical to real ones. The attorney, trusting the AI's confident tone, submitted the brief to a federal judge and was subsequently sanctioned and fined.

---

## Tabular Proof: Insurance Denial Model

We can see this exact same failure mode in the `Insurance Denial` audit in this repository. 

The biased model (`unfair.py`) is trained on a dataset of patient features to predict whether an insurance claim will be high-cost (which flags it for denial). Among the features are `bmi`, `smoker` status, and `diabetic` status. 

In this dataset, smoking is a dominant predictive signal: smokers are highly likely to have high claim costs. The model splits on this feature early. However, when we look at underweight patients who smoke and are diabetic, the training evidence is incredibly thin. 

By grouping the entire dataset into bins of BMI category, smoking status, and diabetes status, we can compare the **Sample Density** (how many patients fit this profile in the training data) against the **Model's Prediction Probability of Denial**:

| BMI Category | Smoker | Diabetic | Sample Count (Density) | Mean Predicted Probability of Denial | Mean Prediction |
|---|---|---|:---:|:---:|:---:|
| **Underweight** | **Yes** | **No** | **1** | **98.0%** | **Deny (1.0)** |
| **Underweight** | **Yes** | **Yes** | **4** | **95.3%** | **Deny (1.0)** |
| Underweight | No | Yes | 6 | 28.0% | Approve (0.0) |
| Underweight | No | No | 9 | 33.9% | Approve (0.0) |
| Normal | Yes | No | 22 | 99.1% | Deny (1.0) |
| Normal | Yes | Yes | 28 | 99.3% | Deny (1.0) |
| Overweight | Yes | Yes | 35 | 99.5% | Deny (1.0) |
| Overweight | Yes | No | 39 | 99.5% | Deny (1.0) |
| Obese | Yes | Yes | 69 | 99.6% | Deny (1.0) |
| Obese | Yes | No | 76 | 99.7% | Deny (1.0) |
| Normal | No | No | 83 | 31.6% | Approve (0.0) |
| Normal | No | Yes | 90 | 28.9% | Approve (0.0) |
| Overweight | No | Yes | 148 | 36.4% | Approve (0.0) |
| Overweight | No | No | 167 | 33.2% | Approve (0.0) |
| Obese | No | Yes | 262 | 43.6% | Approve (0.0) |
| Obese | No | No | 301 | 42.5% | Approve (0.0) |

### The Hallucination Mechanism

Look at the first two rows. There is only **one** underweight smoker in the entire dataset, and only **four** underweight smokers who are diabetic. The training evidence for these sub-populations is virtually non-existent. 

Yet, the model predicts a **98.0%** and **95.3% probability of denial** for these applicants. 

The model is "hallucinating" certainty. Because it has plenty of training data showing that obese and overweight smokers are denied, and because it has no local density to tell it that underweight smokers might have different risk profiles, it confidently extrapolates the "smoker = high risk" rule to the underweight region. The model outputs near-absolute certainty (95%+ confidence) in a region of the feature space where it has almost zero evidence.

This is a classic failure of Out-of-Distribution (OOD) Detection and Probability Calibration. As outlined in the [Scikit-learn Documentation on Probability Calibration](https://scikit-learn.org/stable/modules/calibration.html), a classifier's raw output score is rarely a true probability. Without explicit calibration, the model lacks the "scaffolding" to know when it is guessing in the dark.

---

## Detection Code & Verification Scaffolding

To prevent AI systems from hallucinating high-confidence decisions in sparse regions, you must audit the feature space for density. If a model is making a high-confidence prediction on a query that falls into an under-represented combination of features, the system should flag it for human review or refuse to predict.

```python
import pandas as pd
import numpy as np

def audit_hallucination_risk(df, cat_cols, prob_col, density_threshold=10, confidence_threshold=0.9):
    """
    Audit a model's predictions to find feature combinations where the model
    makes highly confident predictions despite thin training evidence.
    
    cat_cols: list of categorical columns defining the feature space
    prob_col: column name containing the predicted probability of the positive class
    density_threshold: minimum number of samples required to consider a region dense
    confidence_threshold: probability above which a prediction is considered highly confident
    """
    # Calculate density for each combination in the dataset
    density_map = df.groupby(cat_cols).size().reset_index(name='density')
    
    # Calculate model statistics for each combination
    stats = df.groupby(cat_cols).agg(
        mean_prob=(prob_col, 'mean'),
        max_prob=(prob_col, 'max')
    ).reset_index()
    
    # Merge density and prediction stats
    audit_df = pd.merge(stats, density_map, on=cat_cols)
    
    # Flag combinations with low density but high confidence
    audit_df['hallucination_risk'] = (
        (audit_df['density'] < density_threshold) & 
        ((audit_df['mean_prob'] >= confidence_threshold) | (audit_df['mean_prob'] <= (1 - confidence_threshold)))
    )
    
    flagged = audit_df[audit_df['hallucination_risk'] == True]
    
    if not flagged.empty:
        print(f"⚠ WARNING: Detected {len(flagged)} combinations with high hallucination risk:")
        for _, row in flagged.iterrows():
            comb_desc = ", ".join([f"{col}={row[col]}" for col in cat_cols])
            print(f"  · Combo: [{comb_desc}] | Density: {row['density']} | Confidence: {row['mean_prob']:.2%}")
    else:
        print("✓ No high-risk sparse combinations detected.")
        
    return audit_df

# Example usage with simulated output matching our Insurance Denial audit
data = {
    'bmi_cat': ['Underweight', 'Underweight', 'Obese', 'Obese'],
    'smoker': ['Yes', 'No', 'Yes', 'No'],
    'diabetic': ['No', 'No', 'No', 'No'],
    'prob_denial': [0.98, 0.33, 0.99, 0.42]
}
df_sim = pd.DataFrame(data)

# Let's assume we also have matching count frequencies in our dataset
# We expand the df to simulate the densities listed in our table
expanded_data = []
for _, row in df_sim.iterrows():
    count = 1 if row['bmi_cat'] == 'Underweight' and row['smoker'] == 'Yes' else (
            9 if row['bmi_cat'] == 'Underweight' else (
            76 if row['bmi_cat'] == 'Obese' and row['smoker'] == 'Yes' else 301))
    for _ in range(count):
        expanded_data.append(row.to_dict())
df_expanded = pd.DataFrame(expanded_data)

audit_hallucination_risk(df_expanded, ['bmi_cat', 'smoker', 'diabetic'], 'prob_denial')
# Output:
# ⚠ WARNING: Detected 1 combinations with high hallucination risk:
#   · Combo: [bmi_cat=Underweight, smoker=Yes, diabetic=No] | Density: 1 | Confidence: 98.00%
```

For generative text tasks, similar "System 2" measurement and scaffolding frameworks exist:
* **[Ragas (Retrieval Augmented Generation Assessment)](https://docs.ragas.io/en/stable/)**: A framework for evaluating faithfulness, context precision, and hallucination mathematically.
* **[LangGraph Multi-Agent Architecture](https://towardsai.net/p/machine-learning/langgraph-multi-agent-architecture-building-a-self-critiquing-ai-debate-system)**: Building a self-critiquing AI debate system, highly relevant for building adversarial verification loops and stateful scaffolding.
* **[LangChain Reflection Agents](https://www.langchain.com/blog/reflection-agents)**: Official documentation on implementing reflective and adversarial evaluation steps.

---

## How to Mitigate It

Reducing hallucination in high-stakes domains-such as legal tech, medicine, or finance where incorrect outputs have severe real-world consequences-requires moving beyond training better models. As practitioners in legal tech (such as AI Workdeck) have noted, the fix is not better models alone, but better *scaffolding* around them.

The key patterns for reducing hallucination include:

1. **Retrieval-First Architecture:** Instead of relying on the model's parameterized memory for factual lookup, force the model through a retrieval step. The model must locate relevant verified documents first, and then synthesize the response from those documents *only*.
2. **Source Grounding:** Require that every generated output traces back to a specific document passage or citation. If a generated claim cannot be mathematically mapped to a retrieved source text, the output is flagged as ungrounded and rejected.
3. **Adversarial Verification:** Implement a multi-agent validation step where a second model pass acts as a critic, challenging the primary model's outputs (e.g., asking: *"Does this case citation actually exist in the database? Does the summary faithfully represent the source passage?"*).
4. **Confidence Calibration:** Models are often confidently wrong. Calibrating the model's confidence scores against actual accuracy rates for the target domain ensures that the probability score aligns with reality, helping users understand when they can trust the output.

---

## Limitations and Trade-offs

Mitigating hallucination requires navigating structural trade-offs in model architecture and deployment rules. As detailed in [*Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models*](https://arxiv.org/abs/2309.01219), these trade-offs include:

### 1. "Hallucination" Conflates Distinct Failure Modes
Using the term "hallucination" as a catch-all is misleading because it groups different technical failures based on exact taxonomy:
* **Confabulation:** The model invents facts entirely out of thin air (e.g., ChatGPT creating *Varghese v. China Southern Airlines*).
* **Extrinsic Hallucination / Context-Conflict:** The model outputs details that are factually plausible but cannot be verified from the input context (e.g., summarizing an article and adding unmentioned background facts).
* **Intrinsic Hallucination / Input-Conflict:** The model directly contradicts the provided source context (e.g., misreading a medical report and stating a negative test was positive).
Mitigation strategies differ: extrinsic hallucinations require strict context grounding, whereas confabulations require factual database retrieval.

### 2. Retrieval-Augmented Generation (RAG) is Not a Cure
RAG restricts a model's generation space by supplying verified external documents (e.g., searching a database of real court cases before answering). While RAG drastically reduces hallucination rates, it does not solve it. Models can still misinterpret highly complex sentences, ignore parts of the retrieved documents, or synthesize conflicting sources into a new, confidently wrong claim.

### 3. Refusal Fine-Tuning Degrades Helpfulness (RLHF Over-Conservatism)
To prevent hallucination, models are often fine-tuned (e.g., via Reinforcement Learning from Human Feedback) to refuse prompts when they are uncertain. However, this introduces a severe trade-off between honesty and helpfulness:
* **Cautiousness Creep:** Models become overly timid, refusing to answer benign queries that they actually have sufficient evidence for, reducing their overall utility.
* **Calibration Distortion:** Forcing a model to output a canned refusal statement when its internal probability falls below a threshold corrupts its confidence scoring. Instead of a model saying "I am 55% sure of this," it simply refuses, destroying valuable probabilistic nuance. This connects directly to the [Calibration explainer](calibration.md).

---

## Related Concepts

* [Calibration](calibration.md) - Measures whether a model's confidence matches its actual accuracy. Hallucinating models are severely miscalibrated in sparse regions.
* [Sampling Bias](sampling-bias.md) - When certain feature combinations are completely omitted or underrepresented in the training data, it guarantees the existence of the sparse spaces where models are forced to hallucinate.
* **Retrieval-Augmented Generation (RAG) Architecture** - How it mitigates but doesn't solve hallucination. (See: IBM Research on *What is RAG?*)
* **Sycophancy in LLMs** - When models hallucinate to agree with the user. (See: Anthropic Research on *Sycophancy*)

---

## Related Projects in This Repo

* [`Insurance Denial/`](../Insurance%20Denial/) - The healthcare denial audit where the biased model (`unfair.py`) contains the exact sparse BMI, smoking, and diabetic combinations where the model hallucinates 95%+ denial probabilities on thin training data.

---

## Further Reading

* [Mata v. Avianca, No. 1:22-cv-01461 (S.D.N.Y. 2023)](https://www.courtlistener.com/docket/63107798/mata-v-avianca-inc/) - U.S. District Court docket containing Judge P. Kevin Castel's complete sanctions opinion detailing the ChatGPT-fabricated cases.
* [Ji et al. (2023): Survey of Hallucination in Natural Language Processing](https://arxiv.org/abs/2202.03629) - The definitive academic survey classifying the types, causes, and detection methods for generative hallucinations.
* [Rawte et al. (2023): A Survey of Hallucination in Large Foundation Models](https://arxiv.org/abs/2309.05922)
* [McKenna et al. (2023): Sources of Hallucination by Large Language Models on Inference Tasks](https://aclanthology.org/2023.emnlp-main.245/)
* [Chouldechova (2017): Fair Prediction with Disparate Impact](https://arxiv.org/abs/1703.00056) - Foundational paper on metric trade-offs, showing how calibration requirements can force models to rely on coarse, unfair correlations in sparse subgroups.

---

*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
