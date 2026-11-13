# Can AI Moderation Avoid Mislabeling Harmless Identity Mentions?

### Evaluating group-aware DistilBERT reweighting for toxicity detection, subgroup fairness, and identity-swap robustness

**Author:** Ariana Islam  
**Research area:** Natural Language Processing · Responsible AI · Algorithmic Fairness  
**Dataset:** [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)

## Overview

Automated moderation models can mistake harmless references to identities for toxic language. This empirical study examines whether **group-aware sample reweighting** changes the behavior of a DistilBERT toxicity classifier while preserving its ability to recognize harmful comments.

Four configurations were compared: standard TF-IDF + logistic regression, class-balanced TF-IDF + logistic regression, standard DistilBERT, and group-weighted DistilBERT. Evaluation covers classification quality, false-positive rates across **six identity-mention groups**, and the stability of predictions under controlled identity-word substitutions.

**Main finding:** Reweighting reduced the mean absolute change in model scores on 144 constructed identity-swap pairs by **25.48%**, with a positive template-clustered bootstrap confidence interval. It did **not** establish a statistically reliable improvement in subgroup false-positive-rate parity or overall toxic-class F1.

> **Interpretation:** The results support improved robustness on the specific constructed substitutions tested—not a general claim that the weighted model is fairer in real-world moderation.

## Research questions

1. How well do classical TF-IDF baselines and DistilBERT detect toxic comments?
2. How do false-positive rates vary across comments mentioning different identities?
3. Can group-aware loss weighting reduce subgroup disparities without degrading classification performance?
4. How sensitive are standard and weighted DistilBERT to substitutions of identity terms in otherwise identical sentences?

## Dataset and experimental design

The original Jigsaw file contained **1,804,874** comments. After removing missing text, excluding conflicting-label texts, and removing exact-text duplicates, **1,780,226 unique comments** remained. Toxicity was binarized at a target threshold of **0.5**.

| Evaluation split | Size | Purpose |
| --- | ---: | --- |
| Training | 20,000 | Fit baseline and transformer models |
| Validation | 5,000 | Select decision thresholds and checkpoints |
| Representative held-out test | 5,000 | Measure classification performance |
| Identity-enriched held-out test | 3,000 | Examine subgroup errors |
| Constructed identity-swap pairs | 144 pairs | Test sensitivity to controlled substitutions |

The six identity-mention categories were **male, female, Christian, Muslim, Black, and White**. These refer to identities *mentioned in comments*, not to the demographics of comment authors. Identity groups can overlap.

## Approach

**Baseline models:** TF-IDF text features with ordinary and class-balanced logistic regression.

**Transformer models:** Fine-tuning `distilbert-base-uncased` with either standard classification loss or a group-aware weighted cross-entropy loss that increases the contribution of selected underrepresented identity/toxicity combinations.

**Model selection:** Validation toxic-class F1 was used to select checkpoints and classification thresholds. The two test sets were reserved for final evaluation.

**Fairness analysis:** In addition to precision, recall, toxic-class F1, and ROC-AUC, the study measured the false-positive rate (FPR) among non-toxic comments for each identity-mention group. The reported subgroup FPR gap is the difference between the highest and lowest group FPRs.

**Counterfactual robustness:** Constructed sentence pairs differ only in an identity term. Mean absolute changes in predicted toxicity scores were compared across models. Bootstrap analyses quantified uncertainty, including clustering by original sentence template for the identity-swap evaluation.

### Transformer training configuration

| Setting | Value |
| --- | --- |
| Checkpoint | `distilbert-base-uncased` |
| Epochs | 2 |
| Learning rate | 2 × 10⁻⁵ |
| Training microbatch | 8 |
| Gradient accumulation | 2 (effective batch size: 16) |
| Validation batch size | 16 |
| Maximum sequence length | 256 tokens |
| Weight decay | 0.01 |
| Random seed | 42 |

## Results

### Representative held-out test (5,000 comments)

| Model | Accuracy | Precision | Recall | Toxic F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary logistic regression | 0.9168 | 0.4773 | 0.4486 | 0.4625 | 0.8493 |
| Balanced logistic regression | 0.9224 | 0.5179 | 0.3985 | 0.4504 | 0.8496 |
| **Standard DistilBERT** | **0.9386** | **0.6230** | **0.5840** | **0.6028** | **0.9333** |
| Weighted DistilBERT | 0.9382 | 0.6203 | 0.5815 | 0.6003 | 0.9302 |

The standard and weighted transformer models performed similarly; weighting did not improve the observed representative-test toxic-class F1.

### Group error rates and identity-swap sensitivity

| Comparison | Standard DistilBERT | Weighted DistilBERT | Interpretation |
| --- | ---: | ---: | --- |
| Identity-enriched subgroup FPR gap | 3.60 percentage points | 3.22 percentage points | Smaller observed gap; not statistically established |
| Mean absolute identity-swap score change | 0.009168 | 0.006832 | **25.48% relative reduction** on constructed pairs |

**Bootstrap uncertainty (5,000 repetitions):**

| Difference (metric units) | Observed | 95% bootstrap interval |
| --- | ---: | --- |
| Representative toxic F1 (weighted − standard) | −0.002587 | [−0.015576, 0.009725] |
| Enriched subgroup FPR gap (weighted − standard) | −0.003710 | [−0.018862, 0.006745] |
| Identity-swap sensitivity (standard − weighted) | +0.002336 | [0.000983, 0.004191] |

Only the identity-swap sensitivity interval excludes zero under the stated bootstrap procedure. The observed FPR gap decrease should not be presented as a proven subgroup fairness improvement.

## Limitations

- Each transformer configuration was trained with a single random seed; bootstrapping held-out predictions does not account for variation across retraining runs.
- Only 20,000 of the 1.78 million cleaned comments were used for training.
- The identity-enriched test set is not representative of natural comment prevalence, and identity groups overlap.
- The 144 identity-swap pairs were constructed templates and may not reflect natural conversational usage.
- Model-specific validation thresholds affect threshold-dependent FPR comparisons.

## Reproducibility and code

**Core tools:** Python, PyTorch, Hugging Face Transformers, scikit-learn, pandas.

The research report documents the experimental settings, evaluation methodology, and uncertainty analysis. To reproduce the project, obtain the Jigsaw dataset from the competition page and follow the code and dependency instructions provided with the repository.

> **Repository setup:** Add verified script entry points, required package versions, and commands here once the uploaded source files and directory paths are organized. No untested run commands are listed in this README.

## Repository contents

This repository contains project code and documentation for the toxicity-fairness study. See the file browser above for the current folder and script layout.

**Recommended additions:** a `requirements.txt` or `pyproject.toml`, documented run commands, figures exported from the experiments, and the full PDF research report. Avoid committing the full Jigsaw dataset or model checkpoints that exceed GitHub's size limits.

## Future work

- Repeat transformer training across several random seeds.
- Explore larger training samples and alternative reweighting strategies.
- Compare subgroup metrics at matched overall FPR or recall.
- Test on additional toxicity datasets and more natural, manually reviewed identity swaps.

## Research contribution

This project contributes an **empirical evaluation framework**, not a new transformer architecture: baseline construction, dataset cleaning, weighted-loss implementation, model fine-tuning, threshold selection, identity-group error analysis, counterfactual robustness testing, and bootstrap uncertainty estimation.

## Citation

If you use or discuss this project, please credit:

**Ariana Islam.** *Can AI Moderation Avoid Mislabeling Harmless Identity Mentions? Testing Group-Aware Training with DistilBERT on Jigsaw Comments.* Research report, 2026.

---

*Fairness is multidimensional. Better stability under controlled text edits does not necessarily imply equal error rates across groups.*
