# Evaluation Log

This document serves as a chronological log of model evaluations, tracking performance improvements and experiments over time.

## Iteration 1: Initial Model Baseline

**Goal**: Establish baseline performance for standard, irony-augmented, and obfuscated datasets.

### Dataset Details

-   **Total Samples**: 3,000 for each corpus (3,000 Raw Corpus / 3,000 Pre-filtered Corpus).
-   **Full Statistics**: See **[data/raw/README.md](../data/raw/README.md)**.

### Detailed Reports

-   **[Pre-filtered Corpus Results](./pre-filtered-corpus/README.md)**: Models trained on the clean, filtered dataset.
-   **[Raw Corpus Results](./raw-corpus/README.md)**: Models trained on the full, unfiltered dataset.

### Summary & Best Models

#### Best Model Rankings

| Corpus Variation | Best Model | Accuracy | Notes |
| :--- | :--- | :---: | :--- |
| **Pre-filtered** | **BERT (Base)** | **86.22%** | Best overall performance. Finetuning shines on cleaner data. |
| **Raw** | **Naive Bayes** | **83.78%** | Surprisingly robust to noise. Outperforms BERT on raw data. |

#### Key Findings

1.  **Data Quality Matters for Deep Learning**: BERT's performance drops significantly (from ~86% to ~82%) when moving from the pre-filtered to the raw corpus. This suggests that the "noise" in the raw corpus (which might include irrelevant tweets or confusing structures) negatively impacts the model's ability to learn effectively.
2.  **Robustness of Simple Models**: Naive Bayes remains remarkably consistent across both datasets (~83%). It serves as a very strong baseline and might be preferable for low-latency or low-resource scenarios where data cleaning is difficult.
3.  **Irony Handling**: Most models show comparable performance on the "Irony" dataset variation, often slightly better than the "Standard" or "Obfuscated" versions. This is an interesting phenomenon that warrants further investigation—perhaps the irony markers are strong features.

### Next Steps

Based on these results, the following actions are recommended:

1.  **Error Analysis**:
    - Investigate *why* BERT fails on the raw corpus. Are there specific types of tweets that confuse it?
    - Analyze the misclassified examples from the Naive Bayes model to see if there's a pattern that BERT captures but Naive Bayes misses (and vice-versa).
2.  **Data Augmentation**:
    - Explore if data augmentation techniques can help BERT generalize better on the raw corpus.
3.  **Ensemble Methods**:
    - Consider an ensemble of BERT and Naive Bayes, as they might be making different types of errors.
