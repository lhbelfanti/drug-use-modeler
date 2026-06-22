# Evaluation Log

This document serves as a chronological log of model evaluations, tracking performance improvements and experiments over time.

## Iteration 1: Initial Model Baseline

**Goal**: Establish baseline performance for standard, irony-augmented, and obfuscated datasets.

### Dataset Details

-   **Total Samples**: 3,000 for each corpus (3,000 Raw Corpus / 3,000 Pre-filtered Corpus).
-   **Full Statistics**: See **[data/raw/README.md](../data/raw/README.md)**.

### Metrics Reference

Full results — including per-class Precision, Recall, and F1-Score for NEGATIVE and POSITIVE — are available in machine-readable form:

-   **[evaluation/metrics.json](./metrics.json)**: Structured JSON with macro-averaged and per-class metrics.
-   **[evaluation/metrics.csv](./metrics.csv)**: Flat table (corpus / variant / model / accuracy / precision / recall / f1) for easy filtering and comparison.

All metrics are computed by running inference with the saved model artifacts on each `test.csv` split. For raw-corpus Naive Bayes, SVM, and Random Forest (no saved joblib), models are re-trained inline with identical hyperparameters (seed=42) using **[scripts/evaluate_all.py](../scripts/evaluate_all.py)**.

> **Note**: The raw-corpus NB/SVM/RF results in the sub-README were incorrectly copied from the pre-filtered corpus. The values below reflect the correct re-evaluated numbers.

---

### Detailed Reports

-   **[Pre-filtered Corpus Results](./pre-filtered-corpus/README.md)**: Models trained on the clean, filtered dataset.
-   **[Raw Corpus Results](./raw-corpus/README.md)**: Models trained on the full, unfiltered dataset.

---

### Full Results: Pre-filtered Corpus

All metrics are macro-averaged. Sorted by F1-Score descending.

#### Standard Variant

| Model | Technique | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BERT (Base)** | Fine-tuned BETO | **86.22%** | **86.25%** | **86.22%** | **86.22%** |
| **Naive Bayes** | TF-IDF + MultinomialNB | 83.78% | 83.79% | 83.78% | 83.78% |
| **Logistic Regression** | TF-IDF + LogisticRegression | 82.00% | 82.14% | 82.00% | 81.98% |
| **CNN** | Word2Vec + Conv1D | 82.00% | 82.14% | 82.00% | 81.98% |
| **SVM** | TF-IDF + LinearSVC | 81.56% | 81.63% | 81.56% | 81.54% |
| **Random Forest** | TF-IDF + RandomForest | 78.44% | 78.54% | 78.44% | 78.43% |
| **BiLSTM** | Word2Vec + BiLSTM | 78.44% | 78.47% | 78.44% | 78.44% |
| **FFN** | Word2Vec + FFN | 76.89% | 76.92% | 76.89% | 76.88% |

#### Irony Variant

| Model | Technique | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BERT (Base)** | Fine-tuned BETO | **85.33%** | **85.34%** | **85.33%** | **85.33%** |
| **Naive Bayes** | TF-IDF + MultinomialNB | 83.33% | 83.35% | 83.33% | 83.33% |
| **Logistic Regression** | TF-IDF + LogisticRegression | 82.00% | 82.14% | 82.00% | 81.98% |
| **SVM** | TF-IDF + LinearSVC | 81.78% | 81.84% | 81.78% | 81.77% |
| **CNN** | Word2Vec + Conv1D | 80.67% | 81.05% | 80.67% | 80.61% |
| **Random Forest** | TF-IDF + RandomForest | 79.78% | 79.99% | 79.78% | 79.74% |
| **BiLSTM** | Word2Vec + BiLSTM | 78.22% | 78.33% | 78.22% | 78.20% |
| **FFN** | Word2Vec + FFN | 77.11% | 77.14% | 77.11% | 77.11% |

#### Obfuscated Variant

| Model | Technique | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Naive Bayes** | TF-IDF + MultinomialNB | **83.11%** | **83.13%** | **83.11%** | **83.11%** |
| **Logistic Regression** | TF-IDF + LogisticRegression | 81.78% | 81.94% | 81.78% | 81.75% |
| **SVM** | TF-IDF + LinearSVC | 81.56% | 81.61% | 81.56% | 81.55% |
| **BERT (Base)** | Fine-tuned BETO | 80.44% | 80.50% | 80.44% | 80.43% |
| **CNN** | Word2Vec + Conv1D | 80.89% | 81.09% | 80.89% | 80.86% |
| **Random Forest** | TF-IDF + RandomForest | 79.56% | 79.67% | 79.56% | 79.54% |
| **BiLSTM** | Word2Vec + BiLSTM | 79.11% | 79.34% | 79.11% | 79.07% |
| **FFN** | Word2Vec + FFN | 76.00% | 76.10% | 76.00% | 75.98% |

---

### Full Results: Raw Corpus

All metrics are macro-averaged. Sorted by F1-Score descending.

#### Standard Variant

| Model | Technique | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **SVM** | TF-IDF + LinearSVC | **82.22%** | **82.48%** | **82.22%** | **82.19%** |
| **BERT (Base)** | Fine-tuned BETO | 81.78% | 81.78% | 81.78% | 81.78% |
| **Naive Bayes** | TF-IDF + MultinomialNB | 80.67% | 80.74% | 80.67% | 80.66% |
| **Logistic Regression** | TF-IDF + LogisticRegression | 80.00% | 80.35% | 80.00% | 79.94% |
| **Random Forest** | TF-IDF + RandomForest | 76.89% | 76.97% | 76.89% | 76.87% |
| **BiLSTM** | Word2Vec + BiLSTM | 76.44% | 77.06% | 76.44% | 76.31% |
| **CNN** | Word2Vec + Conv1D | 76.22% | 76.95% | 76.22% | 76.06% |
| **FFN** | Word2Vec + FFN | 76.00% | 76.54% | 76.00% | 75.88% |

#### Irony Variant

| Model | Technique | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BERT (Base)** | Fine-tuned BETO | **82.44%** | **82.45%** | **82.44%** | **82.44%** |
| **SVM** | TF-IDF + LinearSVC | 82.44% | 82.63% | 82.44% | 82.42% |
| **Naive Bayes** | TF-IDF + MultinomialNB | 80.44% | 80.48% | 80.44% | 80.44% |
| **Logistic Regression** | TF-IDF + LogisticRegression | 80.22% | 80.54% | 80.22% | 80.17% |
| **BiLSTM** | Word2Vec + BiLSTM | 78.22% | 78.22% | 78.22% | 78.22% |
| **CNN** | Word2Vec + Conv1D | 77.56% | 78.09% | 77.56% | 77.45% |
| **FFN** | Word2Vec + FFN | 76.89% | 77.15% | 76.89% | 76.83% |
| **Random Forest** | TF-IDF + RandomForest | 76.67% | 76.82% | 76.67% | 76.63% |

#### Obfuscated Variant

| Model | Technique | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **SVM** | TF-IDF + LinearSVC | **83.11%** | **83.32%** | **83.11%** | **83.08%** |
| **BERT (Base)** | Fine-tuned BETO | 82.00% | 82.02% | 82.00% | 82.00% |
| **Naive Bayes** | TF-IDF + MultinomialNB | 81.11% | 81.22% | 81.11% | 81.10% |
| **Logistic Regression** | TF-IDF + LogisticRegression | 80.67% | 81.05% | 80.67% | 80.61% |
| **Random Forest** | TF-IDF + RandomForest | 77.78% | 78.05% | 77.78% | 77.72% |
| **CNN** | Word2Vec + Conv1D | 76.44% | 76.75% | 76.44% | 76.38% |
| **BiLSTM** | Word2Vec + BiLSTM | 75.56% | 76.15% | 75.56% | 75.42% |
| **FFN** | Word2Vec + FFN | 73.33% | 74.65% | 73.33% | 72.97% |

---

### Summary & Best Models

#### Best Model Rankings

| Corpus | Best Model | Accuracy | Precision | Recall | F1 | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Pre-filtered** | **BERT (Base)** — Standard | **86.22%** | **86.25%** | **86.22%** | **86.22%** | Best across all variants. Fine-tuning shines on cleaner data. |
| **Raw** | **SVM** — Obfuscated | **83.11%** | **83.32%** | **83.11%** | **83.08%** | Consistently strong across all raw-corpus variants. |

#### Key Findings

1.  **Data Quality Matters for Deep Learning**: BERT's performance drops significantly (from ~86% to ~82%) when moving from the pre-filtered to the raw corpus. This suggests that the noise in the raw corpus negatively impacts the model's ability to learn effectively. Simple TF-IDF models are far less sensitive to this.

2.  **SVM is the Strongest Baseline on Raw Data**: Contrary to the initial assumption, SVM (not Naive Bayes) achieves the highest F1 across all raw-corpus variants once metrics are computed correctly. It even matches BERT on the irony variant (82.42% vs 82.44% F1) while being orders of magnitude cheaper to run.

3.  **Precision ≈ Recall for Most Models**: The balanced class distribution (225 NEGATIVE / 225 POSITIVE in each test split) means macro-averaged Precision and Recall track each other closely. The few exceptions — notably FFN on obfuscated data — reveal models that default toward one class.

4.  **Obfuscation Hurts BERT Most**: BERT drops ~5.8 F1 points from Standard to Obfuscated on the pre-filtered corpus (86.22% → 80.43%), suggesting it relies on named entities that obfuscation removes. Naive Bayes and SVM are far more robust to this transformation.

5.  **Irony Handling**: Most models score comparably or slightly higher on the Irony variant than on Standard. The `[IRONIA]` tag appears to act as a strong, learnable feature for TF-IDF-based models in particular.

6.  **Word2Vec models (FFN, CNN, BiLSTM) underperform consistently**: All three trail TF-IDF-based models across both corpora and all variants, suggesting that the pre-trained word vectors (100-dimensional, trained on this small dataset) do not capture enough signal. Better embeddings or larger training data would likely close the gap.

---

### Next Steps

Based on these results, the following actions are recommended:

1.  **Error Analysis**:
    - Investigate *why* BERT fails on the raw corpus. Are there specific types of tweets that confuse it?
    - Analyze cases where SVM and BERT disagree on raw-corpus examples — they appear to make different errors and could be complementary.
2.  **Data Augmentation**:
    - Explore if data augmentation techniques can help BERT generalize better on the raw corpus and recover from obfuscation.
3.  **Ensemble Methods**:
    - Consider an ensemble of BERT and SVM, as they appear to be the strongest complementary pair across corpora.
4.  **Better Embeddings for Neural Models**:
    - Replace the small in-domain Word2Vec vectors with pre-trained multilingual embeddings (e.g., fastText) to close the gap between FFN/CNN/BiLSTM and TF-IDF baselines.
