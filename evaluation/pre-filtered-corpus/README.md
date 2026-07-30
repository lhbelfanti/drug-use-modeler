# Evaluation Results: Pre-filtered Corpus

This directory contains the evaluation results for models trained on the **pre-filtered corpus**.

> **Note**: For current dataset statistics, see [data/raw/README.md](../../data/raw/README.md). Split: 70/15/15, stratified jointly by class (POSITIVE/NEGATIVE) and substance (Cocaína/Marihuana/Heroína/Ecstasy), with cross-split text duplicates resolved (see `resolve_cross_split_duplicates` in [`02_preprocessing.ipynb`](../../notebooks/02_preprocessing.ipynb)). BETO's checkpoint is now selected on the validation split, and test is evaluated once — see [Iteration 3](../README.md#iteration-3-honest-beto-selection--deduplicated-split) in the evaluation log.

## Comparative Results

| Model | Technique | Standard Acc. | Irony Acc. | Obfuscated Acc. |
| :--- | :--- | :---: | :---: | :---: |
| **BERT (Base)** | Fine-tuned BETO | **86.00%** | **85.33%** | **85.78%** |
| **Logistic Regression** | TF-IDF + LogisticRegression | 80.22% | 80.44% | 78.89% |
| **SVM** | TF-IDF + LinearSVC | 79.78% | 79.11% | 79.78% |
| **TextCNN** | Word2Vec + Conv1D | 77.33% | 79.33% | 78.22% |
| **Random Forest** | TF-IDF + RandomForest | 76.67% | 79.11% | 76.44% |
| **BiLSTM** | Word2Vec + BiLSTM | 76.67% | 78.44% | 75.11% |
| **Naive Bayes** | TF-IDF + MultinomialNB | 77.33% | 77.78% | 76.89% |
| **FFN** | Word2Vec + FFN | 72.89% | 76.67% | 75.78% |

## Summary

The **BERT (Base)** model achieves the best performance across all three variants of the pre-filtered corpus, reaching **86.00%** (Standard), **85.33%** (Irony), and **85.78%** (Obfuscated) accuracy — checkpoint selection now uses the validation split (not the test split), and training runs for the full 3 epochs. BERT leads consistently across all three variants; no classical or Word2Vec-based model comes within 5 points of it here.

**BERT (Base)** is the recommended model for the pre-filtered corpus across all three pipeline variants.
