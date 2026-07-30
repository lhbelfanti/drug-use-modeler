# Evaluation Results: Raw Corpus

This directory contains the evaluation results for models trained on the **raw corpus**.

> **Note**: For current dataset statistics, see [data/raw/README.md](../../data/raw/README.md). Split: 70/15/15, stratified jointly by class (POSITIVE/NEGATIVE) and substance (Cocaína/Marihuana/Heroína/Ecstasy), with cross-split text duplicates resolved (see `resolve_cross_split_duplicates` in [`02_preprocessing.ipynb`](../../notebooks/02_preprocessing.ipynb)). BETO's checkpoint is now selected on the validation split, and test is evaluated once — see [Iteration 3](../README.md#iteration-3-honest-beto-selection--deduplicated-split) in the evaluation log.

## Comparative Results

| Model | Technique | Standard Acc. | Irony Acc. | Obfuscated Acc. |
| :--- | :--- | :---: | :---: | :---: |
| **BERT (Base)** | Fine-tuned BETO | **85.56%** | **87.56%** | **87.78%** |
| **Logistic Regression** | TF-IDF + LogisticRegression | 80.67% | 80.67% | 81.78% |
| **TextCNN** | Word2Vec + Conv1D | 79.78% | 79.33% | 80.00% |
| **SVM** | TF-IDF + LinearSVC | 80.00% | 80.00% | 80.22% |
| **BiLSTM** | Word2Vec + BiLSTM | 79.11% | 77.56% | 80.22% |
| **Naive Bayes** | TF-IDF + MultinomialNB | 79.78% | 79.78% | 80.22% |
| **FFN** | Word2Vec + FFN | 78.00% | 74.67% | 77.78% |
| **Random Forest** | TF-IDF + RandomForest | 74.44% | 76.00% | 76.22% |

## Summary

**BERT (Base)** is the best model across all three variants of the raw corpus, reaching **85.56%** (Standard), **87.56%** (Irony), and **87.78%** (Obfuscated) accuracy — checkpoint selection now uses the validation split (not the test split), and training runs for the full 3 epochs. Notably, **Obfuscated is now BERT's best variant on the raw corpus, and the raw corpus as a whole now outperforms the pre-filtered corpus for BERT** (87.78% vs. 86.00%) — the opposite of the pre-filtered-favors-BERT pattern seen in Iteration 2. See [Iteration 3](../README.md#iteration-3-honest-beto-selection--deduplicated-split) for the full analysis.

**BERT (Base)** is the recommended model for the raw corpus.
