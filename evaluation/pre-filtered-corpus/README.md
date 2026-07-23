# Evaluation Results: Pre-filtered Corpus

This directory contains the evaluation results for models trained on the **pre-filtered corpus**.

> **Note**: For current dataset statistics, see [data/raw/README.md](../../data/raw/README.md). Split: 70/15/15, stratified jointly by class (POSITIVE/NEGATIVE) and substance (Cocaína/Marihuana/Heroína/Ecstasy).

## Comparative Results

| Model | Technique | Standard Acc. | Irony Acc. | Obfuscated Acc. |
| :--- | :--- | :---: | :---: | :---: |
| **BERT (Base)** | Fine-tuned BETO | **84.22%** | **84.00%** | **83.56%** |
| **TextCNN** | Word2Vec + Conv1D | 80.44% | 81.33% | 78.44% |
| **Logistic Regression** | TF-IDF + LogisticRegression | 80.44% | 80.00% | 80.44% |
| **SVM** | TF-IDF + LinearSVC | 80.44% | 79.78% | 80.00% |
| **BiLSTM** | Word2Vec + BiLSTM | 78.44% | 78.89% | 76.00% |
| **Naive Bayes** | TF-IDF + MultinomialNB | 78.00% | 77.56% | 78.00% |
| **FFN** | Word2Vec + FFN | 76.44% | 77.11% | 75.11% |
| **Random Forest** | TF-IDF + RandomForest | 76.00% | 76.22% | 76.22% |

## Summary

The **BERT (Base)** model achieves the best performance across all three variants of the pre-filtered corpus, reaching **84.22%** (Standard), **84.00%** (Irony), and **83.56%** (Obfuscated) accuracy. Unlike the earlier (non-drug-stratified) split, BERT no longer drops behind Naive Bayes on the Obfuscated variant — it now leads consistently.

**BERT (Base)** is the recommended model for the pre-filtered corpus across all three pipeline variants.
