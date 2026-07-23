# Evaluation Results: Raw Corpus

This directory contains the evaluation results for models trained on the **raw corpus**.

> **Note**: For current dataset statistics, see [data/raw/README.md](../../data/raw/README.md). Split: 70/15/15, stratified jointly by class (POSITIVE/NEGATIVE) and substance (Cocaína/Marihuana/Heroína/Ecstasy).

## Comparative Results

| Model | Technique | Standard Acc. | Irony Acc. | Obfuscated Acc. |
| :--- | :--- | :---: | :---: | :---: |
| **BERT (Base)** | Fine-tuned BETO | **81.33%** | **83.56%** | **82.22%** |
| **SVM** | TF-IDF + LinearSVC | 80.22% | 80.67% | 80.00% |
| **TextCNN** | Word2Vec + Conv1D | 78.22% | 79.11% | 81.11% |
| **Naive Bayes** | TF-IDF + MultinomialNB | 78.89% | 78.67% | 78.00% |
| **Logistic Regression** | TF-IDF + LogisticRegression | 78.67% | 78.67% | 78.44% |
| **BiLSTM** | Word2Vec + BiLSTM | 77.56% | 75.56% | 78.89% |
| **Random Forest** | TF-IDF + RandomForest | 75.56% | 77.33% | 74.89% |
| **FFN** | Word2Vec + FFN | 72.89% | 74.00% | 72.67% |

## Summary

With the drug-stratified split, **BERT (Base)** is now the best model across all three variants of the raw corpus, reaching **81.33%** (Standard), **83.56%** (Irony), and **82.22%** (Obfuscated) accuracy. This reverses the earlier (non-drug-stratified) finding that Naive Bayes outperformed BERT on this corpus — that result was an artifact of a test split that happened to be easier for TF-IDF-based models, not a genuine robustness advantage of Naive Bayes.

**BERT (Base)** is the recommended model for the raw corpus.
