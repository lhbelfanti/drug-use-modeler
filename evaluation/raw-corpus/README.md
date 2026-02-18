# Evaluation Results: Raw Corpus

This directory contains the evaluation results for models trained on the **raw corpus**.

> **Note**: For current dataset statistics, see [data/raw/README.md](../../data/raw/README.md). The results below are based on the dataset version available at the time of training (~2,633 samples).

## Comparative Results

| Model | Technique | Standard Acc. | Irony Acc. | Obfuscated Acc. |
| :--- | :--- | :---: | :---: | :---: |
| **Naive Bayes** | TF-IDF + MultinomialNB | **83.78%** | **83.33%** | **83.11%** |
| **BERT (Base)** | Fine-tuned BETO | 81.78% | 82.44% | 82.00% |
| **SVM** | TF-IDF + LinearSVC | 81.56% | 81.78% | 81.56% |
| **Logistic Regression** | TF-IDF + LogisticRegression | 80.00% | 80.22% | 80.67% |
| **Random Forest** | TF-IDF + RandomForest | 78.44% | 79.78% | 79.56% |
| **BiLSTM** | Word2Vec + BiLSTM | 76.44% | 78.22% | 75.56% |
| **TextCNN** | Word2Vec + Conv1D | 76.22% | 77.56% | 76.44% |
| **FFN** | Word2Vec + FFN | 76.00% | 76.89% | 73.33% |

## Summary

Surprisingly, the **Naive Bayes** model outperforms all other models, including BERT, across all three data variations (Standard, Irony, Obfuscated) on the raw corpus. It achieves consistent accuracy around **83%**.

**BERT (Base)** follows closely behind but does not surpass the simpler Naive Bayes baseline in this specific setting. This could indicate that the noise in the raw corpus affects the deep learning model more than the probabilistic Naive Bayes, or that the dataset size is small enough that the simpler model generalizes better.

**Naive Bayes** is the recommended model for the raw corpus due to its superior performance and lower computational cost.
