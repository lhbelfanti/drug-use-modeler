# Evaluation Results: Pre-filtered Corpus

This directory contains the evaluation results for models trained on the **pre-filtered corpus**.

> **Note**: For current dataset statistics, see [data/raw/README.md](../../data/raw/README.md).

## Comparative Results

| Model | Technique | Standard Acc. | Irony Acc. | Obfuscated Acc. |
| :--- | :--- | :---: | :---: | :---: |
| **BERT (Base)** | Fine-tuned BETO | **86.22%** | **85.33%** | 80.44% |
| **Naive Bayes** | TF-IDF + MultinomialNB | 83.78% | 83.33% | **83.11%** |
| **Logistic Regression** | TF-IDF + LogisticRegression | 82.00% | 82.00% | 81.78% |
| **TextCNN** | Word2Vec + Conv1D | 82.00% | 80.67% | 80.89% |
| **SVM** | TF-IDF + LinearSVC | 81.56% | 81.78% | 81.56% |
| **Random Forest** | TF-IDF + RandomForest | 78.44% | 79.78% | 79.56% |
| **BiLSTM** | Word2Vec + BiLSTM | 78.44% | 78.22% | 79.11% |
| **FFN** | Word2Vec + FFN | 76.89% | 77.11% | 76.00% |

## Summary

The **BERT (Base)** model achieves the best performance on the Standard and Irony datasets, reaching an accuracy of **86.22%** and **85.33%** respectively.

However, for the **Obfuscated** dataset, the **Naive Bayes** model performs best with **83.11%** accuracy, suggesting it might be more robust to the loss of specific named entities or that the BERT model relies more heavily on those entities in this specific context.

Overall, **BERT (Base)** is the recommended model for the pre-filtered corpus if obfuscation is not a primary concern or if the slight drop in obfuscated performance is acceptable.
