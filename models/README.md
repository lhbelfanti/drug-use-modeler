# Models

This directory stores trained models and their associated artifacts (vectorizers, tokenizers, etc.), organized by model type.

## Model Registry

| Model | Source Notebook | Technique | Accuracy | Artifacts Path |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | [03_logistic_regression.ipynb](../notebooks/03_logistic_regression.ipynb) | TF-IDF + Logistic Regression | **82.00%** | `models/logistic_regression/` |
| **SVM** | [04_svm.ipynb](../notebooks/04_svm.ipynb) | TF-IDF + LinearSVC | **81.56%** | `models/svm/` |
| **Naive Bayes** | [05_naive_bayes.ipynb](../notebooks/05_naive_bayes.ipynb) | TF-IDF + MultinomialNB | **83.78%** | `models/naive_bayes/` |

## Directory Structure
```
models/
├── logistic_regression/
│   ├── model.joblib
│   └── vectorizer.joblib
├── svm/
│   ├── model.joblib
│   └── vectorizer.joblib
└── naive_bayes/
    ├── model.joblib
    └── vectorizer.joblib
```
