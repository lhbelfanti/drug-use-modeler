# Models

This directory stores trained models and their associated artifacts, organized by model type and data variation.

## Model Registry

Each model is trained on **two** data pipelines:
- **Standard**: Basic cleaning (lowercase, URL/emoji removal, tag preservation).
- **Irony**: Standard + irony markers (e.g. `ahrre`, `(?`, `xD`) tagged as `[IRONIA]`.

| Model | Source Notebook | Technique | Standard Acc. | Irony Acc. | Delta (Δ) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Naive Bayes** | [05_naive_bayes.ipynb](../notebooks/05_naive_bayes.ipynb) | TF-IDF + MultinomialNB | **83.78%** | 83.33% | -0.45% |
| **Logistic Regression** | [03_logistic_regression.ipynb](../notebooks/03_logistic_regression.ipynb) | TF-IDF + LogisticRegression | 82.00% | 82.00% | 0.00% |
| **SVM** | [04_svm.ipynb](../notebooks/04_svm.ipynb) | TF-IDF + LinearSVC | 81.56% | 81.78% | +0.22% |
| **Random Forest** | [06_random_forest.ipynb](../notebooks/06_random_forest.ipynb) | TF-IDF + RandomForest | 78.44% | 79.78% | +1.34% |

## Directory Structure
```
models/
├── logistic_regression/
│   ├── standard/
│   │   ├── model.joblib
│   │   └── vectorizer.joblib
│   └── irony/
│       ├── model.joblib
│       └── vectorizer.joblib
├── svm/
│   ├── standard/ ...
│   └── irony/ ...
├── naive_bayes/
│   ├── standard/ ...
│   └── irony/ ...
└── random_forest/
    ├── standard/ ...
    └── irony/ ...
```
