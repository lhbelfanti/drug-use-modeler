# Models

This directory stores trained models and their associated artifacts (vectorizers, tokenizers, etc.).

## Model Registry

| Model Artifacts | Source Notebook | Technique | Description |
| :--- | :--- | :--- | :--- |
| `logistic_regression_model.joblib`<br>`tfidf_vectorizer_logistic.joblib` | [03_tfidf_logistic_regression.ipynb](../notebooks/03_tfidf_logistic_regression.ipynb) | **TF-IDF + Logistic Regression**<br>*(n-grams=1-2, balanced)* | Baseline model. Uses simple word frequency features to detect drug usage intent (e.g., "quiero", "necesito"). |
