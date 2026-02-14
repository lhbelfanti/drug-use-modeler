# Models

This directory stores trained models and their associated artifacts, organized by model type and data variation.

## Model Registry

Each model is trained on **two** data pipelines:
- **Standard**: Basic cleaning (lowercase, URL/emoji removal, tag preservation).
- **Irony**: Standard + irony markers (e.g. `ahrre`, `(?`, `xD`) tagged as `[IRONIA]`.

### Traditional ML Models

| Model | Source Notebook | Technique | Standard Acc. | Irony Acc. | Delta (Δ) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Naive Bayes** | [05_naive_bayes.ipynb](../notebooks/05_naive_bayes.ipynb) | TF-IDF + MultinomialNB | **83.78%** | 83.33% | -0.45% |
| **Logistic Regression** | [03_logistic_regression.ipynb](../notebooks/03_logistic_regression.ipynb) | TF-IDF + LogisticRegression | 82.00% | 82.00% | 0.00% |
| **SVM** | [04_svm.ipynb](../notebooks/04_svm.ipynb) | TF-IDF + LinearSVC | 81.56% | 81.78% | +0.22% |
| **Random Forest** | [06_random_forest.ipynb](../notebooks/06_random_forest.ipynb) | TF-IDF + RandomForest | 78.44% | 79.78% | +1.34% |

### Deep Learning Models

| Model | Source Notebook | Technique | Standard Acc. | Irony Acc. | Delta (Δ) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **TextCNN** | [09_cnn.ipynb](../notebooks/09_cnn.ipynb) | Word2Vec + Conv1D(3,4,5) | **81.33%** | 81.11% | -0.22% |
| **BiLSTM** | [10_rnn.ipynb](../notebooks/10_rnn.ipynb) | Word2Vec + BiLSTM(64) | 78.22% | 78.00% | -0.22% |
| **FFN** | [08_feed_forward.ipynb](../notebooks/08_feed_forward.ipynb) | Word2Vec + FFN | 76.00% | 77.78% | +1.78% |

### Embeddings

| Source Notebook | Technique | Details |
| :--- | :--- | :--- |
| [07_word2vec_embeddings.ipynb](../notebooks/07_word2vec_embeddings.ipynb) | Word2Vec (Skip-gram) | 100-dim, window=5, vocab ~2240 |

## Directory Structure
```
models/
├── logistic_regression/{standard,irony}/
├── svm/{standard,irony}/
├── naive_bayes/{standard,irony}/
├── random_forest/{standard,irony}/
├── word2vec/{standard,irony}/
├── ffn/{standard,irony}/
├── cnn/{standard,irony}/
└── rnn/{standard,irony}/
```
