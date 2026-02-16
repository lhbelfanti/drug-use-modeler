# Models

This directory stores trained models and their associated artifacts, organized by model type and data variation.

## Model Registry

Each model is trained on **two** data pipelines:
- **Standard**: Basic cleaning (lowercase, URL/emoji removal, tag preservation).
- **Irony**: Standard + irony markers (e.g. `ahrre`, `(?`, `xD`) tagged as `[IRONIA]`.

### Traditional ML Models

| Model | Source Notebook | Technique | Standard Acc. | Irony Acc. | Obfuscated Acc. |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Naive Bayes** | [05_naive_bayes.ipynb](../notebooks/05_naive_bayes.ipynb) | TF-IDF + MultinomialNB | 83.78% | 83.33% | 83.11% |
| **Logistic Regression** | [03_logistic_regression.ipynb](../notebooks/03_logistic_regression.ipynb) | TF-IDF + LogisticRegression | 82.00% | 82.00% | 81.78% |
| **SVM** | [04_svm.ipynb](../notebooks/04_svm.ipynb) | TF-IDF + LinearSVC | 81.56% | 81.78% | 81.56% |
| **Random Forest** | [06_random_forest.ipynb](../notebooks/06_random_forest.ipynb) | TF-IDF + RandomForest | 78.44% | 79.78% | 79.56% |

### Deep Learning Models

| Model | Source Notebook | Technique | Standard Acc. | Irony Acc. | Obfuscated Acc. |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **BERT (Base)** | [11_bert_base.ipynb](../notebooks/11_bert_base.ipynb) | Fine-tuned BETO (Spanish BERT) | **86.22%** | **85.33%** | 80.44% |
| **TextCNN** | [09_cnn.ipynb](../notebooks/09_cnn.ipynb) | Word2Vec + Conv1D(3,4,5) | 82.00% | 80.67% | 80.89% |
| **BiLSTM** | [10_rnn.ipynb](../notebooks/10_rnn.ipynb) | Word2Vec + BiLSTM(64) | 78.44% | 78.22% | 79.11% |
| **FFN** | [08_feed_forward.ipynb](../notebooks/08_feed_forward.ipynb) | Word2Vec + FFN | 76.89% | 77.11% | 76.00% |

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
├── rnn/{standard,irony}/
└── bert_base/{standard,irony}/
```
