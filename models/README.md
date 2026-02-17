# Models

This directory stores trained models and their associated artifacts, organized by corpus, model type, and data variation.

## Data Pipelines

Each model is trained on **three** data pipelines:
- **Standard**: Basic cleaning (lowercase, URL/emoji removal, tag preservation).
- **Irony**: Standard + irony markers (e.g. `ahrre`, `(?`, `xD`) tagged as `[IRONIA]`.
- **Obfuscated**: Standard + personal names replaced with `[PERSONA]` via spaCy NER.

---

## Results: `pre-filtered-corpus`

> Filtered dataset (2,550 samples: 1,275 per class).

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

---

## Results: `raw-corpus`

> Full unfiltered dataset (2,633 samples).

### Traditional ML Models

| Model | Source Notebook | Technique | Standard Acc. | Irony Acc. | Obfuscated Acc. |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Naive Bayes** | [05_naive_bayes.ipynb](../notebooks/05_naive_bayes.ipynb) | TF-IDF + MultinomialNB | 83.78% | 83.33% | 83.11% |
| **Logistic Regression** | [03_logistic_regression.ipynb](../notebooks/03_logistic_regression.ipynb) | TF-IDF + LogisticRegression | 80.00% | 80.22% | 80.67% |
| **SVM** | [04_svm.ipynb](../notebooks/04_svm.ipynb) | TF-IDF + LinearSVC | 81.56% | 81.78% | 81.56% |
| **Random Forest** | [06_random_forest.ipynb](../notebooks/06_random_forest.ipynb) | TF-IDF + RandomForest | 78.44% | 79.78% | 79.56% |

### Deep Learning Models

| Model | Source Notebook | Technique | Standard Acc. | Irony Acc. | Obfuscated Acc. |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **BERT (Base)** | [11_bert_base.ipynb](../notebooks/11_bert_base.ipynb) | Fine-tuned BETO (Spanish BERT) | 81.78% | 82.44% | 82.00% |
| **BiLSTM** | [10_rnn.ipynb](../notebooks/10_rnn.ipynb) | Word2Vec + BiLSTM(64) | 76.44% | 78.22% | 75.56% |
| **TextCNN** | [09_cnn.ipynb](../notebooks/09_cnn.ipynb) | Word2Vec + Conv1D(3,4,5) | 76.22% | 77.56% | 76.44% |
| **FFN** | [08_feed_forward.ipynb](../notebooks/08_feed_forward.ipynb) | Word2Vec + FFN | 76.00% | 76.89% | 73.33% |

### Embeddings

| Source Notebook | Technique | Details |
| :--- | :--- | :--- |
| [07_word2vec_embeddings.ipynb](../notebooks/07_word2vec_embeddings.ipynb) | Word2Vec (Skip-gram) | 100-dim, window=5 |

---

## Directory Structure
```
models/
├── pre-filtered-corpus/
│   ├── logistic_regression/{standard,irony,obfuscated}/
│   ├── svm/{standard,irony,obfuscated}/
│   ├── naive_bayes/{standard,irony,obfuscated}/
│   ├── random_forest/{standard,irony,obfuscated}/
│   ├── word2vec/{standard,irony,obfuscated}/
│   ├── ffn/{standard,irony,obfuscated}/
│   ├── cnn/{standard,irony,obfuscated}/
│   ├── rnn/{standard,irony,obfuscated}/
│   └── bert_base/{standard,irony,obfuscated}/
└── raw-corpus/
    ├── logistic_regression/{standard,irony,obfuscated}/
    ├── svm/{standard,irony,obfuscated}/
    ├── naive_bayes/{standard,irony,obfuscated}/
    ├── random_forest/{standard,irony,obfuscated}/
    ├── word2vec/{standard,irony,obfuscated}/
    ├── ffn/{standard,irony,obfuscated}/
    ├── cnn/{standard,irony,obfuscated}/
    ├── rnn/{standard,irony,obfuscated}/
    └── bert_base/{standard,irony,obfuscated}/
```
