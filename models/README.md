# Models

This directory stores trained models and their associated artifacts, organized by corpus, model type, and data variation.

## Data Pipelines

Each model is trained on **three** data pipelines:
- **Standard**: Basic cleaning (lowercase, URL/emoji removal, tag preservation).
- **Irony**: Standard + irony markers (e.g. `ahrre`, `(?`, `xD`) tagged as `[IRONIA]`.
- **Obfuscated**: Standard + personal names replaced with `[PERSONA]` via spaCy NER.

---

## Results: `pre-filtered-corpus`

> **Note**: For current dataset statistics, see [data/raw/README.md](../data/raw/README.md). Split: 70/15/15, stratified jointly by class (POSITIVE/NEGATIVE) and substance (Cocaína/Marihuana/Heroína/Ecstasy).

### Traditional ML Models

| Model | Source Notebook | Technique | Standard Acc. | Irony Acc. | Obfuscated Acc. |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Logistic Regression** | [03_logistic_regression.ipynb](../notebooks/03_logistic_regression.ipynb) | TF-IDF + LogisticRegression | 80.44% | 80.00% | 80.44% |
| **SVM** | [04_svm.ipynb](../notebooks/04_svm.ipynb) | TF-IDF + LinearSVC | 80.44% | 79.78% | 80.00% |
| **Naive Bayes** | [05_naive_bayes.ipynb](../notebooks/05_naive_bayes.ipynb) | TF-IDF + MultinomialNB | 78.00% | 77.56% | 78.00% |
| **Random Forest** | [06_random_forest.ipynb](../notebooks/06_random_forest.ipynb) | TF-IDF + RandomForest | 76.00% | 76.22% | 76.22% |

### Deep Learning Models

| Model | Source Notebook | Technique | Standard Acc. | Irony Acc. | Obfuscated Acc. |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **BERT (Base)** | [11_bert_base.ipynb](../notebooks/11_bert_base.ipynb) | Fine-tuned BETO (Spanish BERT) | **84.22%** | **84.00%** | **83.56%** |
| **TextCNN** | [09_cnn.ipynb](../notebooks/09_cnn.ipynb) | Word2Vec + Conv1D(3,4,5) | 80.44% | 81.33% | 78.44% |
| **BiLSTM** | [10_rnn.ipynb](../notebooks/10_rnn.ipynb) | Word2Vec + BiLSTM(64) | 78.44% | 78.89% | 76.00% |
| **FFN** | [08_feed_forward.ipynb](../notebooks/08_feed_forward.ipynb) | Word2Vec + FFN | 76.44% | 77.11% | 75.11% |

---

## Results: `raw-corpus`

> **Note**: For current dataset statistics, see [data/raw/README.md](../data/raw/README.md). Split: 70/15/15, stratified jointly by class (POSITIVE/NEGATIVE) and substance (Cocaína/Marihuana/Heroína/Ecstasy).

### Traditional ML Models

| Model | Source Notebook | Technique | Standard Acc. | Irony Acc. | Obfuscated Acc. |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **SVM** | [04_svm.ipynb](../notebooks/04_svm.ipynb) | TF-IDF + LinearSVC | 80.22% | 80.67% | 80.00% |
| **Naive Bayes** | [05_naive_bayes.ipynb](../notebooks/05_naive_bayes.ipynb) | TF-IDF + MultinomialNB | 78.89% | 78.67% | 78.00% |
| **Logistic Regression** | [03_logistic_regression.ipynb](../notebooks/03_logistic_regression.ipynb) | TF-IDF + LogisticRegression | 78.67% | 78.67% | 78.44% |
| **Random Forest** | [06_random_forest.ipynb](../notebooks/06_random_forest.ipynb) | TF-IDF + RandomForest | 75.56% | 77.33% | 74.89% |

### Deep Learning Models

| Model | Source Notebook | Technique | Standard Acc. | Irony Acc. | Obfuscated Acc. |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **BERT (Base)** | [11_bert_base.ipynb](../notebooks/11_bert_base.ipynb) | Fine-tuned BETO (Spanish BERT) | **81.33%** | **83.56%** | **82.22%** |
| **TextCNN** | [09_cnn.ipynb](../notebooks/09_cnn.ipynb) | Word2Vec + Conv1D(3,4,5) | 78.22% | 79.11% | 81.11% |
| **BiLSTM** | [10_rnn.ipynb](../notebooks/10_rnn.ipynb) | Word2Vec + BiLSTM(64) | 77.56% | 75.56% | 78.89% |
| **FFN** | [08_feed_forward.ipynb](../notebooks/08_feed_forward.ipynb) | Word2Vec + FFN | 72.89% | 74.00% | 72.67% |

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
