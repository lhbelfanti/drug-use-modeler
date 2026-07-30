# Models

This directory stores trained models and their associated artifacts, organized by corpus, model type, and data variation.

## Data Pipelines

Each model is trained on **three** data pipelines:
- **Standard**: Basic cleaning (lowercase, URL/emoji removal, tag preservation).
- **Irony**: Standard + irony markers (e.g. `ahrre`, `(?`, `xD`) tagged as `[IRONIA]`.
- **Obfuscated**: Standard + personal names replaced with `[PERSONA]` via spaCy NER.

---

## Results: `pre-filtered-corpus`

> **Note**: For current dataset statistics, see [data/raw/README.md](../data/raw/README.md). Split: 70/15/15, stratified jointly by class (POSITIVE/NEGATIVE) and substance (Cocaína/Marihuana/Heroína/Ecstasy), with cross-split text duplicates resolved. BETO's checkpoint is selected on the validation split, and test is evaluated once — see [Iteration 3](../evaluation/README.md#iteration-3-honest-beto-selection--deduplicated-split) in the evaluation log.

### Traditional ML Models

| Model | Source Notebook | Technique | Standard Acc. | Irony Acc. | Obfuscated Acc. |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Logistic Regression** | [03_logistic_regression.ipynb](../notebooks/03_logistic_regression.ipynb) | TF-IDF + LogisticRegression | 80.22% | 80.44% | 78.89% |
| **SVM** | [04_svm.ipynb](../notebooks/04_svm.ipynb) | TF-IDF + LinearSVC | 79.78% | 79.11% | 79.78% |
| **Naive Bayes** | [05_naive_bayes.ipynb](../notebooks/05_naive_bayes.ipynb) | TF-IDF + MultinomialNB | 77.33% | 77.78% | 76.89% |
| **Random Forest** | [06_random_forest.ipynb](../notebooks/06_random_forest.ipynb) | TF-IDF + RandomForest | 76.67% | 79.11% | 76.44% |

### Deep Learning Models

| Model | Source Notebook | Technique | Standard Acc. | Irony Acc. | Obfuscated Acc. |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **BERT (Base)** | [11_bert_base.ipynb](../notebooks/11_bert_base.ipynb) | Fine-tuned BETO (Spanish BERT) | **86.00%** | **85.33%** | **85.78%** |
| **TextCNN** | [09_cnn.ipynb](../notebooks/09_cnn.ipynb) | Word2Vec + Conv1D(3,4,5) | 77.33% | 79.33% | 78.22% |
| **BiLSTM** | [10_rnn.ipynb](../notebooks/10_rnn.ipynb) | Word2Vec + BiLSTM(64) | 76.67% | 78.44% | 75.11% |
| **FFN** | [08_feed_forward.ipynb](../notebooks/08_feed_forward.ipynb) | Word2Vec + FFN | 72.89% | 76.67% | 75.78% |

---

## Results: `raw-corpus`

> **Note**: For current dataset statistics, see [data/raw/README.md](../data/raw/README.md). Split: 70/15/15, stratified jointly by class (POSITIVE/NEGATIVE) and substance (Cocaína/Marihuana/Heroína/Ecstasy), with cross-split text duplicates resolved. BETO's checkpoint is selected on the validation split, and test is evaluated once — see [Iteration 3](../evaluation/README.md#iteration-3-honest-beto-selection--deduplicated-split) in the evaluation log.

### Traditional ML Models

| Model | Source Notebook | Technique | Standard Acc. | Irony Acc. | Obfuscated Acc. |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **SVM** | [04_svm.ipynb](../notebooks/04_svm.ipynb) | TF-IDF + LinearSVC | 80.00% | 80.00% | 80.22% |
| **Naive Bayes** | [05_naive_bayes.ipynb](../notebooks/05_naive_bayes.ipynb) | TF-IDF + MultinomialNB | 79.78% | 79.78% | 80.22% |
| **Logistic Regression** | [03_logistic_regression.ipynb](../notebooks/03_logistic_regression.ipynb) | TF-IDF + LogisticRegression | 80.67% | 80.67% | 81.78% |
| **Random Forest** | [06_random_forest.ipynb](../notebooks/06_random_forest.ipynb) | TF-IDF + RandomForest | 74.44% | 76.00% | 76.22% |

### Deep Learning Models

| Model | Source Notebook | Technique | Standard Acc. | Irony Acc. | Obfuscated Acc. |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **BERT (Base)** | [11_bert_base.ipynb](../notebooks/11_bert_base.ipynb) | Fine-tuned BETO (Spanish BERT) | **85.56%** | **87.56%** | **87.78%** |
| **TextCNN** | [09_cnn.ipynb](../notebooks/09_cnn.ipynb) | Word2Vec + Conv1D(3,4,5) | 79.78% | 79.33% | 80.00% |
| **BiLSTM** | [10_rnn.ipynb](../notebooks/10_rnn.ipynb) | Word2Vec + BiLSTM(64) | 79.11% | 77.56% | 80.22% |
| **FFN** | [08_feed_forward.ipynb](../notebooks/08_feed_forward.ipynb) | Word2Vec + FFN | 78.00% | 74.67% | 77.78% |

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
