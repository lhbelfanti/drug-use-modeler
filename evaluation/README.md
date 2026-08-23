# Evaluation Log

This document serves as a chronological log of model evaluations, tracking performance improvements and experiments over time.

## Iteration 1: Initial Model Baseline

**Goal**: Establish baseline performance for standard, irony-augmented, and obfuscated datasets.

### Dataset Details

-   **Total Samples**: 3,000 for each corpus (3,000 Raw Corpus / 3,000 Pre-filtered Corpus).
-   **Full Statistics**: See **[data/raw/README.md](../data/raw/README.md)**.

### Metrics Reference

Full results — including per-class Precision, Recall, and F1-Score for NEGATIVE and POSITIVE — are available in machine-readable form:

-   **[evaluation/metrics.json](./metrics.json)**: Structured JSON with macro-averaged and per-class metrics.
-   **[evaluation/metrics.csv](./metrics.csv)**: Flat table (corpus / variant / model / accuracy / precision / recall / f1) for easy filtering and comparison.

All metrics are computed by running inference with the saved model artifacts on each `test.csv` split. For raw-corpus Naive Bayes, SVM, and Random Forest (no saved joblib), models are re-trained inline with identical hyperparameters (seed=42) using **[scripts/evaluate_all.py](../scripts/evaluate_all.py)**.

> **Note**: The raw-corpus NB/SVM/RF results in the sub-README were incorrectly copied from the pre-filtered corpus. The values below reflect the correct re-evaluated numbers.

---

### Detailed Reports

-   **[Pre-filtered Corpus Results](./pre-filtered-corpus/README.md)**: Models trained on the clean, filtered dataset.
-   **[Raw Corpus Results](./raw-corpus/README.md)**: Models trained on the full, unfiltered dataset.

---

### Full Results: Pre-filtered Corpus

All metrics are macro-averaged. Sorted by F1-Score descending.

#### Standard Variant

| Model | Technique | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BERT (Base)** | Fine-tuned BETO | **86.22%** | **86.25%** | **86.22%** | **86.22%** |
| **Naive Bayes** | TF-IDF + MultinomialNB | 83.78% | 83.79% | 83.78% | 83.78% |
| **Logistic Regression** | TF-IDF + LogisticRegression | 82.00% | 82.14% | 82.00% | 81.98% |
| **CNN** | Word2Vec + Conv1D | 82.00% | 82.14% | 82.00% | 81.98% |
| **SVM** | TF-IDF + LinearSVC | 81.56% | 81.63% | 81.56% | 81.54% |
| **Random Forest** | TF-IDF + RandomForest | 78.44% | 78.54% | 78.44% | 78.43% |
| **BiLSTM** | Word2Vec + BiLSTM | 78.44% | 78.47% | 78.44% | 78.44% |
| **FFN** | Word2Vec + FFN | 76.89% | 76.92% | 76.89% | 76.88% |

#### Irony Variant

| Model | Technique | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BERT (Base)** | Fine-tuned BETO | **85.33%** | **85.34%** | **85.33%** | **85.33%** |
| **Naive Bayes** | TF-IDF + MultinomialNB | 83.33% | 83.35% | 83.33% | 83.33% |
| **Logistic Regression** | TF-IDF + LogisticRegression | 82.00% | 82.14% | 82.00% | 81.98% |
| **SVM** | TF-IDF + LinearSVC | 81.78% | 81.84% | 81.78% | 81.77% |
| **CNN** | Word2Vec + Conv1D | 80.67% | 81.05% | 80.67% | 80.61% |
| **Random Forest** | TF-IDF + RandomForest | 79.78% | 79.99% | 79.78% | 79.74% |
| **BiLSTM** | Word2Vec + BiLSTM | 78.22% | 78.33% | 78.22% | 78.20% |
| **FFN** | Word2Vec + FFN | 77.11% | 77.14% | 77.11% | 77.11% |

#### Obfuscated Variant

| Model | Technique | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Naive Bayes** | TF-IDF + MultinomialNB | **83.11%** | **83.13%** | **83.11%** | **83.11%** |
| **Logistic Regression** | TF-IDF + LogisticRegression | 81.78% | 81.94% | 81.78% | 81.75% |
| **SVM** | TF-IDF + LinearSVC | 81.56% | 81.61% | 81.56% | 81.55% |
| **BERT (Base)** | Fine-tuned BETO | 80.44% | 80.50% | 80.44% | 80.43% |
| **CNN** | Word2Vec + Conv1D | 80.89% | 81.09% | 80.89% | 80.86% |
| **Random Forest** | TF-IDF + RandomForest | 79.56% | 79.67% | 79.56% | 79.54% |
| **BiLSTM** | Word2Vec + BiLSTM | 79.11% | 79.34% | 79.11% | 79.07% |
| **FFN** | Word2Vec + FFN | 76.00% | 76.10% | 76.00% | 75.98% |

---

### Full Results: Raw Corpus

All metrics are macro-averaged. Sorted by F1-Score descending.

#### Standard Variant

| Model | Technique | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **SVM** | TF-IDF + LinearSVC | **82.22%** | **82.48%** | **82.22%** | **82.19%** |
| **BERT (Base)** | Fine-tuned BETO | 81.78% | 81.78% | 81.78% | 81.78% |
| **Naive Bayes** | TF-IDF + MultinomialNB | 80.67% | 80.74% | 80.67% | 80.66% |
| **Logistic Regression** | TF-IDF + LogisticRegression | 80.00% | 80.35% | 80.00% | 79.94% |
| **Random Forest** | TF-IDF + RandomForest | 76.89% | 76.97% | 76.89% | 76.87% |
| **BiLSTM** | Word2Vec + BiLSTM | 76.44% | 77.06% | 76.44% | 76.31% |
| **CNN** | Word2Vec + Conv1D | 76.22% | 76.95% | 76.22% | 76.06% |
| **FFN** | Word2Vec + FFN | 76.00% | 76.54% | 76.00% | 75.88% |

#### Irony Variant

| Model | Technique | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BERT (Base)** | Fine-tuned BETO | **82.44%** | **82.45%** | **82.44%** | **82.44%** |
| **SVM** | TF-IDF + LinearSVC | 82.44% | 82.63% | 82.44% | 82.42% |
| **Naive Bayes** | TF-IDF + MultinomialNB | 80.44% | 80.48% | 80.44% | 80.44% |
| **Logistic Regression** | TF-IDF + LogisticRegression | 80.22% | 80.54% | 80.22% | 80.17% |
| **BiLSTM** | Word2Vec + BiLSTM | 78.22% | 78.22% | 78.22% | 78.22% |
| **CNN** | Word2Vec + Conv1D | 77.56% | 78.09% | 77.56% | 77.45% |
| **FFN** | Word2Vec + FFN | 76.89% | 77.15% | 76.89% | 76.83% |
| **Random Forest** | TF-IDF + RandomForest | 76.67% | 76.82% | 76.67% | 76.63% |

#### Obfuscated Variant

| Model | Technique | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **SVM** | TF-IDF + LinearSVC | **83.11%** | **83.32%** | **83.11%** | **83.08%** |
| **BERT (Base)** | Fine-tuned BETO | 82.00% | 82.02% | 82.00% | 82.00% |
| **Naive Bayes** | TF-IDF + MultinomialNB | 81.11% | 81.22% | 81.11% | 81.10% |
| **Logistic Regression** | TF-IDF + LogisticRegression | 80.67% | 81.05% | 80.67% | 80.61% |
| **Random Forest** | TF-IDF + RandomForest | 77.78% | 78.05% | 77.78% | 77.72% |
| **CNN** | Word2Vec + Conv1D | 76.44% | 76.75% | 76.44% | 76.38% |
| **BiLSTM** | Word2Vec + BiLSTM | 75.56% | 76.15% | 75.56% | 75.42% |
| **FFN** | Word2Vec + FFN | 73.33% | 74.65% | 73.33% | 72.97% |

---

### Summary & Best Models

#### Best Model Rankings

| Corpus | Best Model | Accuracy | Precision | Recall | F1 | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Pre-filtered** | **BERT (Base)** — Standard | **86.22%** | **86.25%** | **86.22%** | **86.22%** | Best across all variants. Fine-tuning shines on cleaner data. |
| **Raw** | **SVM** — Obfuscated | **83.11%** | **83.32%** | **83.11%** | **83.08%** | Consistently strong across all raw-corpus variants. |

#### Key Findings

1.  **Data Quality Matters for Deep Learning**: BERT's performance drops significantly (from ~86% to ~82%) when moving from the pre-filtered to the raw corpus. This suggests that the noise in the raw corpus negatively impacts the model's ability to learn effectively. Simple TF-IDF models are far less sensitive to this.

2.  **SVM is the Strongest Baseline on Raw Data**: Contrary to the initial assumption, SVM (not Naive Bayes) achieves the highest F1 across all raw-corpus variants once metrics are computed correctly. It even matches BERT on the irony variant (82.42% vs 82.44% F1) while being orders of magnitude cheaper to run.

3.  **Precision ≈ Recall for Most Models**: The balanced class distribution (225 NEGATIVE / 225 POSITIVE in each test split) means macro-averaged Precision and Recall track each other closely. The few exceptions — notably FFN on obfuscated data — reveal models that default toward one class.

4.  **Obfuscation Hurts BERT Most**: BERT drops ~5.8 F1 points from Standard to Obfuscated on the pre-filtered corpus (86.22% → 80.43%), suggesting it relies on named entities that obfuscation removes. Naive Bayes and SVM are far more robust to this transformation.

5.  **Irony Handling**: Most models score comparably or slightly higher on the Irony variant than on Standard. The `[IRONIA]` tag appears to act as a strong, learnable feature for TF-IDF-based models in particular.

6.  **Word2Vec models (FFN, CNN, BiLSTM) underperform consistently**: All three trail TF-IDF-based models across both corpora and all variants, suggesting that the pre-trained word vectors (100-dimensional, trained on this small dataset) do not capture enough signal. Better embeddings or larger training data would likely close the gap.

---

### Next Steps

Based on these results, the following actions are recommended:

1.  **Error Analysis**:
    - Investigate *why* BERT fails on the raw corpus. Are there specific types of tweets that confuse it?
    - Analyze cases where SVM and BERT disagree on raw-corpus examples — they appear to make different errors and could be complementary.
2.  **Data Augmentation**:
    - Explore if data augmentation techniques can help BERT generalize better on the raw corpus and recover from obfuscation.
3.  **Ensemble Methods**:
    - Consider an ensemble of BERT and SVM, as they appear to be the strongest complementary pair across corpora.
4.  **Better Embeddings for Neural Models**:
    - Replace the small in-domain Word2Vec vectors with pre-trained multilingual embeddings (e.g., fastText) to close the gap between FFN/CNN/BiLSTM and TF-IDF baselines.

---

## Iteration 2: Drug-Stratified Split

**Goal**: Iteration 1's train/val/test split (`train_test_split(..., stratify=label)`) was stratified only by class (POSITIVE/NEGATIVE), not by the substance each tweet refers to (Cocaína/Marihuana/Heroína/Ecstasy). `corpus-creator` was updated to persist `search_criteria_id` on each corpus row, which let the split be redone stratified jointly by class **and** substance (70/15/15, same random seed). All models below were re-trained and re-evaluated from scratch on the new split.

### Dataset Details

- **Total Samples**: 3,000 for each corpus, same composition as Iteration 1 (500 pairs Cocaína, 500 pairs Marihuana, 376 pairs Heroína, 124 pairs Ecstasy).
- **Test split**: 450 samples (225 POSITIVE / 225 NEGATIVE), proportionally balanced by substance within each class.

### Full Results: Pre-filtered Corpus

All metrics are macro-averaged. Sorted by F1-Score descending.

#### Standard Variant

| Model | Technique | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BERT (Base)** | Fine-tuned BETO | **84.22%** | **84.52%** | **84.22%** | **84.19%** |
| **Logistic Regression** | TF-IDF + LogisticRegression | 80.44% | 80.53% | 80.44% | 80.43% |
| **SVM** | TF-IDF + LinearSVC | 80.44% | 80.50% | 80.44% | 80.43% |
| **CNN** | Word2Vec + Conv1D | 80.44% | 81.65% | 80.44% | 80.26% |
| **RNN (BiLSTM)** | Word2Vec + BiLSTM | 78.44% | 78.74% | 78.44% | 78.39% |
| **Naive Bayes** | TF-IDF + MultinomialNB | 78.00% | 78.12% | 78.00% | 77.98% |
| **FFN** | Word2Vec + FFN | 76.44% | 76.55% | 76.44% | 76.42% |
| **Random Forest** | TF-IDF + RandomForest | 76.00% | 76.03% | 76.00% | 75.99% |

#### Irony Variant

| Model | Technique | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BERT (Base)** | Fine-tuned BETO | **84.00%** | **84.27%** | **84.00%** | **83.97%** |
| **CNN** | Word2Vec + Conv1D | 81.33% | 82.36% | 81.33% | 81.18% |
| **Logistic Regression** | TF-IDF + LogisticRegression | 80.00% | 80.09% | 80.00% | 79.99% |
| **SVM** | TF-IDF + LinearSVC | 79.78% | 79.81% | 79.78% | 79.77% |
| **RNN (BiLSTM)** | Word2Vec + BiLSTM | 78.89% | 78.99% | 78.89% | 78.87% |
| **Naive Bayes** | TF-IDF + MultinomialNB | 77.56% | 77.65% | 77.56% | 77.54% |
| **FFN** | Word2Vec + FFN | 77.11% | 77.45% | 77.11% | 77.04% |
| **Random Forest** | TF-IDF + RandomForest | 76.22% | 76.26% | 76.22% | 76.21% |

#### Obfuscated Variant

| Model | Technique | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BERT (Base)** | Fine-tuned BETO | **83.56%** | **83.77%** | **83.56%** | **83.53%** |
| **Logistic Regression** | TF-IDF + LogisticRegression | 80.44% | 80.53% | 80.44% | 80.43% |
| **SVM** | TF-IDF + LinearSVC | 80.00% | 80.00% | 80.00% | 80.00% |
| **CNN** | Word2Vec + Conv1D | 78.44% | 78.74% | 78.44% | 78.39% |
| **Naive Bayes** | TF-IDF + MultinomialNB | 78.00% | 78.09% | 78.00% | 77.98% |
| **Random Forest** | TF-IDF + RandomForest | 76.22% | 76.31% | 76.22% | 76.20% |
| **RNN (BiLSTM)** | Word2Vec + BiLSTM | 76.00% | 76.21% | 76.00% | 75.95% |
| **FFN** | Word2Vec + FFN | 75.11% | 75.93% | 75.11% | 74.91% |

### Full Results: Raw Corpus

All metrics are macro-averaged. Sorted by F1-Score descending.

#### Standard Variant

| Model | Technique | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BERT (Base)** | Fine-tuned BETO | **81.33%** | **81.42%** | **81.33%** | **81.32%** |
| **SVM** | TF-IDF + LinearSVC | 80.22% | 80.44% | 80.22% | 80.19% |
| **Naive Bayes** | TF-IDF + MultinomialNB | 78.89% | 78.89% | 78.89% | 78.89% |
| **Logistic Regression** | TF-IDF + LogisticRegression | 78.67% | 78.89% | 78.67% | 78.62% |
| **CNN** | Word2Vec + Conv1D | 78.22% | 78.67% | 78.22% | 78.14% |
| **RNN (BiLSTM)** | Word2Vec + BiLSTM | 77.56% | 77.56% | 77.56% | 77.56% |
| **Random Forest** | TF-IDF + RandomForest | 75.56% | 75.57% | 75.56% | 75.55% |
| **FFN** | Word2Vec + FFN | 72.89% | 73.15% | 72.89% | 72.81% |

#### Irony Variant

| Model | Technique | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BERT (Base)** | Fine-tuned BETO | **83.56%** | **83.73%** | **83.56%** | **83.53%** |
| **SVM** | TF-IDF + LinearSVC | 80.67% | 80.89% | 80.67% | 80.63% |
| **CNN** | Word2Vec + Conv1D | 79.11% | 79.79% | 79.11% | 78.99% |
| **Naive Bayes** | TF-IDF + MultinomialNB | 78.67% | 78.67% | 78.67% | 78.67% |
| **Logistic Regression** | TF-IDF + LogisticRegression | 78.67% | 78.85% | 78.67% | 78.63% |
| **Random Forest** | TF-IDF + RandomForest | 77.33% | 77.41% | 77.33% | 77.32% |
| **RNN (BiLSTM)** | Word2Vec + BiLSTM | 75.56% | 75.96% | 75.56% | 75.46% |
| **FFN** | Word2Vec + FFN | 74.00% | 74.00% | 74.00% | 74.00% |

#### Obfuscated Variant

| Model | Technique | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BERT (Base)** | Fine-tuned BETO | **82.22%** | **82.39%** | **82.22%** | **82.20%** |
| **CNN** | Word2Vec + Conv1D | 81.11% | 81.16% | 81.11% | 81.10% |
| **SVM** | TF-IDF + LinearSVC | 80.00% | 80.24% | 80.00% | 79.96% |
| **RNN (BiLSTM)** | Word2Vec + BiLSTM | 78.89% | 79.31% | 78.89% | 78.81% |
| **Logistic Regression** | TF-IDF + LogisticRegression | 78.44% | 78.65% | 78.44% | 78.41% |
| **Naive Bayes** | TF-IDF + MultinomialNB | 78.00% | 78.00% | 78.00% | 78.00% |
| **Random Forest** | TF-IDF + RandomForest | 74.89% | 75.03% | 74.89% | 74.85% |
| **FFN** | Word2Vec + FFN | 72.67% | 72.70% | 72.67% | 72.66% |

### Best Model Rankings

| Corpus | Best Model | Accuracy | Precision | Recall | F1 | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Pre-filtered** | **BERT (Base)** — Standard | **84.22%** | **84.52%** | **84.22%** | **84.19%** | Best across all variants. |
| **Raw** | **BERT (Base)** — Irony | **83.56%** | **83.73%** | **83.56%** | **83.53%** | Best across all variants. |

### Key Findings (superseding Iteration 1)

1.  **BERT wins everywhere now**: With the drug-stratified split, BERT (Base) is the top model in **all six** corpus/variant combinations — it no longer loses to SVM or Naive Bayes on the raw corpus or on the Obfuscated variant. The Iteration 1 finding that "SVM is the strongest baseline on raw data" and "Obfuscation hurts BERT most" were both artifacts of a test split that, by chance, was easier for TF-IDF-based models — not genuine robustness advantages of those models.

2.  **Smaller, more honest numbers**: Best accuracy dropped from 86.22% (Iteration 1, pre-filtered/standard) to 84.22% under the same conditions. This is expected: Iteration 1's split could leave an easier substance mix in test than in train purely by chance; stratifying by substance removes that source of overestimation.

3.  **Obfuscation is much less costly for BERT now**: BERT drops only ~0.7 F1 points from Standard to Obfuscated on the pre-filtered corpus (84.19% → 83.53%), versus ~5.8 points in Iteration 1. The earlier "BERT relies heavily on named entities" narrative does not hold up once the split is properly stratified.

4.  **Word2Vec models still underperform**: FFN, CNN and BiLSTM continue to trail BERT and TF-IDF baselines, consistent with Iteration 1.

### External LLM Benchmark (GPT-5.4-mini / Gemini 3.1 Pro) — superseded, see Iteration 3

> This benchmark was run against the Iteration 2 test split, which had cross-split text duplicates. It is kept here for history; see [Iteration 3](#iteration-3-honest-beto-selection--deduplicated-split) for the benchmark re-run on the corrected split.

The external LLM benchmark (see [`ahbgpt`](https://github.com/lhbelfanti/ahbgpt), `FINAL_RESULTS.md`) was also re-run on the new drug-stratified test set. Best results per model, pre-filtered corpus:

| Model | Accuracy | Notes |
| :--- | :---: | :--- |
| **Gemini 3.1 Pro** | **93.3%** | Best prompt version (v4/v5 tied). Now clearly ahead of the specialized model. |
| **BETO (this project)** | 84.22% | Best across all local models. |
| **GPT-5.4-mini** | 81.1% | Best prompt version (v2). |

Unlike Iteration 1's benchmark (where BETO and GPT-5.4-mini were near-tied, ~86% vs ~84%), Gemini 3.1 Pro now leads by a wide margin (~9 points over BETO) on the corrected, drug-stratified test set.

---

## Iteration 3: Honest BETO Selection + Deduplicated Split

**Goal**: Iteration 2 had two remaining methodological issues, both found during a pre-submission review:

1.  **BETO's checkpoint was selected using the test set.** `11_bert_base.ipynb` / `scripts/train_bert_base.py` passed `eval_dataset=test_ds` to the HuggingFace `Trainer`, so `load_best_model_at_end` picked whichever epoch scored highest **on the test set itself** — an optimistic, non-blind estimate that none of the other 7 models received. The fix: the `Trainer` now evaluates against `val.csv` (previously loaded and unused) at the end of each epoch, and test is scored exactly once, after training, with no selection involved. Training also now runs the full 3 epochs (the script had regressed to `EPOCHS=1` at some point, undocumented).
2.  **The pre-filtered corpus had a text-corruption bug, and the split had cross-split duplicates.** `corpus-creator`'s cleaning rules (`\bm\b -> me`, `\bd\b -> de`, `\bt\b -> te`, `\bNas\b -> Unas`, ...) relied on Go's RE2 `\b`, which is ASCII-only and doesn't treat accented vowels as word characters — so `más` became `meás`, `día` became `deía`, `escenas` became `escéUnas`, etc. (352/3000 rows affected in the pre-filtered corpus). Fixed at the source in `corpus-creator` (`cmd/api/corpus/cleaner/clean.go`), which now maps accented characters to reversible ASCII placeholders before applying any rule. Separately, ~30 tweets (independently posted by different users with identical text, e.g. "quiero inyectarme heroína") were landing in more than one split, letting a model partially memorize a "held out" text. `02_preprocessing.ipynb` now calls `resolve_cross_split_duplicates` after the split, which consolidates every duplicate-text group into a single split via same-label/same-drug swaps — split sizes and the class/substance balance are unchanged.

### Dataset Details

Unchanged from Iteration 2: 3,000 samples per corpus, 70/15/15 split (2,100/450/450), test set 225 POSITIVE / 225 NEGATIVE, proportionally balanced by substance (Cocaína 150, Marihuana 150, Heroína 112, Ecstasy 38 — i.e. 75/75/56/19 pairs). The corpus composition itself did not change — only *which* tweets land in which split, and the pre-filtered corpus's text content (bug fix).

### Full Results: Pre-filtered Corpus

All metrics are macro-averaged. Sorted by F1-Score descending.

#### Standard Variant

| Model | Technique | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BERT (Base)** | Fine-tuned BETO | **86.00%** | **86.01%** | **86.00%** | **86.00%** |
| **Logistic Regression** | TF-IDF + LogisticRegression | 80.22% | 80.29% | 80.22% | 80.21% |
| **SVM** | TF-IDF + LinearSVC | 79.78% | 79.81% | 79.78% | 79.77% |
| **Naive Bayes** | TF-IDF + MultinomialNB | 77.33% | 77.39% | 77.33% | 77.32% |
| **CNN** | Word2Vec + Conv1D | 77.33% | 77.47% | 77.33% | 77.30% |
| **Random Forest** | TF-IDF + RandomForest | 76.67% | 76.67% | 76.67% | 76.67% |
| **RNN (BiLSTM)** | Word2Vec + BiLSTM | 76.67% | 76.71% | 76.67% | 76.66% |
| **FFN** | Word2Vec + FFN | 72.89% | 72.98% | 72.89% | 72.86% |

#### Irony Variant

| Model | Technique | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BERT (Base)** | Fine-tuned BETO | **85.33%** | **85.34%** | **85.33%** | **85.33%** |
| **Logistic Regression** | TF-IDF + LogisticRegression | 80.44% | 80.50% | 80.44% | 80.43% |
| **CNN** | Word2Vec + Conv1D | 79.33% | 79.40% | 79.33% | 79.32% |
| **SVM** | TF-IDF + LinearSVC | 79.11% | 79.12% | 79.11% | 79.11% |
| **Random Forest** | TF-IDF + RandomForest | 79.11% | 79.12% | 79.11% | 79.11% |
| **RNN (BiLSTM)** | Word2Vec + BiLSTM | 78.44% | 78.51% | 78.44% | 78.43% |
| **Naive Bayes** | TF-IDF + MultinomialNB | 77.78% | 77.83% | 77.78% | 77.77% |
| **FFN** | Word2Vec + FFN | 76.67% | 76.68% | 76.67% | 76.66% |

#### Obfuscated Variant

| Model | Technique | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BERT (Base)** | Fine-tuned BETO | **85.78%** | **85.80%** | **85.78%** | **85.78%** |
| **SVM** | TF-IDF + LinearSVC | 79.78% | 79.81% | 79.78% | 79.77% |
| **Logistic Regression** | TF-IDF + LogisticRegression | 78.89% | 79.02% | 78.89% | 78.87% |
| **CNN** | Word2Vec + Conv1D | 78.22% | 78.28% | 78.22% | 78.21% |
| **Naive Bayes** | TF-IDF + MultinomialNB | 76.89% | 76.92% | 76.89% | 76.88% |
| **Random Forest** | TF-IDF + RandomForest | 76.44% | 76.52% | 76.44% | 76.43% |
| **FFN** | Word2Vec + FFN | 75.78% | 75.78% | 75.78% | 75.78% |
| **RNN (BiLSTM)** | Word2Vec + BiLSTM | 75.11% | 75.14% | 75.11% | 75.10% |

### Full Results: Raw Corpus

All metrics are macro-averaged. Sorted by F1-Score descending.

#### Standard Variant

| Model | Technique | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BERT (Base)** | Fine-tuned BETO | **85.56%** | **85.71%** | **85.56%** | **85.54%** |
| **Logistic Regression** | TF-IDF + LogisticRegression | 80.67% | 80.74% | 80.67% | 80.66% |
| **SVM** | TF-IDF + LinearSVC | 80.00% | 80.01% | 80.00% | 80.00% |
| **Naive Bayes** | TF-IDF + MultinomialNB | 79.78% | 79.83% | 79.78% | 79.77% |
| **CNN** | Word2Vec + Conv1D | 79.78% | 79.99% | 79.78% | 79.74% |
| **RNN (BiLSTM)** | Word2Vec + BiLSTM | 79.11% | 79.51% | 79.11% | 79.04% |
| **FFN** | Word2Vec + FFN | 78.00% | 78.16% | 78.00% | 77.97% |
| **Random Forest** | TF-IDF + RandomForest | 74.44% | 74.50% | 74.44% | 74.43% |

#### Irony Variant

| Model | Technique | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BERT (Base)** | Fine-tuned BETO | **87.56%** | **87.60%** | **87.56%** | **87.55%** |
| **Logistic Regression** | TF-IDF + LogisticRegression | 80.67% | 80.77% | 80.67% | 80.65% |
| **SVM** | TF-IDF + LinearSVC | 80.00% | 80.01% | 80.00% | 80.00% |
| **Naive Bayes** | TF-IDF + MultinomialNB | 79.78% | 79.83% | 79.78% | 79.77% |
| **CNN** | Word2Vec + Conv1D | 79.33% | 79.50% | 79.33% | 79.30% |
| **RNN (BiLSTM)** | Word2Vec + BiLSTM | 77.56% | 77.96% | 77.56% | 77.47% |
| **Random Forest** | TF-IDF + RandomForest | 76.00% | 76.02% | 76.00% | 76.00% |
| **FFN** | Word2Vec + FFN | 74.67% | 76.30% | 74.67% | 74.27% |

#### Obfuscated Variant

| Model | Technique | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BERT (Base)** | Fine-tuned BETO | **87.78%** | **87.80%** | **87.78%** | **87.78%** |
| **Logistic Regression** | TF-IDF + LogisticRegression | 81.78% | 81.84% | 81.78% | 81.77% |
| **SVM** | TF-IDF + LinearSVC | 80.22% | 80.24% | 80.22% | 80.22% |
| **Naive Bayes** | TF-IDF + MultinomialNB | 80.22% | 80.25% | 80.22% | 80.22% |
| **RNN (BiLSTM)** | Word2Vec + BiLSTM | 80.22% | 80.29% | 80.22% | 80.21% |
| **CNN** | Word2Vec + Conv1D | 80.00% | 80.01% | 80.00% | 80.00% |
| **FFN** | Word2Vec + FFN | 77.78% | 77.80% | 77.78% | 77.77% |
| **Random Forest** | TF-IDF + RandomForest | 76.22% | 76.37% | 76.22% | 76.19% |

### Best Model Rankings

| Corpus | Best Model | Accuracy | Precision | Recall | F1 | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Pre-filtered** | **BERT (Base)** — Standard | **86.00%** | **86.01%** | **86.00%** | **86.00%** | Best across all variants. |
| **Raw** | **BERT (Base)** — Obfuscated | **87.78%** | **87.80%** | **87.78%** | **87.78%** | Best across all variants, and the best result overall. |

### Key Findings (superseding Iteration 2)

1.  **All BETO numbers went up, not down.** Fixing the test-set-selection leak removed an optimistic bias, but training the full 3 epochs (instead of the 1 epoch the script had regressed to) more than compensated: every one of the 6 combinations improved over Iteration 2, by 1.5 to 5.6 points of accuracy.

2.  **The raw corpus now beats the pre-filtered corpus for BETO** (87.78% vs. 86.00%, both Obfuscated/Standard respectively) — the reverse of Iteration 2, where pre-filtered was best across the board. This also reverses the Iteration 2 narrative that obfuscation was neutral-to-harmful for BETO: on the raw corpus, **Obfuscated is now BETO's best variant**, beating Standard by 2.22 points.

3.  **Classical/DL models moved by a few points in either direction**, consistent with resolving ~30 cross-split duplicates (some models lost a small memorization edge; others gained slightly different train data). No model's ranking relative to the others changed.

4.  **Word2Vec models still underperform.** FFN, CNN and BiLSTM continue to trail BERT and TF-IDF baselines, consistent with Iterations 1 and 2.

### External LLM Benchmark (GPT-5.4-mini / Gemini 3.1 Pro)

Re-run against this corrected, deduplicated test set, and with a corrected protocol — see [`ahbgpt`](https://github.com/lhbelfanti/ahbgpt) `FINAL_RESULTS.md` for the full breakdown (confusion matrices, classification reports per prompt version).

**Protocol change from Iteration 2**: both models are now called **once per tweet** (450 independent API calls), instead of sending all 450 tweets in a single prompt. This removes the cross-tweet context leakage of the old protocol (e.g. the model inferring the ~50/50 class balance from seeing the whole test set at once) and makes the LLM evaluation directly comparable to how BETO and the classical/DL models are evaluated. Gemini 3.1 Pro is now called via its own API (Batch mode, `gemini-3.1-pro-preview`, `thinking_level=low`) instead of manual copy-paste into the web UI.

| Corpus | Model | Best prompt | Accuracy | F1 (macro) |
| :--- | :--- | :---: | :---: | :---: |
| Pre-filtered | **Gemini 3.1 Pro** | V1 | **93.11%** | **93.09%** |
| Pre-filtered | GPT-5.4-mini | V1 | 85.33% | 85.18% |
| Raw | **Gemini 3.1 Pro** | V1 | **92.22%** | **92.19%** |
| Raw | GPT-5.4-mini | V1 | 85.11% | 84.95% |

For reference, BETO (this project) peaks at 86.00% (pre-filtered/standard) and 87.78% (raw/obfuscated) — Gemini 3.1 Pro still leads by a wide margin (~6–7 points), while GPT-5.4-mini is now essentially tied with BETO.

**Prompt version V1 — the simplest of the five, with no edge-case handling — was the best prompt for both models on both corpora.** This is a different (and more reassuring) picture than Iteration 2, where the best prompt varied by model/corpus and required cherry-picking. The accuracy still varies meaningfully across prompt versions, which is disclosed rather than hidden:

| Corpus | Model | Min | Max | Range |
| :--- | :--- | :---: | :---: | :---: |
| Pre-filtered | Gemini 3.1 Pro | 86.22% | 93.11% | 6.89 pp |
| Pre-filtered | GPT-5.4-mini | 73.11% | 85.33% | 12.22 pp |
| Raw | Gemini 3.1 Pro | 86.00% | 92.22% | 6.22 pp |
| Raw | GPT-5.4-mini | 75.50% | 85.11% | 9.61 pp |

---

## Iteration 4: Multi-Seed Variance + FFN/CNN/RNN Reproducibility Fix

**Goal**: address two issues raised in a pre-submission review by the thesis director (Claudio):

1. All Iteration 1-3 numbers come from a single training run per configuration, with no measure of run-to-run variance — conclusions were being drawn from differences of 1-2 tweets.
2. `gpt-5.4-mini` (OpenAI's economy tier) was being compared against `gemini-3.1-pro-preview` (Google's flagship) — a tier mismatch.

### Reproducibility fix: FFN/CNN/RNN never had a fixed seed

While parametrizing the seed for the 5-run sweep below, found that `08_feed_forward.ipynb`, `09_cnn.ipynb` and `10_rnn.ipynb` never called `torch.manual_seed()`. Every prior run of these three notebooks, across Iterations 1-3, was non-deterministic. Fixed by adding `SEED = 42` plus `torch.manual_seed(SEED)` / `np.random.seed(SEED)` / `random.seed(SEED)` at the top of each notebook.

Even with the seed fixed, exact bit-for-bit reproducibility is not guaranteed on CPU: PyTorch's multi-threaded CPU matrix reductions are not fully deterministic unless `torch.use_deterministic_algorithms(True)` is set and threading is restricted to 1, which would slow training meaningfully and wasn't judged worth it — the 5-seed sweep below already characterizes and reports this residual variance directly, seed=42 included. `metrics.csv` / `metrics.json` were regenerated with the fix; FFN/CNN/RNN numbers below now differ slightly from Iterations 1-3 (every other model is deterministic given a fixed seed and is unchanged):

| Model | Variant (Pre-filtered) | Old (unseeded) | New (SEED=42) |
| :--- | :--- | :---: | :---: |
| FFN | Standard | 72.89% | 74.22% |
| TextCNN | Standard | 77.33% | 78.67% |
| BiLSTM | Standard | 76.67% | 76.89% |

(Irony/Obfuscated and the raw-corpus variants changed similarly — see `metrics.csv` for the full table.)

### 5-seed variance study

Every one of the 48 configurations (2 corpora × 3 variants × 8 models) was re-run with 5 seeds: 42, 123, 2024, 7, 99. Seed 42 is the existing Iteration 3 canonical run (not re-run, to save ~5-6 hours of compute); the other 4 were run fresh. `scripts/run_multiseed.sh` orchestrates the sweep, calling `scripts/train_bert_multiseed.py` for BERT and re-running notebooks 03-10 per seed/corpus for the rest. Results:

- `evaluation/multiseed_all_runs.csv` — every individual run (48 configs × 5 seeds = 240 rows).
- `evaluation/multiseed_summary.csv` — mean ± standard deviation per configuration.

**Key finding**: BETO's best configuration (pre-filtered/standard) is **85.42% ± 1.13pp** across 5 seeds — the single-run Iteration 3 value (86.00%) falls within the normal range, not a lucky outlier. More importantly: BETO's 0.67pp edge over GPT-5.4-mini reported in Iteration 3 is smaller than BETO's own run-to-run standard deviation. That comparison was never distinguishable from noise on a single run.

Naive Bayes and Logistic Regression show ~0 variance across seeds (their solvers are effectively deterministic given fixed data). Random Forest and the neural models (FFN/CNN/RNN/BERT) show real variance, on the order of 1-2pp of accuracy — consistent with the reproducibility finding above.

Note: this 5-seed treatment was **not** applied to the external LLMs (GPT/Gemini, next section) — repeating that benchmark 5x would have meant 5x its API cost, and it wasn't part of what was requested. The LLM numbers below remain single-run; this asymmetry is disclosed rather than hidden.

### LLM tier symmetry: `gpt-5.6-sol` added

`gpt-5.4-mini` (economy tier) vs. `gemini-3.1-pro-preview` (Google's flagship) was a tier mismatch. Added `gpt-5.6-sol` — OpenAI's current flagship (confirmed via `client.models.list()`; there is no `gemini-3.x-pro` newer than 3.1, so Gemini 3.1 Pro remains Google's current flagship too). Same protocol as the existing models (one API call per tweet, 5 prompt versions, both corpora). Full results, confusion matrices and per-version breakdown in [`ahbgpt`](https://github.com/lhbelfanti/ahbgpt) `FINAL_RESULTS.md`.

| Corpus | Model | Best prompt | Accuracy | F1 (macro) |
| :--- | :--- | :---: | :---: | :---: |
| Pre-filtered | **Gemini 3.1 Pro** | V1 | **93.11%** | **93.09%** |
| Pre-filtered | GPT-5.6-sol | V1 | 92.67% | 92.66% |
| Pre-filtered | GPT-5.4-mini | V1 | 85.33% | 85.18% |
| Raw | **GPT-5.6-sol** | V1 | **93.11%** | **93.11%** |
| Raw | Gemini 3.1 Pro | V1 | 92.22% | 92.19% |
| Raw | GPT-5.4-mini | V1 | 85.11% | 84.95% |

`gpt-5.6-sol` and Gemini 3.1 Pro are effectively tied — each leads on one corpus, within half a point of the other on both — a very different picture from the ~7-8 point gap `gpt-5.4-mini` showed against Gemini.
