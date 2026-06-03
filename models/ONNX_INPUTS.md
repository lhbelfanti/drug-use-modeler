# ONNX Model Input/Output Reference

All models live under `models/<corpus>/<family>/<variant>/` where:

- **corpus**: `pre-filtered-corpus` | `raw-corpus`
- **family**: `logistic_regression` | `naive_bayes` | `random_forest` | `svm` | `ffn` | `cnn` | `rnn` | `bert_base`
- **variant**: `standard` | `irony` | `obfuscated`

Scikit-learn families (`logistic_regression`, `naive_bayes`, `random_forest`, `svm`) expose two files per variant: a **vectorizer** and a **classifier**. The other families expose a single `model.onnx`.

> To inspect any model at runtime run:
> ```bash
> python scripts/inspect_onnx_inputs.py                        # all models
> python scripts/inspect_onnx_inputs.py --filter bert_base     # one family
> ```

---

## Scikit-learn — Vectorizer (`vectorizer.onnx`)

Applies to: `logistic_regression`, `naive_bayes`, `random_forest`, `svm`

| # | Name | Shape | Type |
|---|------|-------|------|
| **Input 0** | `text_input` | `[batch_size, 1]` | `tensor(string)` |
| **Output 0** | `variable` | `[batch_size, vocab_size]` | `tensor(float)` |

Each sample must be wrapped in a 2-D array with shape `(N, 1)` where each cell holds one raw text string. The output is a dense TF-IDF feature matrix.

---

## Scikit-learn — Classifier (`model.onnx`)

Applies to: `logistic_regression`, `naive_bayes`, `random_forest`, `svm`

| # | Name | Shape | Type |
|---|------|-------|------|
| **Input 0** | `float_input` | `[batch_size, n_features]` | `tensor(float)` |
| **Output 0** | `label` | `[batch_size]` | `tensor(int64)` |
| **Output 1** | `probabilities` | `[batch_size, 2]` | `tensor(float)` |

`n_features` equals the vocabulary size of the paired vectorizer. Feed the vectorizer output directly into this input. `probabilities[:, 1]` is the positive-class score.

---

## FFN (`model.onnx`)

Feed-Forward Network trained on averaged Word2Vec embeddings.

| # | Name | Shape | Type |
|---|------|-------|------|
| **Input 0** | `input` | `[batch_size, input_dim]` | `tensor(float)` |
| **Output 0** | `output` | `[batch_size]` | `tensor(float)` |

`input_dim` matches the Word2Vec embedding dimension for that specific model (inspect with the script above to get the exact value). Output is a scalar probability in `[0, 1]`.

---

## CNN / TextCNN (`model.onnx`)

Convolutional network with a frozen embedding layer.

| # | Name | Shape | Type |
|---|------|-------|------|
| **Input 0** | `input_ids` | `[batch_size, sequence_length]` | `tensor(int64)` |
| **Output 0** | `output` | `[batch_size]` | `tensor(float)` |

`input_ids` are integer token indices from the model's vocabulary. `sequence_length` is dynamic. Output is a scalar probability in `[0, 1]`.

---

## RNN / BiLSTM (`model.onnx`)

Bidirectional LSTM with a frozen embedding layer.

| # | Name | Shape | Type |
|---|------|-------|------|
| **Input 0** | `input_ids` | `[batch_size, sequence_length]` | `tensor(int64)` |
| **Output 0** | `output` | `[batch_size]` | `tensor(float)` |

Same token-index convention as CNN. Output is a scalar probability in `[0, 1]`.

---

## BERT Base (`model/model.onnx`)

HuggingFace `bert-base-*` fine-tuned for sequence classification.

| # | Name | Shape | Type |
|---|------|-------|------|
| **Input 0** | `input_ids` | `[batch_size, sequence_length]` | `tensor(int64)` |
| **Input 1** | `attention_mask` | `[batch_size, sequence_length]` | `tensor(int64)` |
| **Output 0** | `logits` | `[batch_size, 2]` | `tensor(float)` |

Both inputs come from the tokenizer stored in the sibling `tokenizer/` directory. `logits[:, 1]` is the positive-class raw score (apply softmax for probabilities). `sequence_length` is dynamic.
