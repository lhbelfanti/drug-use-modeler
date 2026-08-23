"""
Evaluate all trained models and compute full metrics:
Accuracy, Precision, Recall, F1-Score (macro avg + per class).

For raw-corpus NB/SVM/RF (no saved joblib), models are re-trained inline
with identical hyperparameters (deterministic, seed=42).

Outputs:
  evaluation/metrics.json  — full structured results
  evaluation/metrics.csv   — flat table for easy comparison
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Constants ─────────────────────────────────────────────────────────────────

CORPORA = ['pre-filtered-corpus', 'raw-corpus']
VARIANTS = ['standard', 'irony', 'obfuscated']
LABEL_MAP = {'NEGATIVE': 0, 'POSITIVE': 1}
LABEL_NAMES = ['NEGATIVE', 'POSITIVE']
PROCESSED_DIR = 'data/processed'
MODELS_DIR = os.environ.get('MULTISEED_MODELS_DIR', 'models')
OUT_SUFFIX = os.environ.get('MULTISEED_OUT_SUFFIX', '')
MAX_LEN = 50
EMBED_DIM = 100
BERT_MAX_LEN = 128
BERT_BATCH_SIZE = 32

# ── PyTorch architectures (must match training exactly) ───────────────────────

class FFN(nn.Module):
    def __init__(self, input_dim=100):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


class TextCNN(nn.Module):
    def __init__(self, embedding_matrix, num_filters=100, filter_sizes=(3, 4, 5), dropout=0.3):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(torch.FloatTensor(embedding_matrix))
        self.embedding.weight.requires_grad = False
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=fs) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(x))
            c = F.max_pool1d(c, c.size(2)).squeeze(2)
            conv_outs.append(c)
        out = self.dropout(torch.cat(conv_outs, dim=1))
        return torch.sigmoid(self.fc(out)).squeeze(1)


class BiLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=64, dropout=0.3):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(torch.FloatTensor(embedding_matrix))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        combined = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return torch.sigmoid(self.fc(self.dropout(combined))).squeeze(1)


# ── Shared helpers ────────────────────────────────────────────────────────────

def build_embedding_matrix(w2v_model):
    vocab = w2v_model.wv.key_to_index
    matrix = np.zeros((len(vocab) + 1, EMBED_DIM))
    word2idx = {'<PAD>': 0}
    for word, idx in vocab.items():
        word2idx[word] = idx + 1
        matrix[idx + 1] = w2v_model.wv[word]
    return matrix, word2idx


def texts_to_sequences(texts, word2idx, max_len):
    sequences = []
    for text in texts:
        tokens = str(text).split()
        seq = [word2idx.get(w, 0) for w in tokens[:max_len]]
        seq += [0] * (max_len - len(seq))
        sequences.append(seq)
    return np.array(sequences)


def extract_metrics(y_true, y_pred):
    report = classification_report(
        y_true, y_pred, target_names=LABEL_NAMES, output_dict=True, zero_division=0
    )
    macro = report['macro avg']
    return {
        'accuracy': round(float(report['accuracy']), 4),
        'precision': round(float(macro['precision']), 4),
        'recall': round(float(macro['recall']), 4),
        'f1': round(float(macro['f1-score']), 4),
        'per_class': {
            lbl: {
                'precision': round(float(report[lbl]['precision']), 4),
                'recall': round(float(report[lbl]['recall']), 4),
                'f1': round(float(report[lbl]['f1-score']), 4),
                'support': int(report[lbl]['support']),
            }
            for lbl in LABEL_NAMES
        },
    }


def load_test(data_dir):
    return pd.read_csv(f'{data_dir}/test.csv').fillna('')


# ── Evaluators ────────────────────────────────────────────────────────────────

def eval_sklearn_from_disk(model_dir, data_dir):
    model = joblib.load(f'{model_dir}/model.joblib')
    vectorizer = joblib.load(f'{model_dir}/vectorizer.joblib')
    test_df = load_test(data_dir)
    X_test = vectorizer.transform(test_df['text_clean'])
    y_pred = model.predict(X_test)
    return extract_metrics(test_df['label'].values, y_pred)


def eval_sklearn_retrain(clf_cls, clf_kwargs, data_dir):
    """Re-train a sklearn model inline (for raw-corpus NB/SVM/RF missing joblib)."""
    train_df = pd.read_csv(f'{data_dir}/train.csv').fillna('')
    test_df = load_test(data_dir)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df['text_clean'])
    X_test = vectorizer.transform(test_df['text_clean'])
    clf = clf_cls(**clf_kwargs)
    clf.fit(X_train, train_df['label'])
    y_pred = clf.predict(X_test)
    return extract_metrics(test_df['label'].values, y_pred)


def eval_ffn(model_dir, w2v_dir):
    X_test = np.load(f'{w2v_dir}/doc_vectors_test.npy', allow_pickle=True)
    labels_test = np.load(f'{w2v_dir}/labels_test.npy', allow_pickle=True)
    y_true = np.array([LABEL_MAP[l] for l in labels_test])

    model = FFN(input_dim=X_test.shape[1])
    model.load_state_dict(torch.load(f'{model_dir}/model.pt', map_location='cpu', weights_only=True))
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test)).squeeze()
    y_pred = (preds >= 0.5).int().numpy()
    return extract_metrics(y_true, y_pred)


def eval_cnn(model_dir, w2v_path, data_dir):
    w2v = Word2Vec.load(w2v_path)
    embed_matrix, word2idx = build_embedding_matrix(w2v)
    test_df = load_test(data_dir)
    X_test = texts_to_sequences(test_df['text_clean'], word2idx, MAX_LEN)
    y_true = test_df['label'].map(LABEL_MAP).values

    model = TextCNN(embed_matrix)
    model.load_state_dict(torch.load(f'{model_dir}/model.pt', map_location='cpu', weights_only=True))
    model.eval()
    with torch.no_grad():
        preds = model(torch.LongTensor(X_test))
    y_pred = (preds >= 0.5).int().numpy()
    return extract_metrics(y_true, y_pred)


def eval_rnn(model_dir, w2v_path, data_dir):
    w2v = Word2Vec.load(w2v_path)
    embed_matrix, word2idx = build_embedding_matrix(w2v)
    test_df = load_test(data_dir)
    X_test = texts_to_sequences(test_df['text_clean'], word2idx, MAX_LEN)
    y_true = test_df['label'].map(LABEL_MAP).values

    model = BiLSTM(embed_matrix)
    model.load_state_dict(torch.load(f'{model_dir}/model.pt', map_location='cpu', weights_only=True))
    model.eval()
    with torch.no_grad():
        preds = model(torch.LongTensor(X_test))
    y_pred = (preds >= 0.5).int().numpy()
    return extract_metrics(y_true, y_pred)


def eval_bert(model_dir, tokenizer_dir, data_dir):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    test_df = load_test(data_dir)
    texts = test_df['text_clean'].tolist()
    y_true = test_df['label'].map(LABEL_MAP).values

    all_preds = []
    for i in range(0, len(texts), BERT_BATCH_SIZE):
        batch = texts[i:i + BERT_BATCH_SIZE]
        inputs = tokenizer(
            batch,
            padding='max_length',
            truncation=True,
            max_length=BERT_MAX_LEN,
            return_tensors='pt',
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        all_preds.extend(torch.argmax(logits, dim=-1).tolist())

    return extract_metrics(y_true, np.array(all_preds))


# ── Model registry ────────────────────────────────────────────────────────────

def get_evaluators(corpus, variant):
    """
    Returns a dict of model_name -> callable that returns metrics.
    Handles missing raw-corpus sklearn models by re-training inline.
    """
    base = f'{MODELS_DIR}/{corpus}'
    data_dir = f'{PROCESSED_DIR}/{corpus}/{variant}'
    w2v_dir = f'{base}/word2vec/{variant}'
    w2v_path = f'{w2v_dir}/word2vec.model'

    sklearn_models = {
        'logistic_regression': {
            'cls': LogisticRegression,
            'kwargs': {'class_weight': 'balanced', 'max_iter': 1000, 'random_state': 42},
        },
        'naive_bayes': {
            'cls': MultinomialNB,
            'kwargs': {},
        },
        'svm': {
            'cls': LinearSVC,
            'kwargs': {'class_weight': 'balanced', 'random_state': 42, 'dual': 'auto'},
        },
        'random_forest': {
            'cls': RandomForestClassifier,
            'kwargs': {'n_estimators': 100, 'class_weight': 'balanced', 'random_state': 42},
        },
    }

    evaluators = {}

    for name, spec in sklearn_models.items():
        model_dir = f'{base}/{name}/{variant}'
        joblib_path = f'{model_dir}/model.joblib'
        if os.path.exists(joblib_path):
            evaluators[name] = lambda md=model_dir, dd=data_dir: eval_sklearn_from_disk(md, dd)
        else:
            evaluators[name] = lambda cls=spec['cls'], kw=spec['kwargs'], dd=data_dir: \
                eval_sklearn_retrain(cls, kw, dd)

    evaluators['ffn'] = lambda md=f'{base}/ffn/{variant}', wd=w2v_dir: eval_ffn(md, wd)
    evaluators['cnn'] = lambda md=f'{base}/cnn/{variant}', wp=w2v_path, dd=data_dir: eval_cnn(md, wp, dd)
    evaluators['rnn'] = lambda md=f'{base}/rnn/{variant}', wp=w2v_path, dd=data_dir: eval_rnn(md, wp, dd)
    evaluators['bert_base'] = lambda md=f'{base}/bert_base/{variant}/model', \
                                     td=f'{base}/bert_base/{variant}/tokenizer', \
                                     dd=data_dir: eval_bert(md, td, dd)

    return evaluators


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    results = {}
    rows = []

    for corpus in CORPORA:
        results[corpus] = {}
        for variant in VARIANTS:
            results[corpus][variant] = {}
            evaluators = get_evaluators(corpus, variant)

            for model_name, evaluate in evaluators.items():
                print(f'  {corpus} / {variant} / {model_name} ...', flush=True)
                try:
                    metrics = evaluate()
                    results[corpus][variant][model_name] = metrics
                    rows.append({
                        'corpus': corpus,
                        'variant': variant,
                        'model': model_name,
                        'accuracy': metrics['accuracy'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f1': metrics['f1'],
                    })
                    print(
                        f'    acc={metrics["accuracy"]:.4f}  '
                        f'prec={metrics["precision"]:.4f}  '
                        f'rec={metrics["recall"]:.4f}  '
                        f'f1={metrics["f1"]:.4f}'
                    )
                except Exception as e:
                    print(f'    ERROR: {e}')
                    results[corpus][variant][model_name] = {'error': str(e)}

    out_dir = 'evaluation' if not OUT_SUFFIX else 'evaluation/multiseed'
    os.makedirs(out_dir, exist_ok=True)

    json_path = f'{out_dir}/metrics{OUT_SUFFIX}.json'
    csv_path = f'{out_dir}/metrics{OUT_SUFFIX}.csv'

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved {json_path}')

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f'Saved {csv_path}')


if __name__ == '__main__':
    main()
