"""Run BERT fine-tuning directly (no notebook timeout constraints).

Checkpoint selection uses the VALIDATION split (val.csv); the TEST split is
evaluated exactly once at the end, after the best-on-validation checkpoint is
loaded. Test never participates in choosing which epoch's weights to keep.
"""
import numpy as np
import pandas as pd
import torch
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

MODEL_NAME = 'dccuchile/bert-base-spanish-wwm-cased'
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
LABEL_MAP = {'NEGATIVE': 0, 'POSITIVE': 1}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def load_dataset(data_dir):
    """Load train/val/test CSVs and convert to HuggingFace Datasets."""
    train_df = pd.read_csv(f'{data_dir}/train.csv').fillna('')
    val_df   = pd.read_csv(f'{data_dir}/val.csv').fillna('')
    test_df  = pd.read_csv(f'{data_dir}/test.csv').fillna('')

    for df in (train_df, val_df, test_df):
        df['label'] = df['label'].map(LABEL_MAP)

    train_ds = Dataset.from_pandas(train_df[['text_clean', 'label']])
    val_ds   = Dataset.from_pandas(val_df[['text_clean', 'label']])
    test_ds  = Dataset.from_pandas(test_df[['text_clean', 'label']])

    return train_ds, val_ds, test_ds

def tokenize_fn(examples):
    return tokenizer(
        examples['text_clean'],
        padding='max_length',
        truncation=True,
        max_length=MAX_LEN
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

def train_bert(variation_name, data_dir, output_dir, seed=42, save_artifacts=True):
    print(f"\n{'='*20} BERT: {variation_name} (seed={seed}) {'='*20}", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds, val_ds, test_ds = load_dataset(data_dir)
    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds   = val_ds.map(tokenize_fn, batched=True)
    test_ds  = test_ds.map(tokenize_fn, batched=True)

    for ds in (train_ds, val_ds, test_ds):
        ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=f'{output_dir}/checkpoints',
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_steps=50,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        seed=seed,
        report_to='none',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print(f"PIPELINE_TRAIN_DONE {variation_name}", flush=True)

    # Single, blind pass on test: the model was already selected using val.
    test_output = trainer.predict(test_ds)
    y_pred = np.argmax(test_output.predictions, axis=-1)
    y_true = test_output.label_ids
    m = test_output.metrics
    acc, precision, recall, f1 = (
        m['test_accuracy'], m['test_precision'], m['test_recall'], m['test_f1']
    )

    print(
        f"\nBERT ({variation_name}) TEST — Accuracy: {acc:.4f}  "
        f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}",
        flush=True,
    )
    print(classification_report(y_true, y_pred))

    if save_artifacts:
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(f'{output_dir}/model')
        tokenizer.save_pretrained(f'{output_dir}/tokenizer')
        print(f"Model saved to {output_dir}", flush=True)

    # Checkpoints are only needed transiently for load_best_model_at_end; drop them
    # for non-canonical seeds so a 5-seed sweep doesn't multiply disk usage by 5x.
    import shutil
    shutil.rmtree(f'{output_dir}/checkpoints', ignore_errors=True)

    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    CORPUS_NAME = 'raw-corpus'
    PROCESSED_DIR = f'data/processed/{CORPUS_NAME}'
    MODELS_DIR = f'models/{CORPUS_NAME}/bert_base'

    result = train_bert("Standard", f"{PROCESSED_DIR}/standard", f"{MODELS_DIR}/standard")
    print(f"\nStandard: {result}")
