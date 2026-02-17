"""Run BERT fine-tuning directly (no notebook timeout constraints)."""
import numpy as np
import pandas as pd
import torch
import json
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report

MODEL_NAME = 'dccuchile/bert-base-spanish-wwm-cased'
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 2e-5
LABEL_MAP = {'NEGATIVE': 0, 'POSITIVE': 1}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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
    return {'accuracy': accuracy_score(labels, preds)}

def train_bert(variation_name, data_dir, output_dir):
    print(f"\n{'='*20} BERT: {variation_name} {'='*20}")
    
    train_df = pd.read_csv(f'{data_dir}/train.csv').fillna('')
    test_df  = pd.read_csv(f'{data_dir}/test.csv').fillna('')
    
    train_df['label'] = train_df['label'].map(LABEL_MAP)
    test_df['label']  = test_df['label'].map(LABEL_MAP)
    
    train_ds = Dataset.from_pandas(train_df[['text_clean', 'label']])
    test_ds  = Dataset.from_pandas(test_df[['text_clean', 'label']])
    
    train_ds = train_ds.map(tokenize_fn, batched=True)
    test_ds  = test_ds.map(tokenize_fn, batched=True)
    
    train_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
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
        seed=42,
        report_to='none',
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    results = trainer.evaluate()
    acc = results['eval_accuracy']
    print(f"\nBERT ({variation_name}) Accuracy: {acc:.4f}")
    
    preds_output = trainer.predict(test_ds)
    y_pred = np.argmax(preds_output.predictions, axis=-1)
    y_true = preds_output.label_ids
    print(classification_report(y_true, y_pred))
    
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(f'{output_dir}/model')
    tokenizer.save_pretrained(f'{output_dir}/tokenizer')
    print(f"Model saved to {output_dir}")
    
    return acc

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    
    CORPUS_NAME = 'raw-corpus'
    PROCESSED_DIR = f'data/processed/{CORPUS_NAME}'
    MODELS_DIR = f'models/{CORPUS_NAME}/bert_base'

    # acc_std = train_bert("Standard", f"{PROCESSED_DIR}/standard", f"{MODELS_DIR}/standard")
    # acc_iro = train_bert("Irony", f"{PROCESSED_DIR}/irony", f"{MODELS_DIR}/irony")
    acc_obf = train_bert("Obfuscated", f"{PROCESSED_DIR}/obfuscated", f"{MODELS_DIR}/obfuscated")
    
    print(f"\nObfuscated: {acc_obf:.4f}")

