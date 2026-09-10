"""
Desglose de métricas por sustancia para BETO (bert_base), reutilizando el
mismo modelo/corpus/variante que ya usa la Tabla 7 de la tesis
(pre-filtered-corpus, standard, semilla 42), para mantener consistencia
con el desglose por clase ya publicado.

Output:
  evaluation/metrics_by_substance.json
  evaluation/metrics_by_substance.csv
"""

import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CORPUS = 'pre-filtered-corpus'
VARIANT = 'standard'
LABEL_MAP = {'NEGATIVE': 0, 'POSITIVE': 1}
LABEL_NAMES = ['NEGATIVE', 'POSITIVE']
BERT_MAX_LEN = 128
BERT_BATCH_SIZE = 32

model_dir = f'models/{CORPUS}/bert_base/{VARIANT}/model'
tokenizer_dir = f'models/{CORPUS}/bert_base/{VARIANT}/tokenizer'
data_path = f'data/processed/{CORPUS}/{VARIANT}/test.csv'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()

df = pd.read_csv(data_path).fillna('')
texts = df['text_clean'].tolist()
y_true = df['label'].map(LABEL_MAP).values

all_preds = []
for i in range(0, len(texts), BERT_BATCH_SIZE):
    batch = texts[i:i + BERT_BATCH_SIZE]
    inputs = tokenizer(
        batch, padding='max_length', truncation=True,
        max_length=BERT_MAX_LEN, return_tensors='pt',
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    all_preds.extend(torch.argmax(logits, dim=-1).tolist())

df['pred'] = all_preds
df['y_true'] = y_true

results = {}
for drug, group in df.groupby('drug'):
    report = classification_report(
        group['y_true'], group['pred'], labels=[0, 1],
        target_names=LABEL_NAMES, output_dict=True, zero_division=0,
    )
    macro = report['macro avg']
    results[drug] = {
        'support_pairs': len(group) // 2,
        'support_instances': len(group),
        'accuracy': round(float(report['accuracy']), 4),
        'precision_macro': round(float(macro['precision']), 4),
        'recall_macro': round(float(macro['recall']), 4),
        'f1_macro': round(float(macro['f1-score']), 4),
    }

# Overall (sanity check against known Tabla 7 numbers)
overall_report = classification_report(
    df['y_true'], df['pred'], labels=[0, 1],
    target_names=LABEL_NAMES, output_dict=True, zero_division=0,
)
results['__overall__'] = {
    'support_instances': len(df),
    'accuracy': round(float(overall_report['accuracy']), 4),
    'precision_macro': round(float(overall_report['macro avg']['precision']), 4),
    'recall_macro': round(float(overall_report['macro avg']['recall']), 4),
    'f1_macro': round(float(overall_report['macro avg']['f1-score']), 4),
}

with open('evaluation/metrics_by_substance.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

rows = []
for drug, m in results.items():
    rows.append({'drug': drug, **m})
pd.DataFrame(rows).to_csv('evaluation/metrics_by_substance.csv', index=False)

print(json.dumps(results, indent=2, ensure_ascii=False))
