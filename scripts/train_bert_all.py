"""Fine-tune BERT (BETO) for both corpora x all 3 pipeline variants, with
checkpoint selection on val.csv and a single blind evaluation on test.csv.
Reuses train_bert from train_bert_base.py. Writes results incrementally to
retrain_results.json so progress survives an interruption."""
import json
import os
import torch
import numpy as np

from train_bert_base import train_bert

CORPORA = ['raw-corpus', 'pre-filtered-corpus']
VARIANTS = [
    ('Standard', 'standard'),
    ('Irony', 'irony'),
    ('Obfuscated', 'obfuscated'),
]
RESULTS_PATH = 'scripts/retrain_results.json'

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    results = {}
    total = len(CORPORA) * len(VARIANTS)
    done = 0
    for corpus in CORPORA:
        processed_dir = f'data/processed/{corpus}'
        models_dir = f'models/{corpus}/bert_base'
        for label, variant in VARIANTS:
            metrics = train_bert(
                f'{corpus}/{label}',
                f'{processed_dir}/{variant}',
                f'{models_dir}/{variant}',
            )
            results[f'{corpus}/{variant}'] = metrics
            done += 1
            print(f"PIPELINE_COMPLETE {done}/{total} {corpus}/{variant} {metrics}", flush=True)

            os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
            with open(RESULTS_PATH, 'w') as f:
                json.dump(results, f, indent=2)

    print("\n=== Summary ===", flush=True)
    for k, v in results.items():
        print(f"{k}: {v}", flush=True)
    print("ALL_DONE", flush=True)
