"""Fine-tune BERT (BETO) for both corpora x all 3 pipeline variants, on the
new stratified-by-drug splits. Reuses train_bert from train_bert_base.py."""
import torch
import numpy as np

from train_bert_base import train_bert

CORPORA = ['raw-corpus', 'pre-filtered-corpus']
VARIANTS = [
    ('Standard', 'standard'),
    ('Irony', 'irony'),
    ('Obfuscated', 'obfuscated'),
]

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    results = {}
    for corpus in CORPORA:
        processed_dir = f'data/processed/{corpus}'
        models_dir = f'models/{corpus}/bert_base'
        for label, variant in VARIANTS:
            acc = train_bert(
                f'{corpus}/{label}',
                f'{processed_dir}/{variant}',
                f'{models_dir}/{variant}',
            )
            results[f'{corpus}/{variant}'] = acc

    print("\n=== Summary ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
