"""Fine-tune BERT (BETO) for both corpora x all 3 pipeline variants, for a single
non-canonical seed (seed=42 is the already-published Iteration 3 run and is not
re-run here). Model artifacts are NOT persisted (save_artifacts=False in
train_bert) — only metrics are kept, to avoid multiplying BERT's ~440MB per
config x 6 configs x 4 extra seeds of disk usage.

Usage: python train_bert_multiseed.py <seed>
Writes: evaluation/multiseed/bert_seed<seed>.json
"""
import json
import os
import sys
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
    seed = int(sys.argv[1])
    torch.manual_seed(seed)
    np.random.seed(seed)

    results_path = f'evaluation/multiseed/bert_seed{seed}.json'
    results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)

    total = len(CORPORA) * len(VARIANTS)
    done = 0
    for corpus in CORPORA:
        processed_dir = f'data/processed/{corpus}'
        # output_dir is only used for the transient checkpoints dir (cleaned up
        # inside train_bert); no final model/tokenizer gets written here.
        models_dir = f'models_multiseed/{corpus}/bert_base'
        for label, variant in VARIANTS:
            key = f'{corpus}/{variant}'
            if key in results:
                done += 1
                continue

            metrics = train_bert(
                f'{corpus}/{label}',
                f'{processed_dir}/{variant}',
                f'{models_dir}/{variant}',
                seed=seed,
                save_artifacts=False,
            )
            results[key] = metrics
            done += 1
            print(f"PIPELINE_COMPLETE {done}/{total} seed={seed} {corpus}/{variant} {metrics}", flush=True)

            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

    print(f"\n=== Summary (seed={seed}) ===", flush=True)
    for k, v in results.items():
        print(f"{k}: {v}", flush=True)
    print("ALL_DONE", flush=True)
