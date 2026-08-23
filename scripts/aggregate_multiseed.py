"""Aggregate the 5-seed sweep (seed 42 = canonical Iteration 3 run, already in
evaluation/metrics.csv; seeds 123/2024/7/99 = evaluation/multiseed/metrics_seed*.csv
+ bert_seed*.json) into mean +/- standard deviation per (corpus, variant, model).

Writes: evaluation/multiseed_summary.csv
"""
import json
import glob
import pandas as pd

CANONICAL_SEED = 42
EXTRA_SEEDS = [123, 2024, 7, 99]

rows = []

# --- canonical seed 42: classical/DL/BERT all come from evaluation/metrics.csv ---
canonical = pd.read_csv('evaluation/metrics.csv')
canonical['seed'] = CANONICAL_SEED
rows.append(canonical)

# --- extra seeds: classical/DL from metrics_seed<N>.csv (bert_base rows dropped;
#     BERT comes from the separate bert_seed<N>.json instead) ---
for seed in EXTRA_SEEDS:
    path = f'evaluation/multiseed/metrics_seed{seed}.csv'
    df = pd.read_csv(path)
    df = df[df['model'] != 'bert_base'].copy()
    df['seed'] = seed
    rows.append(df)

    bert_path = f'evaluation/multiseed/bert_seed{seed}.json'
    with open(bert_path) as f:
        bert_results = json.load(f)
    for key, m in bert_results.items():
        corpus, variant = key.split('/')
        rows.append(pd.DataFrame([{
            'corpus': corpus, 'variant': variant, 'model': 'bert_base',
            'accuracy': m['accuracy'], 'precision': m['precision'],
            'recall': m['recall'], 'f1': m['f1'], 'seed': seed,
        }]))

all_runs = pd.concat(rows, ignore_index=True)
all_runs.to_csv('evaluation/multiseed_all_runs.csv', index=False)
print(f"Saved evaluation/multiseed_all_runs.csv ({len(all_runs)} rows, "
      f"{all_runs['seed'].nunique()} seeds)")

summary = (
    all_runs
    .groupby(['corpus', 'variant', 'model'])[['accuracy', 'precision', 'recall', 'f1']]
    .agg(['mean', 'std'])
)
summary.columns = [f'{metric}_{stat}' for metric, stat in summary.columns]
summary = summary.reset_index()

n_seeds = all_runs.groupby(['corpus', 'variant', 'model'])['seed'].nunique().reset_index(name='n_seeds')
summary = summary.merge(n_seeds, on=['corpus', 'variant', 'model'])

summary.to_csv('evaluation/multiseed_summary.csv', index=False)
print("Saved evaluation/multiseed_summary.csv")
print(summary.to_string(index=False))

incomplete = summary[summary['n_seeds'] < 5]
if len(incomplete):
    print(f"\nWARNING: {len(incomplete)} configs have fewer than 5 seeds:")
    print(incomplete[['corpus', 'variant', 'model', 'n_seeds']].to_string(index=False))
