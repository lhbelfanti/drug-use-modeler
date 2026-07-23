#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."

VENV312=".venv312/bin"
KERNEL="drug-use-modeler-312"
LOG_DIR="scripts/retrain_logs"
mkdir -p "$LOG_DIR"

run_nb() {
    local nb="$1"
    local label="$2"
    echo "=== Running $nb ($label) ==="
    "$VENV312/jupyter" nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.kernel_name="$KERNEL" \
        --ExecutePreprocessor.timeout=3600 \
        "notebooks/$nb.ipynb" > "$LOG_DIR/${nb}_${label}.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "!!! FAILED: $nb ($label) — see $LOG_DIR/${nb}_${label}.log"
        return 1
    fi
    echo "=== Done $nb ($label) ==="
}

set_corpus_name() {
    local nb="$1"
    local corpus="$2"
    python3 - "$nb" "$corpus" << 'EOF'
import json, re, sys
nb_path, corpus = sys.argv[1], sys.argv[2]
path = f"notebooks/{nb_path}.ipynb"
nb = json.load(open(path))
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = cell['source']
    changed = False
    new_src = []
    for line in src:
        if re.match(r"CORPUS_NAME\s*=\s*'", line):
            line = re.sub(r"CORPUS_NAME\s*=\s*'[^']*'", f"CORPUS_NAME = '{corpus}'", line)
            changed = True
        new_src.append(line)
    if changed:
        cell['source'] = new_src
json.dump(nb, open(path, 'w'), indent=1, ensure_ascii=False)
EOF
}

CORPORA=("raw-corpus" "pre-filtered-corpus")

echo "########## STEP 1: word2vec embeddings ##########"
for corpus in "${CORPORA[@]}"; do
    set_corpus_name "07_word2vec_embeddings" "$corpus"
    run_nb "07_word2vec_embeddings" "$corpus" || exit 1
done

echo "########## STEP 2: classical ML (logistic regression) ##########"
for corpus in "${CORPORA[@]}"; do
    set_corpus_name "03_logistic_regression" "$corpus"
    run_nb "03_logistic_regression" "$corpus" || exit 1
done

echo "########## STEP 3: SVM / Naive Bayes / Random Forest (both corpora, notebook-internal) ##########"
run_nb "04_svm" "both" || exit 1
run_nb "05_naive_bayes" "both" || exit 1
run_nb "06_random_forest" "both" || exit 1

echo "########## STEP 4: FFN / CNN / RNN ##########"
for corpus in "${CORPORA[@]}"; do
    for nb in "08_feed_forward" "09_cnn" "10_rnn"; do
        set_corpus_name "$nb" "$corpus"
        run_nb "$nb" "$corpus" || exit 1
    done
done

echo "########## STEP 5: BERT fine-tuning (both corpora x 3 variants) ##########"
"$VENV312/python" scripts/train_bert_all.py > "$LOG_DIR/bert_all.log" 2>&1
if [ $? -ne 0 ]; then
    echo "!!! FAILED: BERT training — see $LOG_DIR/bert_all.log"
    exit 1
fi

echo "########## STEP 6: evaluate_all.py ##########"
"$VENV312/python" scripts/evaluate_all.py > "$LOG_DIR/evaluate_all.log" 2>&1
if [ $? -ne 0 ]; then
    echo "!!! FAILED: evaluate_all.py — see $LOG_DIR/evaluate_all.log"
    exit 1
fi

echo "########## ALL DONE ##########"
