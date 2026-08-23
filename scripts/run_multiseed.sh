#!/usr/bin/env bash
# 5-seed variance sweep requested by Claudio (director): seed 42 is the already
# published Iteration 3 run (evaluation/metrics.csv + evaluation/bert_results.json)
# and is NOT re-run. This script only runs the 4 extra seeds, sequentially
# (never in parallel), redirecting all model artifacts to models_multiseed/ so
# the canonical models/ tree (published to GitHub + Hugging Face) is untouched.
set -uo pipefail
cd "$(dirname "$0")/.."

VENV312=".venv312/bin"
KERNEL="drug-use-modeler-312"
SEEDS=(123 2024 7 99)
LOG_DIR="scripts/multiseed_logs"
mkdir -p "$LOG_DIR" "evaluation/multiseed" "models_multiseed"

CORPORA=("raw-corpus" "pre-filtered-corpus")
CLASSICAL_NBS=("03_logistic_regression" "04_svm" "05_naive_bayes" "06_random_forest")
DL_NBS=("07_word2vec_embeddings" "08_feed_forward" "09_cnn" "10_rnn")
ALL_NBS=("${CLASSICAL_NBS[@]}" "${DL_NBS[@]}")

set_var() {
    # set_var <notebook> <VAR_NAME> <literal_rhs>
    local nb="$1" var="$2" value="$3"
    python3 - "$nb" "$var" "$value" << 'EOF'
import json, re, sys
nb_path, var, value = sys.argv[1], sys.argv[2], sys.argv[3]
path = f"notebooks/{nb_path}.ipynb"
nb = json.load(open(path))
pattern = re.compile(rf"^{re.escape(var)}\s*=\s*.+$")
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    new_src = []
    for line in cell['source']:
        stripped = line.rstrip("\n")
        if pattern.match(stripped):
            line = f"{var} = {value}\n"
        new_src.append(line)
    cell['source'] = new_src
json.dump(nb, open(path, 'w'), indent=1, ensure_ascii=True)
EOF
}

redirect_models_dir() {
    # One-time (per direction) substring swap: models/{CORPUS_NAME -> models_multiseed/{CORPUS_NAME
    local direction="$1" # "to_multiseed" | "to_canonical"
    for nb in "${ALL_NBS[@]}"; do
        python3 - "$nb" "$direction" << 'EOF'
import json, sys
nb_path, direction = sys.argv[1], sys.argv[2]
path = f"notebooks/{nb_path}.ipynb"
nb = json.load(open(path))
a, b = "'../models/{CORPUS_NAME", "'../models_multiseed/{CORPUS_NAME"
src_pat, dst_pat = (a, b) if direction == "to_multiseed" else (b, a)
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    cell['source'] = [line.replace(src_pat, dst_pat) for line in cell['source']]
json.dump(nb, open(path, 'w'), indent=1, ensure_ascii=True)
EOF
    done
}

run_nb() {
    local nb="$1" label="$2"
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Running $nb ($label) ==="
    "$VENV312/jupyter" nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.kernel_name="$KERNEL" \
        --ExecutePreprocessor.timeout=3600 \
        "notebooks/$nb.ipynb" > "$LOG_DIR/${nb}_${label}.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "!!! FAILED: $nb ($label) — see $LOG_DIR/${nb}_${label}.log"
        return 1
    fi
}

echo "########## Redirecting notebook model output to models_multiseed/ ##########"
redirect_models_dir "to_multiseed"

for SEED in "${SEEDS[@]}"; do
    echo "########## SEED $SEED — START $(date) ##########"

    for nb in "${ALL_NBS[@]}"; do
        set_var "$nb" "SEED" "$SEED"
    done

    for corpus in "${CORPORA[@]}"; do
        for nb in "${ALL_NBS[@]}"; do
            set_var "$nb" "CORPUS_NAME" "'$corpus'"
            run_nb "$nb" "seed${SEED}_${corpus}" || exit 1
        done
    done

    "$VENV312/python" scripts/train_bert_multiseed.py "$SEED" > "$LOG_DIR/bert_seed${SEED}.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "!!! FAILED: BERT seed $SEED — see $LOG_DIR/bert_seed${SEED}.log"
        exit 1
    fi

    MULTISEED_MODELS_DIR="models_multiseed" MULTISEED_OUT_SUFFIX="_seed${SEED}" \
        "$VENV312/python" scripts/evaluate_all.py > "$LOG_DIR/evaluate_seed${SEED}.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "!!! FAILED: evaluate_all seed $SEED — see $LOG_DIR/evaluate_seed${SEED}.log"
        exit 1
    fi

    echo "########## SEED $SEED — DONE $(date), freeing disk ##########"
    rm -rf models_multiseed
    mkdir -p models_multiseed
done

echo "########## Restoring notebooks to canonical models/ + SEED=42 + original CORPUS_NAME ##########"
redirect_models_dir "to_canonical"
for nb in "${ALL_NBS[@]}"; do
    set_var "$nb" "SEED" "42"
done
set_var "03_logistic_regression" "CORPUS_NAME" "'pre-filtered-corpus'"
for nb in "04_svm" "05_naive_bayes" "06_random_forest"; do
    set_var "$nb" "CORPUS_NAME" "'raw-corpus'"
done
set_var "07_word2vec_embeddings" "CORPUS_NAME" "'pre-filtered-corpus'"
set_var "08_feed_forward" "CORPUS_NAME" "'pre-filtered-corpus'"
set_var "09_cnn" "CORPUS_NAME" "'pre-filtered-corpus'"
set_var "10_rnn" "CORPUS_NAME" "'pre-filtered-corpus'"

rm -rf models_multiseed

echo "########## Aggregating ##########"
"$VENV312/python" scripts/aggregate_multiseed.py

echo "ALL_DONE $(date '+%Y-%m-%d %H:%M:%S')"
