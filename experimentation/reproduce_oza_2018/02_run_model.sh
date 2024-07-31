#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate dev

models=("logreg" "svc_linear" "svc_rbf" "decision_tree")
transaction_types=("TRANSFER" "CASH_OUT")

for model in "${models[@]}"; do
    echo "Running ${model}..."
    for transaction_type in "${transaction_types[@]}"; do
        python $(dirname "$0")/src/evaluate_model_metrics.py --model_name "${model}" --transaction_type "${transaction_type}"
    done
done
