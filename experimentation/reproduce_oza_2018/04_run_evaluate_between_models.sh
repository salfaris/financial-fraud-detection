#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate dev

# Define a function to evaluate model performance
evaluate_performance() {
    local transaction_type=$1
    echo "Evaluating between-model performance for transaction type: $transaction_type"
    python $(dirname "$0")/src/evaluate_between_ideal_model_performance.py --transaction_type "$transaction_type"
}

# Evaluate performance for the specified transaction types
transaction_types=("TRANSFER" "CASH_OUT")
for transaction_type in "${transaction_types[@]}"; do
    evaluate_performance "$transaction_type"
    echo ""
done
