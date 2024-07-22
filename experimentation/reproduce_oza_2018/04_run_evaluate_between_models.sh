#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate dev

echo "Evaluating between-model performance using PR curve"
python src/evaluate_between_ideal_model_performance.py --transaction_type TRANSFER
python src/evaluate_between_ideal_model_performance.py --transaction_type CASH_OUT