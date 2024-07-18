#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate dev

echo "Running Logistic Regression..."
python src/model.py --model_name logreg --transaction_type TRANSFER
python src/model.py --model_name logreg --transaction_type CASH_OUT

echo "Running Linear SVC..."
python src/model.py --model_name svc_linear --transaction_type TRANSFER
python src/model.py --model_name svc_linear --transaction_type CASH_OUT

echo "Running SVC with RBF kernel..."
python src/model.py --model_name svc_rbf --transaction_type TRANSFER
python src/model.py --model_name svc_rbf --transaction_type CASH_OUT