#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate dev

echo "Running PCA for all models..."
python $(dirname "$0")/src/pca.py
