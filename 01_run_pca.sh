#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate dev

echo "Running PCA for all models..."
python src/pca.py