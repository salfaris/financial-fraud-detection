from joblib import dump, load
from pathlib import Path


def load_model(model_path: Path):
    return load(model_path)


def save_model(model, model_path: Path):
    with open(model_path, "wb") as f:
        dump(model, f)
