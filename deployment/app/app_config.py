from pathlib import Path

ROOT_DIR = Path(__file__).parents[0]  # Dockerize, use parents[0]; otherwise parents[1]
DATA_DIR = ROOT_DIR / "datasets"
MODEL_DIR = ROOT_DIR / "model"
