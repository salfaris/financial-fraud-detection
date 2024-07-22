from joblib import load
from pathlib import Path
from typing import Literal

import numpy as np

from model_config import FEATURE_NAMES

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "model"

TXN_TYPES_MODEL = {
    "TRANSFER": load(MODEL_DIR / "model_TRANSFER_logreg.pkl"),
    "CASH_OUT": load(MODEL_DIR / "model_CASH_OUT_logreg.pkl"),
}

TRANSACTION_TYPE_COLUMN: str = "type"

Output = Literal[0, 1]


def preprocess(batch):
    batch = batch[FEATURE_NAMES + [TRANSACTION_TYPE_COLUMN]]
    return batch


def predict(rowset) -> Output:
    txn_type = rowset[[TRANSACTION_TYPE_COLUMN]].iloc[0]
    if txn_type not in TXN_TYPES_MODEL:
        return 0
    else:
        rowfeatures = rowset[FEATURE_NAMES].to_frame().T
        return TXN_TYPES_MODEL[txn_type].predict(rowfeatures)[0]


def batch_predict(batch) -> np.ndarray:
    return batch.apply(predict, axis=1).values.ravel()


def pipeline(batch):
    batch = preprocess(batch)
    batch["is_fraud"] = batch_predict(batch)
    return batch
