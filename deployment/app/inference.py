from joblib import load
from typing import Literal

import numpy as np
import streamlit as st

import app_config
from model_config import FEATURE_NAMES


@st.cache_resource
def load_model(model_path):
    return load(model_path)


TXN_TYPES_MODEL = {
    "TRANSFER": load_model(app_config.MODEL_DIR / "model_TRANSFER_logreg.pkl"),
    "CASH_OUT": load_model(app_config.MODEL_DIR / "model_CASH_OUT_logreg.pkl"),
}

TRANSACTION_TYPE_COLUMN: str = "type"

Output = Literal[0, 1]


def preprocess(batch):
    batch = batch[FEATURE_NAMES + [TRANSACTION_TYPE_COLUMN]].copy()
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


def predict_pipeline(batch):
    batch = preprocess(batch)
    batch.loc[:, "flagged_fraud"] = batch_predict(batch)
    return batch
