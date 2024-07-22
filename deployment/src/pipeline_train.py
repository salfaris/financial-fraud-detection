from joblib import dump, Parallel, delayed
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC

from absl import app, logging

from model_config import FEATURE_NAMES, TARGET_NAME

ModelName = Literal["logreg", "svc_linear", "svc_rbf"]
TransactionType = Literal["TRANSFER", "CASH_OUT"]

ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / "deployment" / "datasets"
MODEL_DIR = ROOT_DIR / "deployment" / "model"
EXPERIMENT_RESULT_DIR = ROOT_DIR / "experimentation" / "output" / "result"

# I only use `np.random.RandomState` over `np.random.default_rng` because sklearn does
# not support the latter yet.
RNG = np.random.RandomState(10062930)

MODEL_FUNCTIONS = {
    "logreg": lambda cw: LogisticRegression(class_weight=cw, random_state=RNG),
    "svc_linear": lambda cw: LinearSVC(
        class_weight=cw,
        tol=1e-5,
        max_iter=1000,
        dual=False,
        random_state=RNG,
    ),
    "svc_rbf": lambda cw: SVC(
        class_weight=cw,
        kernel="rbf",  # Default kernel but want to emphasize.
        tol=1e-3,
        cache_size=1000,
        random_state=RNG,
    ),
}


def get_data():
    transaction_types: dict[TransactionType, dict] = {
        "CASH_OUT": {},
        "TRANSFER": {},
    }
    for transaction_type in transaction_types:
        transaction_types[transaction_type]["full"] = pd.read_csv(
            DATASET_DIR / "03_features" / f"{transaction_type}_full.csv",
            index_col=0,
        )

    data_TRANSFER = transaction_types["TRANSFER"]["full"]
    data_CASH_OUT = transaction_types["CASH_OUT"]["full"]

    X_TRANSFER, y_TRANSFER = (
        data_TRANSFER.loc[:, FEATURE_NAMES],
        data_TRANSFER.loc[:, [TARGET_NAME]],
    )

    X_CASH_OUT, y_CASH_OUT = (
        data_CASH_OUT.loc[:, FEATURE_NAMES],
        data_CASH_OUT.loc[:, [TARGET_NAME]],
    )

    y_TRANSFER, y_CASH_OUT = y_TRANSFER.values.ravel(), y_CASH_OUT.values.ravel()

    return {
        "TRANSFER": (X_TRANSFER, y_TRANSFER),
        "CASH_OUT": (X_CASH_OUT, y_CASH_OUT),
    }


def build():
    logging.info("Reading payments dataset...")
    txn_data = get_data()
    logging.info("Reading experiment ideal class weight results...")
    icw_result = pd.read_csv(EXPERIMENT_RESULT_DIR / "result_ideal_class_weight.csv")

    def build_icw_model(txn_type: TransactionType, model_name: ModelName):
        """Build the model with ideal class weight."""
        model_icw_data = icw_result.query(
            f"model_name == '{model_name}' and transaction_type == '{txn_type}'"
        )
        if model_icw_data.empty:
            logging.warning(
                f"SKIPPING: No ideal class weight found for model_name='{model_name}'."
            )
        icw_fraud = model_icw_data.iloc[0].class_weight
        model_fn = MODEL_FUNCTIONS[model_name]
        model = model_fn(cw={0: 1, 1: icw_fraud})

        X_txn, y_txn = txn_data[txn_type]
        model.fit(X_txn, y_txn)

        model_path = MODEL_DIR / f"model_{txn_type}_{model_name}.pkl"
        logging.info(f"Saving model '{model_name}' for '{txn_type}' @ '{model_path}'")
        with open(model_path, "wb") as f:
            dump(model, f)

    logging.info("Begin building models...")
    txn_models = [
        ("TRANSFER", "logreg"),
        ("CASH_OUT", "logreg"),
    ]
    Parallel(n_jobs=4)(
        delayed(build_icw_model)(txn_type, model_name)
        for txn_type, model_name in txn_models
    )
    # build_icw_model(txn_type="TRANSFER", model_name="logreg")
    # build_icw_model(txn_type="CASH_OUT", model_name="logreg")


def main(_):
    build()


if __name__ == "__main__":
    app.run(main)
