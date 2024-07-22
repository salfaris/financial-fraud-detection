from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from absl import app, logging

# I only use `np.random.RandomState` over `np.random.default_rng` because sklearn does
# not support the latter yet.
RNG = np.random.RandomState(10062930)

ALL_EXPERIMENTS_ROOT_DIR = Path(__file__).parents[2]
ALL_EXPERIMENTS_DATA_DIR = ALL_EXPERIMENTS_ROOT_DIR / "datasets"
ALL_EXPERIMENTS_STAGED_DATA_DIR = ALL_EXPERIMENTS_DATA_DIR / "02_staged"
THIS_EXPERIMENT_ROOT_DIR = Path(__file__).resolve().parents[1]
THIS_EXPERIMENT_DATA_DIR = THIS_EXPERIMENT_ROOT_DIR / "datasets"
THIS_EXPERIMENT_STAGED_DATA_DIR = THIS_EXPERIMENT_DATA_DIR / "02_staged"

FEATURE_DATASET_DIR = THIS_EXPERIMENT_DATA_DIR / "03_features"

ALL_EXPERIMENTS_STAGED_DATA_DIR.mkdir(exist_ok=True, parents=True)
THIS_EXPERIMENT_STAGED_DATA_DIR.mkdir(exist_ok=True, parents=True)
FEATURE_DATASET_DIR.mkdir(exist_ok=True, parents=True)

FEATURE_NAMES = [
    "amount",
    "old_balance_Source",
    "new_balance_Source",
    "old_balance_Destination",
    "new_balance_Destination",
]
FEATURE_NAMES_WITH_DUMMIES = FEATURE_NAMES + [
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]

TARGET_NAME = "is_fraud"


def preprocess():
    raw_data = pd.read_csv(ALL_EXPERIMENTS_DATA_DIR / "01_raw" / "paysim.csv")

    data = raw_data.drop(columns=["isFlaggedFraud"])
    data = data.rename(
        columns={
            "nameOrig": "name_Source",
            "oldbalanceOrg": "old_balance_Source",
            "newbalanceOrig": "new_balance_Source",
            "nameDest": "name_Destination",
            "oldbalanceDest": "old_balance_Destination",
            "newbalanceDest": "new_balance_Destination",
            "isFraud": "is_fraud",
        }
    )
    # data.to_csv(STAGED_DATASET_DIR / "processed_paysim.csv")
    data_with_dummies = pd.get_dummies(data, columns=["type"])
    data_with_dummies.to_csv(
        THIS_EXPERIMENT_STAGED_DATA_DIR / "processed_paysim_unimodel.csv"
    )
    return data, data_with_dummies


def split_with_dummies(data: pd.DataFrame):
    X = data[FEATURE_NAMES_WITH_DUMMIES]
    y = data[TARGET_NAME]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=0.15,
        random_state=RNG,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        stratify=y_train,
        # 0.7 x N = trainSize x (1-0.15) x N ==> trainSize = 0.7 / (1-0.15).
        test_size=(1 - 0.7 / 0.85),
        random_state=RNG,
    )

    X_train["is_fraud"] = y_train
    X_val["is_fraud"] = y_val
    X_test["is_fraud"] = y_test

    X_train.to_csv(FEATURE_DATASET_DIR / "train_unimodel.csv", index=True)
    X_val.to_csv(FEATURE_DATASET_DIR / "val_unimodel.csv", index=True)
    X_test.to_csv(FEATURE_DATASET_DIR / "test_unimodel.csv", index=True)


def main(_):
    logging.info("RUN: Processing data.")
    data, data_with_dummies = preprocess()
    logging.info("RUN: Splitting dataset into train-test-split.")
    split_with_dummies(data_with_dummies)


if __name__ == "__main__":
    app.run(main)
