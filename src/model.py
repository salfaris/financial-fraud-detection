from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support

# I only use `np.random.RandomState` over `np.random.default_rng` because sklearn does
# not support the latter yet.
RNG = np.random.RandomState(10062930)

ROOT_DIR = Path(__file__).parents[1]
DATASET_DIR = ROOT_DIR / "datasets"
RESULT_DIR = ROOT_DIR / "output" / "report"
RESULT_DIR.mkdir(exist_ok=True, parents=True)

FEATURE_NAMES = [
    "amount",
    "old_balance_Source",
    "new_balance_Source",
    "old_balance_Destination",
    "new_balance_Destination",
]
TARGET_NAME = "is_fraud"


def train(model_name, model_fn):
    transaction_types = {
        "CASH_OUT": {},
        "TRANSFER": {},
    }
    for transaction_type in transaction_types:
        transaction_types[transaction_type]["train"] = pd.read_csv(
            DATASET_DIR / "03_features" / f"{transaction_type}_train.csv",
            index_col=0,
        )
        transaction_types[transaction_type]["val"] = pd.read_csv(
            DATASET_DIR / "03_features" / f"{transaction_type}_val.csv",
            index_col=0,
        )
        transaction_types[transaction_type]["test"] = pd.read_csv(
            DATASET_DIR / "03_features" / f"{transaction_type}_test.csv",
            index_col=0,
        )

    # NOTE :- Do for `TRANSFER` only atm.
    train = transaction_types["TRANSFER"]["train"]
    val = transaction_types["TRANSFER"]["val"]
    test = transaction_types["TRANSFER"]["test"]

    X_train, y_train = train.loc[:, FEATURE_NAMES], train.loc[:, [TARGET_NAME]]
    X_val, y_val = val.loc[:, FEATURE_NAMES], val.loc[:, [TARGET_NAME]]
    X_test, y_test = test.loc[:, FEATURE_NAMES], test.loc[:, [TARGET_NAME]]

    y_train, y_val, y_test = (
        y_train.values.ravel(),
        y_val.values.ravel(),
        y_test.values.ravel(),
    )

    metrics = {
        "precision": [],
        "recall": [],
        "f1-score": [],
    }
    max_class_weight = 512
    fraud_class_weights = range(1, max_class_weight + 1)

    for fraud_weight in fraud_class_weights:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]", end=" ")
        print(f"Training+validating {fraud_weight = }")
        class_weight = {0: 1, 1: fraud_weight}

        # Model training
        model = model_fn(class_weight)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        (
            (_, precision),
            (_, recall),
            (_, f1_score),
            (_, _),
        ) = precision_recall_fscore_support(y_true=y_val, y_pred=y_pred, average=None)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1-score"].append(f1_score)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.insert(0, "class_weight", list(fraud_class_weights))
    metrics_df.to_csv(RESULT_DIR / "result_{model_name}.csv", index=False)


if __name__ == "__main__":
    model_fns = {
        "logreg": lambda cw: LogisticRegression(class_weight=cw, random_state=RNG),
        "svc_linear": lambda cw: LinearSVC(
            class_weight=cw,
            tol=1e-5,
            max_iter=1000,
            dual=False,
            random_state=RNG,
        ),
    }
    train(model_fns["logreg"])
