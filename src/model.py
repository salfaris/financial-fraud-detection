from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import precision_recall_fscore_support, recall_score

from absl import app, flags, logging

from config import FEATURE_NAMES, TARGET_NAME

flags.DEFINE_enum(
    "model_name",
    "logreg",
    ["logreg", "svc_linear", "svc_rbf"],
    "Model name to train and perform validation.",
)
flags.DEFINE_enum(
    "transaction_type", "TRANSFER", ["TRANSFER", "CASH_OUT"], "Transaction type."
)
FLAG = flags.FLAGS

# I only use `np.random.RandomState` over `np.random.default_rng` because sklearn does
# not support the latter yet.
RNG = np.random.RandomState(10062930)

ROOT_DIR = Path(__file__).parents[1]
DATASET_DIR = ROOT_DIR / "datasets"
RESULT_DIR = ROOT_DIR / "output" / "result"
RESULT_DIR.mkdir(exist_ok=True, parents=True)

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


def train(model_name: str, model_fn: callable):
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

    train = transaction_types[FLAG.transaction_type]["train"]
    val = transaction_types[FLAG.transaction_type]["val"]

    X_train, y_train = train.loc[:, FEATURE_NAMES], train.loc[:, [TARGET_NAME]]
    X_val, y_val = val.loc[:, FEATURE_NAMES], val.loc[:, [TARGET_NAME]]

    y_train, y_val = y_train.values.ravel(), y_val.values.ravel()

    metrics = {
        "precision": [],
        "recall": [],
        "f1-score": [],
        "false_positive_rate": [],
    }
    max_class_weight = 512
    fraud_class_weights = range(1, max_class_weight + 1)

    for fraud_weight in fraud_class_weights:
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

        # FalsePositiveRate = 1 - TrueNegativeRate = 1 - Recall(with 0 as target label)
        false_positive_rate = 1 - recall_score(y_true=y_val, y_pred=y_pred, pos_label=0)
        metrics["false_positive_rate"].append(false_positive_rate)

        logging.info(
            f"class_weight: {fraud_weight}, precision: {precision:.5f}, "
            f"recall: {recall:.5f}, F1-score: {f1_score:.5f}, "
            f"FPR: {false_positive_rate:.5f}"
        )

    metrics_df = pd.DataFrame(metrics)
    metrics_df.insert(0, "class_weight", list(fraud_class_weights))
    metrics_df.to_csv(
        RESULT_DIR / f"result_{FLAG.transaction_type.upper()}_{model_name}.csv",
        index=False,
    )


def main(_):
    train(model_name=FLAG.model_name, model_fn=MODEL_FUNCTIONS[FLAG.model_name])


if __name__ == "__main__":
    app.run(main)
