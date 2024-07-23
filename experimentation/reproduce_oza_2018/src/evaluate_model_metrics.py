from joblib import Parallel, delayed, dump, load
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_fscore_support, recall_score

from absl import app, flags, logging

from model_config import MODEL_FUNCTIONS, FEATURE_NAMES, TARGET_NAME

flags.DEFINE_enum(
    "model_name",
    "logreg",
    list(MODEL_FUNCTIONS.keys()),
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
MODEL_DIR = ROOT_DIR / "output" / "model"
RESULT_DIR.mkdir(exist_ok=True, parents=True)


def train(model_name: str, model_fn: callable):
    logging.info("Reading experiment datasets...")
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
    max_class_weight = 10
    fraud_class_weights = range(1, max_class_weight + 1)

    def load_model(model_path: Path):
        return load(model_path)

    def save_model(model, model_path: Path):
        with open(model_path, "wb") as f:
            dump(model, f)

    # Model training
    def train_model(X, y, class_weight, model_path):
        model = model_fn(class_weight)
        model.fit(X, y)
        print(
            f"DONE: Training model `{model_name}` for "
            f"class weight = {class_weight[1]}"
        )
        save_model(model, model_path)
        return model, class_weight

    model_subdir = MODEL_DIR / model_name / FLAG.transaction_type
    model_subdir.mkdir(exist_ok=True, parents=True)
    logging.info(
        "RUN: Adding training step to queue for all "
        f"class weights 1...{max_class_weight}"
    )
    fraud_class_weights_to_train: list[int] = []
    delayed_train_models = []
    for fraud_weight in fraud_class_weights:
        model_path = (
            model_subdir
            / f"{model_name}_{FLAG.transaction_type}_CW{str(fraud_weight).zfill(3)}.pkl"
        )
        if model_path.exists():
            logging.info(
                f"SKIP: Model `{model_name}` already exists @ {model_path.relative_to(ROOT_DIR)}."
            )
            continue
        else:
            logging.info(f"RUN: Adding training step for class weight = {fraud_weight}")
            fraud_class_weights_to_train.append(fraud_weight)
        class_weight = {0: 1, 1: fraud_weight}
        delayed_train_model = delayed(train_model)(
            X_train, y_train, class_weight, model_path
        )
        delayed_train_models.append(delayed_train_model)
    logging.info(
        f"BEGIN: Training {len(fraud_class_weights_to_train)} models in parallel..."
    )
    trained_models = Parallel(n_jobs=6)(delayed_train_models)

    logging.info("DONE: Done training models. Ready to compute metrics.")

    ready_models = []
    fraud_weights = list(map(lambda x: x[1][1], trained_models))
    for weight in range(1, max_class_weight + 1):
        if weight not in fraud_weights:
            model = load_model(
                model_subdir
                / f"{model_name}_{FLAG.transaction_type}_CW{str(weight).zfill(3)}.pkl"
            )
            ready_models.append((model, {0: 1, 1: weight}))
    ready_models.extend(trained_models)
    ready_models = sorted(ready_models, key=lambda x: x[1][1])

    for model, class_weight in ready_models:
        fraud_weight = class_weight[1]

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
