from joblib import Parallel, delayed
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_fscore_support, recall_score
from sklearn.preprocessing import StandardScaler

# For SVC + RBF approximation
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier

from absl import app, flags, logging

from model_config import MODEL_FUNCTIONS, FEATURE_NAMES, TARGET_NAME
from utils import load_model, save_model

flags.DEFINE_enum(
    "model_name",
    "logreg",
    list(MODEL_FUNCTIONS.keys()),
    "Model name to train and perform validation.",
)
flags.DEFINE_enum(
    "transaction_type", "TRANSFER", ["TRANSFER", "CASH_OUT"], "Transaction type."
)
flags.DEFINE_integer("num_workers", 1, "Number of concurrently running workers.")
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
    model_subdir = MODEL_DIR / model_name / FLAG.transaction_type
    model_subdir.mkdir(exist_ok=True, parents=True)

    logging.info("Reading experiment datasets...")
    train = pd.read_csv(
        DATASET_DIR / "03_features" / f"{FLAG.transaction_type}_train.csv",
        index_col=0,
    )
    val = pd.read_csv(
        DATASET_DIR / "03_features" / f"{FLAG.transaction_type}_val.csv",
        index_col=0,
    )

    X_train, y_train = train.loc[:, FEATURE_NAMES], train.loc[:, [TARGET_NAME]]
    X_val, y_val = val.loc[:, FEATURE_NAMES], val.loc[:, [TARGET_NAME]]

    # Special preprocessing step if using SVC RBF (as per Oza's paper)
    if model_name in ["svc_rbf", "svc_rbf_sampler"]:

        # Downsample before scaling makes more sense in my mind.
        if FLAG.transaction_type == "CASH_OUT" and model_name == "svc_rbf":
            train_size = X_train.shape[0]
            keep_train_rate = 0.2
            logging.info(
                "ADD: Perform downsampling step to reduce train size "
                f"from {train_size:,} to {int(train_size*keep_train_rate):,}..."
            )
            from sklearn.model_selection import train_test_split

            X_train, _, y_train, _ = train_test_split(
                X_train,
                y_train,
                stratify=y_train,
                # Downsample to `keep_train_rate`% of the full 1.5+ million
                # CASH_OUT train set.
                train_size=keep_train_rate,
                random_state=RNG,
            )

        logging.info("ADD: standard scaling step since building SVC + RBF kernel...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        save_model(scaler, model_subdir / "standard_scaler.pkl")

        # Perform RBF kernel approximation using the Random Kitchen Sinks method.
        # if FLAG.transaction_type == "CASH_OUT" and model_name == "svc_rbf_sampler":
        if model_name == "svc_rbf_sampler":
            logging.info(
                "ADD: RBF kernel approximation step since building "
                "SVC + RBF kernel sampler for 'CASH_OUT' dataset..."
            )
            rbf_feature = RBFSampler(gamma="scale", random_state=RNG)
            X_train = rbf_feature.fit_transform(X_train)
            X_val = rbf_feature.transform(X_val)
            save_model(rbf_feature, model_subdir / "rbf_sampler.pkl")

    y_train, y_val = y_train.values.ravel(), y_val.values.ravel()

    max_class_weight = 512
    fraud_class_weights = range(1, max_class_weight + 1)

    # Model training
    def train_model(X, y, class_weight, transaction_type, model_path):
        # if model_name == "svc_rbf_sampler" and transaction_type == "CASH_OUT":
        if model_name == "svc_rbf_sampler":
            # Perform RBF kernel approximation using the Random Kitchen Sinks method.
            # Expect data to have passed through the Random Kitchen Sinks method.
            model = SGDClassifier(class_weight=class_weight, random_state=RNG)
        else:
            model = model_fn(class_weight)
        model.fit(X, y)
        print(
            f"DONE: Training model `{model_name}` for "
            f"class weight = {class_weight[1]}"
        )
        save_model(model, model_path)
        return model, class_weight

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
                f"SKIP: Model `{model_name}` already exists @ "
                f"{model_path.relative_to(ROOT_DIR)}."
            )
            continue
        else:
            logging.info(f"RUN: Adding training step for class weight = {fraud_weight}")
            fraud_class_weights_to_train.append(fraud_weight)
        class_weight = {0: 1, 1: fraud_weight}
        delayed_train_model = delayed(train_model)(
            X=X_train,
            y=y_train,
            class_weight=class_weight,
            transaction_type=FLAG.transaction_type,
            model_path=model_path,
        )
        delayed_train_models.append(delayed_train_model)

    if len(fraud_class_weights_to_train) > 0:
        logging.info(
            f"BEGIN: Training {len(fraud_class_weights_to_train)} models in parallel..."
        )
        trained_models = Parallel(n_jobs=FLAG.num_workers)(delayed_train_models)
        fraud_weights = list(map(lambda x: x[1][1], trained_models))
        logging.info("DONE: Done training models.")
    else:
        trained_models = []
        fraud_weights = []
        logging.info("SKIP: Found no models to train.")

    # Models that DO NOT EXIST (via path check) are trained in this run and are stored
    # as a (model, class_weight) tuple in `trained_models`.
    #
    # But we want to evaluate model metrics on all models 1...`max_class_weight`,
    # so we load models that DO EXIST (via path check) into its own 2-tuple list
    # `ready_models` and extend this list with `trained_models`.
    ready_models = []
    logging.info(
        f"RUN: Loading remaining {max_class_weight - len(fraud_class_weights_to_train)}"
        " models to be joined with trained models for metric evaluation..."
    )
    for weight in range(1, max_class_weight + 1):
        if weight not in fraud_weights:
            model = load_model(
                model_subdir
                / f"{model_name}_{FLAG.transaction_type}_CW{str(weight).zfill(3)}.pkl"
            )
            ready_models.append((model, {0: 1, 1: weight}))
    ready_models.extend(trained_models)
    ready_models = sorted(ready_models, key=lambda x: x[1][1])

    def compute_model_metrics(model, class_weight):
        fraud_weight = class_weight[1]
        y_pred = model.predict(X_val.copy())
        (
            (_, precision),
            (_, recall),
            (_, f1_score),
            (_, _),
        ) = precision_recall_fscore_support(y_true=y_val, y_pred=y_pred, average=None)
        false_positive_rate = 1 - recall_score(y_true=y_val, y_pred=y_pred, pos_label=0)

        metrics = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1_score,
            "false_positive_rate": false_positive_rate,
        }
        print(f"RUN: Done compute metrics for {fraud_weight = }")
        return fraud_weight, metrics

    logging.info("BEGIN: Computing model metrics in parallel...")
    model_metrics = Parallel(n_jobs=FLAG.num_workers)(
        delayed(compute_model_metrics)(model, class_weight)
        for model, class_weight in ready_models
    )

    report = {
        "class_weight": [],
        "precision": [],
        "recall": [],
        "f1-score": [],
        "false_positive_rate": [],
    }
    for fraud_weight, weight_metrics in model_metrics:
        precision = weight_metrics["precision"]
        recall = weight_metrics["recall"]
        f1_score = weight_metrics["f1-score"]
        false_positive_rate = weight_metrics["false_positive_rate"]

        report["class_weight"].append(fraud_weight)
        report["precision"].append(precision)
        report["recall"].append(recall)
        report["f1-score"].append(f1_score)
        report["false_positive_rate"].append(false_positive_rate)
        logging.info(
            f"class_weight: {fraud_weight}, precision: {precision:.5f}, "
            f"recall: {recall:.5f}, F1-score: {f1_score:.5f}, "
            f"FPR: {false_positive_rate:.5f}"
        )

    report_df = pd.DataFrame(report)
    report_df.sort_values(by="class_weight", inplace=True)
    report_df.to_csv(
        RESULT_DIR / f"result_{FLAG.transaction_type.upper()}_{model_name}.csv",
        index=False,
    )


def main(_):
    train(model_name=FLAG.model_name, model_fn=MODEL_FUNCTIONS[FLAG.model_name])


if __name__ == "__main__":
    app.run(main)
