from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from absl import app, flags, logging

from model_config import FEATURE_NAMES, TARGET_NAME, MODEL_FUNCTIONS
from plot_config import confplot
import utils

confplot()

flags.DEFINE_enum(
    "transaction_type", "TRANSFER", ["TRANSFER", "CASH_OUT"], "Transaction type."
)
FLAG = flags.FLAGS

ROOT_DIR = Path(__file__).parents[1]
DATASET_DIR = ROOT_DIR / "datasets"
MODEL_DIR = ROOT_DIR / "output" / "model"
RESULT_DIR = ROOT_DIR / "output" / "result"
RESULT_FIG_DIR = ROOT_DIR / "output" / "figures" / "result_model_comparison"
RESULT_FIG_DIR.mkdir(exist_ok=True, parents=True)

SKIP_MODELS = [
    # "svc_rbf",
    "svc_rbf_sampler",
]


def main(_):
    logging.info("BEGIN: Reading ideal class weight data...")
    icw_data = pd.read_csv(RESULT_DIR / "result_ideal_class_weight.csv")

    logging.info(
        f"RUN: Loading {FLAG.transaction_type} train, val, test data into memory..."
    )
    train = pd.read_csv(
        DATASET_DIR / "03_features" / f"{FLAG.transaction_type}_train.csv", index_col=0
    )
    val = pd.read_csv(
        DATASET_DIR / "03_features" / f"{FLAG.transaction_type}_val.csv", index_col=0
    )
    test = pd.read_csv(
        DATASET_DIR / "03_features" / f"{FLAG.transaction_type}_test.csv", index_col=0
    )

    X_train, y_train = train.loc[:, FEATURE_NAMES], train.loc[:, [TARGET_NAME]]
    X_val, y_val = val.loc[:, FEATURE_NAMES], val.loc[:, [TARGET_NAME]]
    X_test, y_test = test.loc[:, FEATURE_NAMES], test.loc[:, [TARGET_NAME]]

    y_train, y_val, y_test = (
        y_train.values.ravel(),
        y_val.values.ravel(),
        y_test.values.ravel(),
    )

    def _get_model_with_type_dir(model_name: str, transaction_type: str) -> Path:
        model_with_type_dir = MODEL_DIR / model_name / transaction_type
        model_with_type_dir.mkdir(exist_ok=True, parents=True)
        return model_with_type_dir

    fitted_models = {model_name: None for model_name in MODEL_FUNCTIONS}
    for model_name, model_fn in MODEL_FUNCTIONS.items():
        if model_name in SKIP_MODELS:
            continue
        logging.info(f"BUILD: Building model '{model_name}'...")
        model_icw_data = icw_data.query(
            f"model_name == '{model_name}' and "
            f"transaction_type == '{FLAG.transaction_type}'"
        )
        if model_icw_data.empty:
            logging.warning(
                f"SKIP: No ideal class weight found for model_name='{model_name}'."
            )
            continue
        icw = model_icw_data.iloc[0].class_weight

        path_model_name = model_name
        if model_name == "svc_rbf_sampler":
            path_model_name = (
                "svc_rbf" if FLAG.transaction_type == "CASH_OUT" else "svc_rbf_sampler"
            )
        model_with_type_path = (
            _get_model_with_type_dir(model_name, FLAG.transaction_type)
            / f"{path_model_name}_{FLAG.transaction_type}_CW{str(icw).zfill(3)}.pkl"
        )
        if model_with_type_path.exists():
            logging.info(
                f"  BUILD: Model already exists. Loading model '{model_name}' from "
                f"disk @ {model_with_type_path.relative_to(ROOT_DIR)}..."
            )
            model = utils.load_model(model_with_type_path)
        else:
            model = model_fn(cw={0: 1, 1: icw})
            model.fit(X_train, y_train)  # model was always trained on training set.
            utils.save_model(model, model_with_type_path)
        fitted_models[model_name] = model

    def compute(X, y, label: str, metrics: dict[str, list] | None = None):
        model_precision_recall_auprc: list[
            tuple[str, np.ndarray, np.ndarray, float]
        ] = []
        if metrics is None:
            metrics = {
                "data_set": [],
                "model_name": [],
                "transaction_type": [],
                "class_weight": [],
                "precision": [],
                "recall": [],
                "f1-score": [],
                "AUPRC": [],
                "confusion_matrix": [],
            }
        for model_name, fitted_model in fitted_models.items():
            logging.info(f"Computing {label} metrics - {model_name}...")
            if fitted_model is None:
                continue

            if model_name in ["svc_rbf", "svc_rbf_sampler"]:
                model_with_type_dir = _get_model_with_type_dir(
                    model_name, FLAG.transaction_type
                )

                # Data scaling
                scaler = utils.load_model(model_with_type_dir / "standard_scaler.pkl")
                X_transformed = scaler.transform(X.copy())

                # RBF sampling
                if model_name == "svc_rbf_sampler":
                    rbf_sampler = utils.load_model(
                        model_with_type_dir / "rbf_sampler.pkl"
                    )
                    X_transformed = rbf_sampler.transform(X_transformed)
            else:
                # Id transformation
                X_transformed = X

            try:
                y_score = fitted_model.predict_proba(X_transformed)
                y_score = y_score[:, 1]
            except AttributeError:
                y_score = fitted_model.decision_function(X_transformed)

            precision, recall, _ = precision_recall_curve(y, y_score, pos_label=1)
            auc_precision_recall = auc(recall, precision)

            # # Use the 0.5 threshold
            # y_pred = np.array(
            #     list(map(lambda score: 1 if score >= 0.5 else 0, y_score))
            # )
            y_pred = fitted_model.predict(X_transformed)

            precision_single = precision_score(y, y_pred)
            recall_single = recall_score(y, y_pred)
            f1_score_single = f1_score(y, y_pred)

            model_precision_recall_auprc.append(
                (model_name, precision, recall, auc_precision_recall)
            )

            conf_matrix = confusion_matrix(y, y_pred)  # can ravel because binary

            metrics["data_set"].append(label)
            metrics["model_name"].append(model_name)
            metrics["transaction_type"].append(FLAG.transaction_type)
            metrics["class_weight"].append(
                fitted_model.get_params()["class_weight"][1]
            )  # class_weight is a dict {0: 1, 1: `fraud_weight`}
            metrics["precision"].append(precision_single)
            metrics["recall"].append(recall_single)
            metrics["f1-score"].append(f1_score_single)
            metrics["AUPRC"].append(auc_precision_recall)
            metrics["confusion_matrix"].append(conf_matrix)

        return metrics, model_precision_recall_auprc

    def viz(model_precision_recall_auprc, dataset_label: str):
        fig, ax = plt.subplots(figsize=(6, 6))

        title_name_map = {
            "logreg": "Logistic Regression",
            "svc_linear": "SVM + linear kernel",
            "svc_rbf": "SVM + RBF kernel",
            "svc_rbf_sampler": "SVM + RBF sampler kernel",
            "decision_tree": "Decision Tree",
        }
        for model_name, precision, recall, auprc in model_precision_recall_auprc:
            ax.plot(
                recall,
                precision,
                label=f"{title_name_map[model_name]} (AUPRC = {auprc:.4f})",
                linestyle="--",
            )

        ax.set_title(
            "Precision-Recall Curve for \n"
            f"{FLAG.transaction_type} transactions on {dataset_label} set"
        )
        ax.set_ylabel("Precision")
        ax.set_xlabel("Recall")
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            RESULT_FIG_DIR
            / f"model_comparison_{FLAG.transaction_type}_{dataset_label.lower()}.png"
        )

    logging.info(
        "\nRUN: Visualising PRC for model with ideal class weights on TRAINING set..."
    )
    dataset_label = "TRAIN"
    metrics, model_precision_recall_auprc = compute(
        X_train, y_train, label=dataset_label
    )
    viz(model_precision_recall_auprc, dataset_label=dataset_label)

    logging.info(
        "\nRUN: Visualising PRC for model with ideal class weights on VALIDATION set..."
    )
    dataset_label = "VALIDATION"
    metrics, model_precision_recall_auprc = compute(
        X_val, y_val, label=dataset_label, metrics=metrics
    )
    viz(model_precision_recall_auprc, dataset_label=dataset_label)

    logging.info(
        "\nRUN: Visualising PRC for model with ideal class weights on TEST set..."
    )
    dataset_label = "TEST"
    metrics, model_precision_recall_auprc = compute(
        X_test, y_test, label=dataset_label, metrics=metrics
    )
    viz(model_precision_recall_auprc, dataset_label=dataset_label)

    pd.DataFrame(metrics).to_csv(
        RESULT_DIR / f"result_auprc_{FLAG.transaction_type}.csv", index=False
    )


if __name__ == "__main__":
    app.run(main)
