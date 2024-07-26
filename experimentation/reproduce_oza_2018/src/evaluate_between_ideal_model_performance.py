from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay

from absl import app, flags, logging

from model_config import FEATURE_NAMES, TARGET_NAME, MODEL_FUNCTIONS
import utils

mpl.rcParams["font.family"] = "Arial"

plt.style.use("bmh")

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

    def viz(X, y, label: str):
        fig, ax = plt.subplots(figsize=(5, 5))

        for model_name, fitted_model in fitted_models.items():
            if fitted_model is None:
                logging.warning(
                    f"SKIP: skip visualising model_name='{model_name}' as no fitted "
                    "model found."
                )
                continue

            logging.info(f"  RUN: Plotting PRC for model_name='{model_name}'...")
            if model_name in ["svc_rbf", "svc_rbf_sampler"]:
                model_with_type_dir = _get_model_with_type_dir(
                    model_name, FLAG.transaction_type
                )

                # Data scaling
                scaler = utils.load_model(model_with_type_dir / "standard_scaler.pkl")
                X_transformed = scaler.transform(X)

                # RBF sampling
                if model_name == "svc_rbf_sampler":
                    rbf_sampler = utils.load_model(
                        model_with_type_dir / "rbf_sampler.pkl"
                    )
                    X_transformed = rbf_sampler.transform(X_transformed)

                PrecisionRecallDisplay.from_estimator(
                    fitted_model, X_transformed, y, pos_label=1, ax=ax
                )
            else:
                PrecisionRecallDisplay.from_estimator(
                    fitted_model, X, y, pos_label=1, ax=ax
                )

        ax.set_title(
            f"Precision-Recall Curve for {FLAG.transaction_type} transactions "
            f"on {label} set"
        )

        fig.tight_layout()
        fig.savefig(
            RESULT_FIG_DIR
            / f"model_comparison_{FLAG.transaction_type}_{label.lower()}.png"
        )

    logging.info(
        "RUN: Visualising PRC for model with ideal class weights on TRAINING set..."
    )
    viz(X_train, y_train, label="TRAIN")
    logging.info(
        "RUN: Visualising PRC for model with ideal class weights on VALIDATION set..."
    )
    viz(X_val, y_val, label="VALIDATION")
    logging.info(
        "RUN: Visualising PRC for model with ideal class weights on TEST set..."
    )
    viz(X_test, y_test, label="TEST")


if __name__ == "__main__":
    app.run(main)
