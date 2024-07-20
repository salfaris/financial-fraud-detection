from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay

from absl import app, flags, logging

from model_config import FEATURE_NAMES, TARGET_NAME, MODEL_FUNCTIONS

mpl.rcParams["font.family"] = "Arial"

plt.style.use("bmh")

flags.DEFINE_enum(
    "transaction_type", "TRANSFER", ["TRANSFER", "CASH_OUT"], "Transaction type."
)
FLAG = flags.FLAGS

ROOT_DIR = Path(__file__).parents[1]
DATASET_DIR = ROOT_DIR / "datasets"
RESULT_DIR = ROOT_DIR / "output" / "result"
RESULT_FIG_DIR = ROOT_DIR / "output" / "figures" / "result_model_comparison"
RESULT_FIG_DIR.mkdir(exist_ok=True, parents=True)


def main(_):
    icw_data = pd.read_csv(RESULT_DIR / "result_ideal_class_weight.csv")

    transaction_types = {
        "CASH_OUT": {},
        "TRANSFER": {},
    }
    for transaction_type in transaction_types:
        transaction_types[transaction_type]["train"] = pd.read_csv(
            DATASET_DIR / "03_features" / f"{transaction_type}_train.csv", index_col=0
        )
        transaction_types[transaction_type]["val"] = pd.read_csv(
            DATASET_DIR / "03_features" / f"{transaction_type}_val.csv", index_col=0
        )
        transaction_types[transaction_type]["test"] = pd.read_csv(
            DATASET_DIR / "03_features" / f"{transaction_type}_test.csv", index_col=0
        )

    train = transaction_types[FLAG.transaction_type]["train"]
    val = transaction_types[FLAG.transaction_type]["val"]
    test = transaction_types[FLAG.transaction_type]["test"]

    X_train, y_train = train.loc[:, FEATURE_NAMES], train.loc[:, [TARGET_NAME]]
    X_val, y_val = val.loc[:, FEATURE_NAMES], val.loc[:, [TARGET_NAME]]
    X_test, y_test = test.loc[:, FEATURE_NAMES], test.loc[:, [TARGET_NAME]]

    y_train, y_val, y_test = (
        y_train.values.ravel(),
        y_val.values.ravel(),
        y_test.values.ravel(),
    )

    fitted_models = {model_name: None for model_name in MODEL_FUNCTIONS}
    for model_name, model_fn in MODEL_FUNCTIONS.items():
        model_icw_data = icw_data.query(
            f"model_name == '{model_name}' and "
            f"transaction_type == '{FLAG.transaction_type}'"
        )
        if model_icw_data.empty:
            logging.warning(
                f"SKIPPING: No ideal class weight found for model_name='{model_name}'."
            )
            continue
        icw = model_icw_data.iloc[0].class_weight
        model = model_fn(cw={0: 1, 1: icw})
        model.fit(X_train, y_train)  # model was always trained on training set.
        fitted_models[model_name] = model

        # SAVE MODEL PARAMS HERE!

    def viz():
        fig, ax = plt.subplots(figsize=(5, 5))

        for model_name, fitted_model in fitted_models.items():
            if fitted_model is None:
                logging.warning(
                    f"SKIPPING: No fitted model found for model_name='{model_name}'."
                )
                continue
            PrecisionRecallDisplay.from_estimator(
                fitted_model, X_val, y_val, pos_label=1, ax=ax
            )

        ax.set_title(f"Precision-Recall Curve for {FLAG.transaction_type} transactions")

        fig.tight_layout()
        fig.savefig(RESULT_FIG_DIR / f"model_comparison_{FLAG.transaction_type}.png")

    viz()


if __name__ == "__main__":
    app.run(main)
