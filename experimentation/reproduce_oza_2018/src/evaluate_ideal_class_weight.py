from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from absl import app, flags, logging

from model_config import MODEL_FUNCTIONS
from plot_config import confplot

confplot()

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
RESULT_FIG_DIR = ROOT_DIR / "output" / "figures" / "result_class_weight"
RESULT_FIG_DIR.mkdir(exist_ok=True, parents=True)


def main(_):
    transaction_type: str = FLAG.transaction_type.upper()
    model_name: str = FLAG.model_name

    result_path = RESULT_DIR / f"result_{transaction_type}_{model_name}.csv"

    if not result_path.exists():
        raise FileNotFoundError(f"Cannot find result @ {result_path}")
    result = pd.read_csv(result_path)

    fpr_less_1_pct = result[result["false_positive_rate"] <= 0.01]

    # PRECISION: Out of all predicted fraud, how many are actually fraud?
    # Good for UX, high precision ==> low FPR
    #
    # RECALL: Out of all actual fraud, how many were predicted fraud?
    # Good for financials, high recall ==> catch more fraud ==> save money.
    #
    # Maximize RECALL since we've maintained low <1% FPR, now we want to catch all the
    # possible fraud.
    ideal_rowset = fpr_less_1_pct.iloc[fpr_less_1_pct["recall"].argmax()]
    logging.info(ideal_rowset)

    # Ideal class weight data for pair (`model_name`, `transaction_type`).
    ideal_cw_data = pd.DataFrame(ideal_rowset).T.reset_index(drop=True)
    ideal_cw_data["class_weight"] = ideal_cw_data["class_weight"].astype(int)
    ideal_cw_data.insert(0, "transaction_type", [transaction_type])
    ideal_cw_data.insert(0, "model_name", [model_name])

    full_ideal_cw_data_path = RESULT_DIR / "result_ideal_class_weight.csv"
    if not full_ideal_cw_data_path.exists():
        full_ideal_cw_data = ideal_cw_data
    else:
        full_ideal_cw_data = pd.read_csv(full_ideal_cw_data_path)

        sub_cw_df = full_ideal_cw_data.query(
            f"model_name == '{model_name}'"
            f"and transaction_type == '{transaction_type}'"
        )
        if sub_cw_df.empty:
            full_ideal_cw_data = pd.concat([full_ideal_cw_data, ideal_cw_data])
        elif sub_cw_df.shape[0] > 1:
            raise ValueError("Ideal class weight not meant to shape like this.")
        else:
            full_ideal_cw_data.iloc[sub_cw_df.index[0]] = ideal_cw_data.squeeze(axis=0)
    full_ideal_cw_data = full_ideal_cw_data.sort_values(
        by=["transaction_type", "model_name"]
    ).reset_index(drop=True)
    full_ideal_cw_data.to_csv(full_ideal_cw_data_path, index=False)

    def viz():
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(result.class_weight, result.precision, label="Precision", color="red")
        ax.plot(result.class_weight, result.recall, label="Recall", color="blue")

        ax.plot(
            result.class_weight,
            result["false_positive_rate"],
            label="FPR",
            color="orange",
        )
        ax.axvline(
            ideal_rowset.class_weight,
            linestyle="--",
            color="gray",
            label=f"Ideal class weight = {ideal_rowset.class_weight:.0f}",
        )

        ax.set_xlabel("Class weights for fraudulent transactions", fontweight="light")

        title_name_map = {
            "logreg": "Logistic Regression",
            "svc_linear": "SVM + linear kernel",
            "svc_rbf": "SVM + RBF kernel",
            "svc_rbf_sampler": "SVM + RBF sampler kernel",
            "decision_tree": "Decision Tree",
        }
        ax.set_title(
            "Metrics on validation set for "
            f"{' '.join(str(FLAG.transaction_type).split('_'))} transactions "
            f"- {title_name_map[model_name]}",
            # fontweight="bold",
        )

        ax.legend()

        fig.savefig(RESULT_FIG_DIR / f"result_{transaction_type}_{model_name}.png")

    viz()


if __name__ == "__main__":
    app.run(main)
