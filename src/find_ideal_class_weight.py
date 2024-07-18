from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from absl import app, flags, logging

plt.style.use("bmh")

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
RESULT_FIG_DIR = ROOT_DIR / "output" / "figures" / "result_class_weight"
RESULT_FIG_DIR.mkdir(exist_ok=True, parents=True)


def main(_):
    result_path = (
        RESULT_DIR / f"result_{FLAG.transaction_type.upper()}_{FLAG.model_name}.csv"
    )
    print(result_path)
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
    ideal = fpr_less_1_pct.iloc[fpr_less_1_pct["recall"].argmax()]
    logging.info(ideal)

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
            ideal.class_weight,
            linestyle="--",
            color="gray",
            label=f"Ideal class weight = {ideal.class_weight:.0f}",
        )

        ax.set_xlabel("Class weights for fraudulent transactions", fontweight="light")

        title_name_map = {
            "logreg": "Logistic Regression",
            "svc_linear": "SVM + linear kernel",
            "svc_rbf": "SVM + RBF kernel",
        }
        ax.set_title(
            f"Metrics on validation set for {FLAG.transaction_type} transactions "
            f"- {title_name_map[FLAG.model_name]}",
            # fontweight="bold",
        )

        ax.legend()

        fig.savefig(
            RESULT_FIG_DIR
            / f"result_{FLAG.transaction_type.upper()}_{FLAG.model_name}.png"
        )

    viz()


if __name__ == "__main__":
    app.run(main)
