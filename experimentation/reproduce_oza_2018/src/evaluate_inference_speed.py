from pathlib import Path
import timeit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
RESULT_DATA_DIR = ROOT_DIR / "output" / "result" / "inference_speed"
RESULT_FIG_DIR = ROOT_DIR / "output" / "figures" / "result_inference_speed"
RESULT_DATA_DIR.mkdir(exist_ok=True, parents=True)
RESULT_FIG_DIR.mkdir(exist_ok=True, parents=True)

SKIP_MODELS = [
    # "svc_rbf",
    "svc_rbf_sampler",
]

RNG = np.random.RandomState(10062930)


class ModelNotFoundError(Exception):
    pass


def _get_model_with_type_dir(model_name: str, transaction_type: str) -> Path:
    model_with_type_dir = MODEL_DIR / model_name / transaction_type
    model_with_type_dir.mkdir(exist_ok=True, parents=True)
    return model_with_type_dir


def main(_):
    logging.info("BEGIN: Reading ideal class weight data...")
    icw_data = pd.read_csv(RESULT_DIR / "result_ideal_class_weight.csv")

    logging.info(
        f"RUN: Loading {FLAG.transaction_type} train, val, test data into memory..."
    )

    val = pd.read_csv(
        DATASET_DIR / "03_features" / f"{FLAG.transaction_type}_val.csv", index_col=0
    )

    X_val, y_val = val.loc[:, FEATURE_NAMES], val.loc[:, [TARGET_NAME]]
    y_val = y_val.values.ravel()

    fitted_models = {model_name: None for model_name in MODEL_FUNCTIONS}
    for model_name, _ in MODEL_FUNCTIONS.items():
        if model_name in SKIP_MODELS:
            continue
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
                f"BUILD: Loading model '{model_name}' from "
                f"disk @ {model_with_type_path.relative_to(ROOT_DIR)}..."
            )
            model = utils.load_model(model_with_type_path)
        else:
            raise ModelNotFoundError(f"TERMINATE: Model {model_name} not fitted.")
        fitted_models[model_name] = model

    sample_size_space = [1, 5, 10, 25, 50, 100]
    repeats_result: dict[str, dict[int, np.ndarray | None]] = {
        model_name: {i: None for i in sample_size_space}
        for model_name in MODEL_FUNCTIONS
    }

    def compute(X, y, sample_size: int):
        for model_name, fitted_model in fitted_models.items():
            logging.info(f"Computing metrics - {model_name}...")
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

            result_path = (
                RESULT_DATA_DIR
                / f"{model_name}_{FLAG.transaction_type}_sampleSize{sample_size}.npy"
            )
            if result_path.exists():
                with open(result_path, "rb") as f:
                    repeats = np.load(f)

            else:
                num_samples = 1000
                num_experiment_repeats = 100

                repeats = timeit.repeat(
                    lambda: fitted_model.predict(X_transformed),
                    number=num_samples,
                    repeat=num_experiment_repeats,
                )
                repeats = np.array(repeats)

                with open(result_path, "wb") as f:
                    np.save(f, repeats)

            logging.info(
                f"{model_name}: {repeats.mean():.4f} ± {repeats.std(ddof=1):.4f}"
            )

            repeats_result[model_name][sample_size] = repeats

    def viz(sample_size: int):
        fig, ax = plt.subplots(figsize=(10, 5))

        model_names = []
        sample_size_runs = []
        for model_name, runs in repeats_result.items():
            run = runs[sample_size]
            if run is None:
                logging.info(
                    f"SKIP: histplot for '{model_name}' because no data found."
                )
                continue
            sample_size_runs.append(run)
            model_names.append(model_name)

        model_tick_label_map = {
            "logreg": "Logistic Regression",
            "svc_linear": "SVM + linear kernel",
            "svc_rbf": "SVM + RBF kernel",
            "svc_rbf_sampler": "SVM + RBF sampler kernel",
            "decision_tree": "Decision Tree",
        }

        for run, model_name in zip(sample_size_runs, model_names):
            sns.kdeplot(
                run,
                label=model_tick_label_map[model_name],
                fill=True,
                bw_adjust=2.0,
                log_scale=True,
                gridsize=500,
            )
        ax.set_xlim(-10, 100)
        ax.legend()
        ax.set_xlabel("Inference time (seconds)")
        ttype = " ".join(str(FLAG.transaction_type).split("_"))
        ax.set_title(
            "Inference time comparison between models on"
            f" n={sample_size} {ttype} transactions"
        )

        fig.tight_layout()
        fig.savefig(RESULT_FIG_DIR / f"hist_{FLAG.transaction_type}_{sample_size}.png")

    for sample_size in sample_size_space:
        logging.info(f"Evaluating {sample_size = }")
        idx = RNG.randint(0, X_val.shape[0], sample_size)
        compute(X_val.iloc[idx], y_val[idx], sample_size=sample_size)
        viz(sample_size)

    stat_data = []
    for model_name, runs in repeats_result.items():
        for sample_size in sample_size_space:
            # for sample_size in [1, 10, 100]:
            run = runs[sample_size]
            if run is None:
                logging.info(
                    f"SKIP: statistics for '{model_name}' because no data found."
                )
                continue
            mean = np.round(run.mean(), 2)
            stddev = np.round(run.std(ddof=1), 2)
            stat_data.append(
                (model_name, sample_size, FLAG.transaction_type, mean, stddev)
            )

    stat_data = pd.DataFrame(
        stat_data,
        columns=[
            "model_name",
            "sample_size",
            "transaction_type",
            "inf_time_mean",
            "inf_time_std",
        ],
    )
    stat_data.sort_values(by=["sample_size", "model_name"], inplace=True)
    print(stat_data)

    stat_data.to_csv(
        RESULT_DATA_DIR / f"inference_time_stats_{FLAG.transaction_type}.csv",
        index=False,
    )


if __name__ == "__main__":
    app.run(main)
