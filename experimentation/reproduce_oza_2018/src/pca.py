import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pathlib import Path

from absl import app, logging

from plot_config import confplot

confplot()

EXPERIMENTATION_DIR = Path(__file__).parents[2]
REPRODUCE_DIR = Path(__file__).parents[1]
DATASET_DIR = EXPERIMENTATION_DIR / "datasets"
PCA_FIGURE_DIR = REPRODUCE_DIR / "output" / "figures" / "pca"
PCA_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def main(_):
    data = pd.read_csv(DATASET_DIR / "02_staged" / "processed_paysim.csv")

    # Filter for transfer types where there is fraudulent transactions (i.e.
    # `is_fraud` == 1). Only consider these types to perform PCA.
    unique_transfer_types = data.groupby("is_fraud")["type"].value_counts().loc[1].index
    logging.info(
        f"Detected {len(unique_transfer_types)} transfer types with fraudulent "
        f"transaction. These are: [{', '.join(unique_transfer_types)}]."
    )

    for transfer_type in unique_transfer_types:
        logging.info(f"Performing PCA for type: {transfer_type}")
        transfer_type_data = data.query(f"type == '{transfer_type}'")

        X = transfer_type_data.iloc[:, :-2]
        X = X.drop(columns=["type", "name_Source", "name_Destination", "step"])
        X = StandardScaler().fit_transform(X)

        X_pca_transformed = PCA(n_components=2).fit_transform(X)
        pca_df = pd.DataFrame(
            data=X_pca_transformed,
            columns=["PC 1", "PC 2"],
            index=transfer_type_data.index,
        )
        pca_df["label"] = transfer_type_data["is_fraud"]
        pca_df["label"] = pca_df["label"].map(
            {0: "Non-fraudulent transaction", 1: "Fraudulent transaction"}
        )

        def visualize_and_save_pca():
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.scatterplot(
                data=pca_df, x="PC 1", y="PC 2", hue="label", ax=ax, alpha=0.5
            )
            ax.set_title(f"PCA for {' '.join(transfer_type.split('_'))} transactions")
            fig.tight_layout()
            fig.savefig(PCA_FIGURE_DIR / f"pca_{transfer_type}.png")

        visualize_and_save_pca()


if __name__ == "__main__":
    app.run(main)
