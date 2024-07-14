import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pathlib import Path

plt.style.use("bmh")

SRC_DIR = Path(__file__).parents[1]
DATASET_DIR = SRC_DIR / "datasets"
FIGURE_DIR = SRC_DIR / "output" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    data = pd.read_csv(DATASET_DIR / "02_staged" / "processed_paysim.csv")

    # Filter for transfer types where there is fraudulent transactions (i.e.
    # `is_fraud` == 1). Only consider these types to perform PCA.
    unique_transfer_types = data.groupby("is_fraud")["type"].value_counts().loc[1].index
    print(
        f"Detected {len(unique_transfer_types)} transfer types with fraudulent"
        f"transaction. These are: [{', '.join(unique_transfer_types)}]."
    )

    for transfer_type in unique_transfer_types:
        print(f"Performing PCA for type: {transfer_type}")
        transfer_type_data = data.query(f"type == '{transfer_type}'")

        X = transfer_type_data.iloc[:, :-2]
        X = X.drop(columns=["type", "name_Source", "name_Destination", "step"])
        X = StandardScaler().fit_transform(X)

        X_pca_transformed = PCA(n_components=2).fit_transform(X)
        pca_df = pd.DataFrame(
            data=X_pca_transformed,
            columns=["PC1", "PC2"],
            index=transfer_type_data.index,
        )
        pca_df["label"] = transfer_type_data["is_fraud"]
        pca_df["label"] = pca_df["label"].map(
            {0: "Non-fraud transaction", 1: "Fraud transaction"}
        )

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="label", ax=ax)
        ax.set_title(f"PCA for type: {transfer_type}")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"pca_{transfer_type}.png")


if __name__ == "__main__":
    main()
