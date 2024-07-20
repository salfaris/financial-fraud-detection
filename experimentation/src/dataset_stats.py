from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).parents[1]
DATASET_DIR = SRC_DIR / "datasets"
OUT_DIR = SRC_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    data = pd.read_csv(DATASET_DIR / "02_staged" / "processed_paysim.csv")

    paysim_data_stats = (
        data.groupby(["type"])["is_fraud"].value_counts().unstack(fill_value=0)
    )

    # Compute sum across fraudulent and non-fraudulent transactions.
    paysim_data_stats["total_transactions"] = paysim_data_stats.sum(axis=1)
    paysim_data_stats.rename(
        columns={0: "non_fraud_transactions", 1: "fraud_transactions"}, inplace=True
    )
    paysim_data_stats.sort_index(inplace=True)

    # Compute sum across all transaction types (CASH IN, CASH OUT, TRANSFER, etc.).
    paysim_data_stats.loc["TOTAL_ALL_TYPE"] = paysim_data_stats.sum(axis=0)

    paysim_data_stats.reset_index(inplace=True)
    paysim_data_stats.columns.name = ""
    paysim_data_stats.to_csv(OUT_DIR / "paysim_dataset_statistics.csv", index=False)


if __name__ == "__main__":
    main()
