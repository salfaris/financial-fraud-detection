from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from model_config import FEATURE_NAMES, TARGET_NAME

SRC_DIR = Path(__file__).parents[1]
DATASET_DIR = SRC_DIR / "datasets"
FEATURE_DATASET_DIR = DATASET_DIR / "03_features"
FEATURE_DATASET_DIR.mkdir(exist_ok=True, parents=True)

# I only use `np.random.RandomState` over `np.random.default_rng` because sklearn does
# not support the latter yet.
RNG = np.random.RandomState(seed=10062930)


def main():
    data = pd.read_csv(DATASET_DIR / "02_staged" / "processed_paysim.csv")

    # Filter for transfer types where there is fraudulent transactions (i.e.
    # `is_fraud` == 1). Only consider these types to perform PCA.
    unique_transaction_types = (
        data.groupby("is_fraud")["type"].value_counts().loc[1].index
    )
    print(
        f"Detected {len(unique_transaction_types)} transaction types with fraudulent"
        f"transactions. These are: [{', '.join(unique_transaction_types)}]."
    )

    for transaction_type in unique_transaction_types:
        print(
            f"Performing train-val-test split for transaction type '{transaction_type}'"
            "."
        )
        transaction_data = data.query(f"type == '{transaction_type}'")
        # NOTE: We reset the index after filtering on transaction type!
        #   This will be important to remember when we reindex things.
        transaction_data.reset_index(drop=True)
        X = transaction_data[FEATURE_NAMES]
        y = transaction_data[TARGET_NAME]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            stratify=y,
            test_size=0.15,
            random_state=RNG,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            stratify=y_train,
            # 0.7 x N = trainSize x (1-0.15) x N ==> trainSize = 0.7 / (1-0.15).
            test_size=(1 - 0.7 / 0.85),
            random_state=RNG,
        )

        X_train["is_fraud"] = y_train
        X_val["is_fraud"] = y_val
        X_test["is_fraud"] = y_test

        X_train.to_csv(
            FEATURE_DATASET_DIR / f"{transaction_type}_train.csv", index=True
        )
        X_val.to_csv(FEATURE_DATASET_DIR / f"{transaction_type}_val.csv", index=True)
        X_test.to_csv(FEATURE_DATASET_DIR / f"{transaction_type}_test.csv", index=True)


if __name__ == "__main__":
    main()
