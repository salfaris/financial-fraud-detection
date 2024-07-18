import pandas as pd

from pathlib import Path

SRC_DIR = Path(__file__).parents[1]
DATASET_DIR = SRC_DIR / "datasets"


def main():
    raw_data = pd.read_csv(DATASET_DIR / "01_raw" / "paysim.csv")
    raw_data = raw_data.rename(
        columns={
            "nameOrig": "name_Source",
            "oldbalanceOrg": "old_balance_Source",
            "newbalanceOrig": "new_balance_Source",
            "nameDest": "name_Destination",
            "oldbalanceDest": "old_balance_Destination",
            "newbalanceDest": "new_balance_Destination",
            "isFraud": "is_fraud",
            "isFlaggedFraud": "is_flagged_fraud",
        }
    )
    raw_data.to_csv(DATASET_DIR / "02_staged" / "processed_paysim.csv", index=False)


if __name__ == "__main__":
    main()
