# scripts/train-test_split.py
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from src.myproject.utils.paths import CONFIG_PATH, CLEANED_DATA, PREPROCESSED_DATA

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

SEED = config["train_test_split"]["random_state"]
TEST_SIZE = config["train_test_split"]["test_size"]

def train_test_split_cleaned_data(cleaned_data_path: str):
    # Load cleaned data
    cleaned_df = pd.read_csv(cleaned_data_path, index_col=0)
    PROC = Path(PREPROCESSED_DATA)
    PROC.mkdir(parents=True, exist_ok=True)

    # Drop `customerID`
    cleaned_df.drop(columns=['customerID'], errors='ignore', inplace=True)

    # Build features/target
    target_col = "Churn"
    feature_cols = [c for c in cleaned_df.columns if c != target_col]

    # Train-test split
    train_df, test_df = train_test_split(
        cleaned_df, test_size=TEST_SIZE, stratify=cleaned_df[target_col], random_state=SEED
    )
    train_df, val_df  = train_test_split(train_df, test_size=TEST_SIZE, stratify=train_df[target_col], random_state=SEED)

    # Save splits (target included)
    num_cols = train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if "SeniorCitizen" in num_cols:  # treat as categorical if needed
        num_cols.remove("SeniorCitizen")

    cat_cols = [c for c in feature_cols if c not in num_cols]

    # Save metadata
    meta = {
        "version": "v1",
        "target": target_col,
        "numeric": {
            col: {
            "min": float(train_df[col].min()),
            "max": float(train_df[col].max()),
            "mean": float(train_df[col].mean()),
            "std": float(train_df[col].std())
        }
        for col in num_cols
    },
    "categorical": {
        col: sorted(train_df[col].dropna().unique().tolist())
        for col in cat_cols
    },
    "split": {
        "seed": SEED,
        "method": "stratified_holdout",
        "sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df)
        }
    }
}

    with open(PROC / "churn_v1_feature_store_meta.json", "w") as f: 
        json.dump(meta, f, indent=2)

    print(f"\nPARQUET & JSON files saved to\n- {PROC}")


if __name__ == "__main__":
    train_test_split_cleaned_data(
        CLEANED_DATA / "cleaned_telco_churn_data_20250821_142727.csv"
        )