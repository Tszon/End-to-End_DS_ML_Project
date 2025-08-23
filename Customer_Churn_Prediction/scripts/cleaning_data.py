# scripts/cleaning_data.py
import pandas as pd
from datetime import datetime
from src.myproject.utils.paths import RAW_DATA, CLEANED_DATA

def load_and_clean_data(raw_data_path: str):
    # Load your CSV from the notebooks folder
    df = pd.read_csv(raw_data_path)

    df['gender'] = df['gender'].map({'Male': 'M', 'Female': 'F'})

    # Drop redundant numeric feature
    # `TotalCharges` = `tenure` x `MonthlyCharges`
    df.drop(columns='TotalCharges', inplace=True, errors='ignore')

    # Save CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = CLEANED_DATA / f"cleaned_telco_churn_data_{timestamp}.csv"
    df.to_csv(output_dir, index=False)
    print(f"Saved cleaned data to:\n- {output_dir}")

    # Data inconsistencies checking
    for col in df.drop(columns=['customerID']).columns:
        print(col, ":",  df[col].unique())


if __name__ == "__main__":
    load_and_clean_data(
        RAW_DATA / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        # Raw data path
    )