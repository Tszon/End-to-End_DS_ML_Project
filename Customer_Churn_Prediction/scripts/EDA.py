# scripts/EDA.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.myproject.utils.paths import CLEANED_DATA, VIS_DIR

def load_data_and_EDA(cleaned_df):
    # === 1. Plot `Tenure` vs. `MonthlyCharges` by `Churn` ===
    sns.scatterplot(data=cleaned_df, y="tenure", x="MonthlyCharges", 
                    hue="Churn", alpha=0.85, s=12)
    plt.title("Tenure vs Monthly Charges by Churn")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "tenure_vs_monthlycharges_by_churn.png", dpi=300)
    plt.close()

    # === 2. Plot `Tenure` vs. `MonthlyCharges` by `InternetService` ===
    sns.scatterplot(data=cleaned_df, y="tenure", x="MonthlyCharges", 
                    hue="InternetService", alpha=0.85, s=12)
    plt.title("Tenure vs Monthly Charges by Internet Service")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "tenure_vs_monthlycharges_by_internetservice.png", dpi=300)
    plt.close()

    # === 3. Plot distributions of `Tenure`by `Churn` ===
    plt.figure(figsize=(8, 5))
    sns.kdeplot(cleaned_df[cleaned_df['Churn'] == 'No']['tenure'], label='Stayed', fill=True)
    sns.kdeplot(cleaned_df[cleaned_df['Churn'] == 'Yes']['tenure'], label='Churned', fill=True)
    plt.xlabel('Tenure (Months)')
    plt.ylabel('Density')

    plt.xticks(np.arange(0, cleaned_df['tenure'].max() + 10, 5))
    plt.axvline(x=3, linestyle='--', color='r')
    plt.axvline(x=69, linestyle='--', color='b')
    plt.title('Distributions of Tenure for Stayed and Churned Customers')
    plt.legend(['Stayed', 'Churned', 'Churn Peak', 'Stay Peak'], loc='upper center')
    plt.savefig(VIS_DIR / "tenure_distribution_by_churn.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    cleaned_df = pd.read_csv(CLEANED_DATA / "cleaned_telco_churn_data_20250821_142727.csv")
    
    load_data_and_EDA(
        cleaned_df=cleaned_df
    )