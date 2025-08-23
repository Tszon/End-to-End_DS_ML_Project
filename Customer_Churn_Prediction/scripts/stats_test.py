# scripts/stats_test.py
import pandas as pd
from src.myproject.utils.paths import CLEANED_DATA

# Q: Do the `Contract` Type & `MonthlyCharges` affect `Churn` rate?

# Chi2 Test of Independence (for cat vars)
# Build contingency table (counts of churn vs non-churn by Contract type)
from scipy.stats import chi2_contingency

def chi2_test_of_independence(cleaned_df):
    contingency_table = pd.crosstab(cleaned_df['Contract'], cleaned_df['Churn'])

    print("\n=== Chi-Square Test ===\n")
    print("Contingency Table:")
    print(contingency_table)

    # Run chi-square test
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)

    print(f"\nChi2 stat: {chi2:.2f}")
    print("Degrees of freedom:", dof)

    print(f"p-value: {p_val:1e}")
    if p_val < 0.05:
        print("=> REJECT null hypothesis : Contract type and Churn are associated.")
    else:
        print("=> FAIL to reject null hypothesis: No association between Contract type and Churn.")

    print("\nExpected frequencies:\n", expected)

# Kruskal-Wallis Test 
# Compare median `MonthlyCharges` across churn groups
from scipy.stats import kruskal

def kruskal_wallis_test(cleaned_df):
    # Split MonthlyCharges into groups based on churn outcome
    charges_churn = cleaned_df.loc[cleaned_df['Churn'] == 'Yes', 'MonthlyCharges']
    charges_no_churn = cleaned_df.loc[cleaned_df['Churn'] == 'No',  'MonthlyCharges']

    # Kruskal-Wallis test (H-test for independent samples)
    stat, p_val = kruskal(charges_churn, charges_no_churn)

    print("\n=== Kruskal-Wallis Test ===\n")
    print(f"H-stat: {stat:3f}")
    
    print(f"p-value: {p_val:1e}\n")
    if p_val < 0.05:
        print("=> REJECT null hypothesis: `MonthlyCharges` distributions differ by `Churn` status.\n")
    else:
        print("=> FAIL to reject null hypothesis: No difference in `MonthlyCharges` by `Churn` status.\n")


if __name__ == "__main__":
    cleaned_df = pd.read_csv(CLEANED_DATA / "cleaned_telco_churn_data_20250821_142727.csv")

    chi2_test_of_independence(cleaned_df)
    kruskal_wallis_test(cleaned_df)