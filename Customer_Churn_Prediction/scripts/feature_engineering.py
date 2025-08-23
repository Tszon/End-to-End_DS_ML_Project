# scripts/feature_engineering.py
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

from src.myproject.utils.paths import PREPROCESSED_DATA, FEATURE_DATA, MODELS_DIR

def feature_engineering(train_parquet, test_parquet, version="v1"):
    # --- Load ---
    train = pd.read_parquet(train_parquet)
    test = pd.read_parquet(test_parquet)

    target_col = "Churn"
    features = [c for c in train.columns if c != target_col]

    X_train = train[features].copy()
    X_test = test[features].copy()
    y_train = train[target_col].map({"No": 0, "Yes": 1}).astype(int)
    y_test = test[target_col].map({"No": 0, "Yes": 1}).astype(int)

    # --- Identify numeric / categorical columns ---
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if "SeniorCitizen" in num_cols:  # often treat as categorical
        num_cols.remove("SeniorCitizen")
    cat_cols = [c for c in features if c not in num_cols]

    # --- Preprocessing pipeline ---
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ]
    )

    # --- Fit on train, transform both ---
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # --- Get feature names after transformation ---
    num_features = num_cols
    cat_features = preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(cat_cols)
    all_features = np.concatenate([num_features, cat_features])

    # --- Convert back to DataFrame for saving ---
    X_train_df = pd.DataFrame(X_train_proc, columns=all_features, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_proc, columns=all_features, index=X_test.index)

    # Add target column back
    X_train_df["Churn"] = y_train.values
    X_test_df["Churn"] = y_test.values

    # --- Save ---
    out_train = FEATURE_DATA / f"churn_{version}_train_engineered_features.parquet"
    out_test = FEATURE_DATA / f"churn_{version}_test_engineered_features.parquet"
    # X_train_df.to_parquet(out_train, index=False)
    # X_test_df.to_parquet(out_test, index=False)
    # print(f"\nSaved feature-engineered datasets:\n- {out_train}\n- {out_test}")

    # Save the preprocessor for future use
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor_v1.joblib")
    print(f"\nSaved preprocessor pipeline to:\n- {MODELS_DIR / 'preprocessor_v1.joblib'}")

    


if __name__ == "__main__":
    feature_engineering(
        train_parquet=PREPROCESSED_DATA / "churn_v1_train.parquet",
        test_parquet=PREPROCESSED_DATA / "churn_v1_test.parquet",
        version="v1"
    )
