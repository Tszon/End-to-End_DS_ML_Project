# models/Logistic_Regression_clf.py
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import (
    classification_report,
    roc_auc_score, roc_curve, f1_score
)
import mlflow, mlflow.sklearn

from src.myproject.utils.paths import (
    CONFIG_PATH, FEATURE_DATA, VIS_DIR
)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Create folder to save churn prediction visualizations
churn_prediction_folder = VIS_DIR / "churn_prediction"
os.makedirs(churn_prediction_folder, exist_ok=True)

def run_logistic_regression(
        train_engineered_parquet, 
        test_engineered_parquet,
        version="v1",
        experiment_name="Logistic_Regression"
    ):
    """Perform Logistic Regression."""
    # --- Load ---
    train = pd.read_parquet(train_engineered_parquet)
    target_col = "Churn"
    features = [c for c in train.columns if c != target_col]

    X_train = train[features].copy()
    y_train = train[target_col].copy()

    test = pd.read_parquet(test_engineered_parquet)
    X_test = test[features].copy()
    y_test = test[target_col].copy()

    # --- Train ---
    log_reg = LogisticRegression(
        class_weight=config["LogisticRegression"]["class_weight"],
        solver=config["LogisticRegression"]["solver"], 
        max_iter=config["LogisticRegression"]["max_iter"],
        random_state=config["LogisticRegression"]["random_state"]
    )
    log_reg.fit(X_train, y_train)

    # --- Set experiment + log MLflow run ---
    if mlflow.active_run():
        mlflow.end_run()
    
    mlflow.set_experiment("Customer_Churn_Project")
    with mlflow.start_run(run_name="logistic_regression"):
        # Set tags
        mlflow.set_tag("experiment", f"{experiment_name}_{version}")
        
        # Log params
        mlflow.log_params({
            "n_features": len(features),
            "class_weight": "balanced",
            "max_iter": 1000,
            "solver": log_reg.solver
        })
        
        # Log features used
        with open(FEATURE_DATA / f"features_log_reg_{version}.json", "w") as f:
            json.dump({"features": features}, f, indent=4)
        print(f"\nFeatures used saved to:\n- {FEATURE_DATA / f'features_log_reg_{version}.json'}")
        mlflow.log_dict({"features": features}, "features.json")

        # --- Evaluate ---
        y_pred = log_reg.predict(X_test)
        y_prob = log_reg.predict_proba(X_test)[:, 1]

        # --- Classification Report ---
        print(f"\nF1: {f1_score(y_test, y_pred):.3f}")
        print(f"\nROC AUC: {roc_auc_score(y_test, y_prob):.3f}")
        print(classification_report(y_test, y_pred))

        # Log metrics
        mlflow.log_metric("f1", f1_score(y_test, y_pred))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob))
        
        class_report_path = churn_prediction_folder / f"class_report_log_reg_{version}.txt"
        with open(class_report_path, "w") as f:
            f.write(classification_report(y_test, y_pred))
        mlflow.log_artifact(class_report_path, artifact_path="reports")

        # 1. Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        _, ax = plt.subplots(figsize=(6, 4))
        # Chance line
        ax.plot([0, 1], [0, 1], linestyle="--", label="Random Guessing")
        # ROC line (keep handle for legend)
        ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
        # Shade only the region above the diagonal
        plt.fill_between(
            fpr, tpr, fpr, where=(tpr >= fpr), interpolate=True, 
            alpha=0.2, color='orange'
        )
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
            title="Logistic Regression ROC Curve")
        ax.legend()

        plot_path = churn_prediction_folder / f"log_reg_roc_curve_{version}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        mlflow.log_artifact(plot_path, artifact_path="plots")
        print(f"\nROC curve saved to:\n- {plot_path}")

        # Log Logistic Regression model
        mlflow.sklearn.log_model(
            sk_model=log_reg,
            name="log_reg_model",
            input_example=X_train.iloc[:1],
            signature=mlflow.models.infer_signature(X_train,  log_reg.predict(X_train))
        )


if __name__ == "__main__":
    run_logistic_regression(
        train_engineered_parquet=FEATURE_DATA / "churn_v1_train_engineered_features.parquet",
        test_engineered_parquet=FEATURE_DATA / "churn_v1_test_engineered_features.parquet"
    )