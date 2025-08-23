# models/ensemble_clfs.py
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, classification_report, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
import mlflow, mlflow.sklearn

from src.myproject.utils.paths import CONFIG_PATH, FEATURE_DATA, MODELS_DIR, VIS_DIR

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Change folder to save churn prediction visualizations
churn_prediction_folder = VIS_DIR / "churn_prediction"

def run_ensemble_classifiers(
        train_engineered_parquet,
        test_engineered_parquet,
        version="v1",
        experiment_name="Ensemble_Clfs"
    ):
    """
    Compare different ensemble classifiers: 
    RandomForest, XGBoost, VotingClassifier.
    """

    # --- Load ---
    train = pd.read_parquet(train_engineered_parquet)
    target_col = "Churn"
    features = [c for c in train.columns if c != target_col]

    X_train = train[features].copy()
    y_train = train[target_col].copy()

    test = pd.read_parquet(test_engineered_parquet)
    X_test = test[features].copy()
    y_test = test[target_col].copy()

    # --- Set experiment + log MLflow run ---
    if mlflow.active_run():
        mlflow.end_run()
    
    mlflow.set_experiment("Customer_Churn_Project")
    with mlflow.start_run(run_name="ensemble_classifiers"):
        # Set tags
        mlflow.set_tag("experiment", f"{experiment_name}_{version}")

        # Log features used
        with open(FEATURE_DATA / f"features_ensemble_clf_{version}.json", "w") as f:
            json.dump({"features": features}, f, indent=4)
        print(f"\nFeatures used saved to:\n- {FEATURE_DATA / f'features_log_reg_{version}.json'}")
        mlflow.log_dict({"features": features}, "features.json")

        # --- Class imbalance handling ---------------------------------------------------
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        scale_pos_weight = neg / max(pos, 1)  # for XGBoost

        # --- CV setup ------------------------------------------------------------------
        cv = StratifiedKFold(
            n_splits=config["StratifiedKFold"]["n_splits"], 
            shuffle=config["StratifiedKFold"]["shuffle"], 
            random_state=config["StratifiedKFold"]["random_state"]
        )
        REFIT_METRIC = "f1"  # use F1 to pick best_model params

        # --- Baseline: Logistic Regression --------------------------------------------
        log_reg = LogisticRegression(
            class_weight=config["LogisticRegression"]["class_weight"],
            solver=config["LogisticRegression"]["solver"], 
            max_iter=config["LogisticRegression"]["max_iter"],
            random_state=config["LogisticRegression"]["random_state"]
        )

        log_reg_grid = {
            "C": np.logspace(-4, 4, 20),
            "penalty": ["l1", "l2"], 
            "solver": ["liblinear", "saga"]
        }

        # --- RandomForest + param grid -------------------------------------------------
        rfc = RandomForestClassifier(
                random_state=config["RandomForest"]["random_state"],
                class_weight=config["RandomForest"]["class_weight"]
            )

        rfc_grid = {
            "n_estimators": np.arange(200, 801, 200),
            "max_depth": [None, 10, 20, 40],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None]
        }

        # --- XGBoost + param grid ------------------------------------------------------
        xgbc = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=config["XGBoost"]["random_state"],
                scale_pos_weight=scale_pos_weight
            )

        xgbc_grid = {
            "n_estimators": np.arange(200, 601, 200),
            "max_depth": [3, 4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
            "gamma": [0, 0.5, 1.0]
        }


        def log_roc_curve(y_true, y_prob, name):
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            plt.figure(figsize=(6, 4))
            plt.plot([0, 1], [0, 1], "--", label="Random")
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
            plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC Curve - {name}")
            
            # Shade only the region above the diagonal
            plt.fill_between(
                fpr, tpr, fpr, where=(tpr >= fpr), interpolate=True, 
                alpha=0.2, color='orange'
            )
            plt.legend(loc="lower right")

            plot_path = churn_prediction_folder / f"{name.lower()}_roc_{version}.png"
            plt.savefig(plot_path, dpi=300)
            plt.close()
            mlflow.log_artifact(plot_path, artifact_path="plots")
            print(f"\nROC curve saved to:\n- {plot_path}")


        # --- Utility: run randomized search, CV metrics, test eval ---------------------
        def tune_and_eval(model, param_grid, name):
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=min(30, np.prod([len(v) for v in param_grid.values()])),  # cap work
                scoring=REFIT_METRIC,
                n_jobs=-1,
                cv=cv,
                random_state=config["RandomizedSearchCV"]["random_state"],
                refit=True,
                verbose=0
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_

            print(f"\n[{name}] best_model params: {getattr(search, 'best_params_', {})}")


            def to_serializable(obj):
                if isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                else:
                    return str(obj)

            # Log best_model hyperparameters
            params_plot_path = MODELS_DIR / "params" / f"{name.lower()}_best_params_{version}.json"
            with open(params_plot_path, "w") as f:
                json.dump(search.best_params_, f, indent=4, default=to_serializable)
            mlflow.log_artifact(params_plot_path, artifact_path="params")

            # Cross-validated comparison (same splits for fairness)
            f1_cv = cross_val_score(best_model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
            auc_cv = cross_val_score(best_model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
            print(f"[{name}] CV F1: {f1_cv.mean():.3f} ± {f1_cv.std():.3f} | "
                f"CV ROC AUC: {auc_cv.mean():.3f} ± {auc_cv.std():.3f}")
            
            # Log metrics
            mlflow.log_metric(f"{name}_cv_best_f1", f1_cv.mean())
            mlflow.log_metric(f"{name}_cv_best_auc", auc_cv.mean())

            # Fit on full train and evaluate on held-out test
            best_model.fit(X_train, y_train)
            y_prob = best_model.predict_proba(X_test)[:, 1]

            # Log ROC curve
            log_roc_curve(y_test, y_prob, name)

            # Choose threshold that maximizes Youden’s J on validation-style curve (optional)
            fpr, tpr, thr = roc_curve(y_test, y_prob)
            j = np.argmax(tpr - fpr)
            thr_star = thr[j]
            y_pred = (y_prob >= thr_star).astype(int)

            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            print(f"[{name}] TEST F1@thr*={thr_star:.3f}: {f1:.3f} | TEST ROC AUC: {auc:.3f}")
            print(f"[{name}] classification_report:\n{classification_report(y_test, y_pred)}")
            
            # Log Classification Report
            report_plot_path = churn_prediction_folder / f"{name.lower()}_report_{version}.txt"
            with open(report_plot_path, "w") as f:
                f.write(classification_report(y_test, y_pred))
            mlflow.log_artifact(report_plot_path, artifact_path="reports")

            # Save the best_model
            joblib.dump(best_model, MODELS_DIR / f"{name}_{version}.joblib")
            print(f"\nSaved {name} model to:\n- {MODELS_DIR / f'{name}_{version}.joblib'}")

            return name, best_model, f1_cv.mean(), auc_cv.mean(), f1, auc

        # --- Run all models ------------------------------------------------------------
        results = []
        for mdl, grid, nm in [
            (log_reg, log_reg_grid, "log_reg"),
            (rfc, rfc_grid, "rf"),
            (xgbc, xgbc_grid,"xgb")
        ]:
            results.append(tune_and_eval(mdl, grid, nm))
            
        # --- Summary table -------------------------------------------------------------
        summary = pd.DataFrame(
            [(nm, f1cv, auccv, f1t, auct) for nm, _, f1cv, auccv, f1t, auct in results],
            columns=["Model", "CV_F1", "CV_ROC_AUC", "Test_F1", "Test_ROC_AUC"]
        ).sort_values("Test_ROC_AUC", ascending=False)

        print("\n=== Model summary ===\n")
        summary_plot_path = churn_prediction_folder / f"ensemble_summary_{version}.txt"
        with open(summary_plot_path, "w") as f:
            f.write(summary.to_string(index=False))

        print(summary.head())
        mlflow.log_artifact(summary_plot_path, artifact_path="results")

        models_by_name = {nm: m for nm, m, *_ in results}

        # --- Voting Classifier ---------------------------------------------------------
        voting = VotingClassifier(
            estimators=[
                ("log_reg",  models_by_name["log_reg"]),
                ("rfc", models_by_name["rf"]),
                ("xgbc", models_by_name["xgb"]),
            ],
            voting=config["VotingClassifier"]["voting"],  # "soft"
            # give RF a little more say (tune if you wish)
            weights=config["VotingClassifier"]["weights"]  # [1,2,2]
        )

        # Log params
        mlflow.log_param("voting_weights", config["VotingClassifier"]["weights"])

        # CV comparison
        f1_cv = cross_val_score(voting, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        auc_cv = cross_val_score(voting, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        print(f"[Voting] CV F1: {f1_cv.mean():.3f} ± {f1_cv.std():.3f} | CV ROC AUC: {auc_cv.mean():.3f} ± {auc_cv.std():.3f}")

        # Log metrics
        mlflow.log_metric("Voting_cv_f1_mean", f1_cv.mean())
        mlflow.log_metric("Voting_cv_auc_mean", auc_cv.mean())

        # Fit & test
        voting.fit(X_train, y_train)
        y_prob = voting.predict_proba(X_test)[:, 1]
        fpr, tpr, thr = roc_curve(y_test, y_prob); thr_star = thr[np.argmax(tpr - fpr)]
        y_pred = (y_prob >= thr_star).astype(int)

        # Log ROC curve
        log_roc_curve(y_test, y_prob, "VotingClassifier")

        print(f"[Voting] TEST F1@thr*={thr_star:.3f}: {f1_score(y_test, y_pred):.3f} | "
            f"TEST ROC AUC: {roc_auc_score(y_test, y_prob):.3f}")
        print(f"[Voting] classification_report:\n{classification_report(y_test, y_pred)}")

        # Save the Voting Classifier model
        joblib.dump(voting, MODELS_DIR / f"{voting}_{version}.joblib", compress=3)
        print(f"\nSaved voting classifier to:\n- {MODELS_DIR / f'voting_{version}.joblib'}")
        
        # Log Classification Report
        report_plot_path = churn_prediction_folder / f"voting_class_report_{version}.txt"
        with open(report_plot_path, "w") as f:
            f.write(classification_report(y_test, y_pred))
        mlflow.log_artifact(report_plot_path, artifact_path="reports")
        
        # Log LogisticRegression model
        mlflow.sklearn.log_model(
            sk_model=models_by_name["log_reg"],
            name="log_reg_model",
            input_example=X_train.iloc[:1],
            signature=mlflow.models.infer_signature(X_train,  models_by_name["log_reg"].predict(X_train))
        )

        # Log RandomForestClassifier model
        mlflow.sklearn.log_model(
            sk_model=models_by_name["rf"],
            name="rf_class_model",
            input_example=X_train.iloc[:1],
            signature=mlflow.models.infer_signature(X_train,  models_by_name["rf"].predict(X_train))
        )
        # Log XGBoostClassifier model
        mlflow.sklearn.log_model(
            sk_model=models_by_name["xgb"],
            name="xgbc_class_model",
            input_example=X_train.iloc[:1],
            signature=mlflow.models.infer_signature(X_train,  models_by_name["xgb"].predict(X_train))
        )

        # Log Voting Classfier model
        mlflow.sklearn.log_model(
            sk_model=voting,
            name="voting_class_model",
            input_example=X_train.iloc[:1],
            signature=mlflow.models.infer_signature(X_train,  voting.predict(X_train))
        )


if __name__ == "__main__":
    run_ensemble_classifiers(
        train_engineered_parquet=FEATURE_DATA / "churn_v1_train_engineered_features.parquet",
        test_engineered_parquet=FEATURE_DATA / "churn_v1_test_engineered_features.parquet"
    )