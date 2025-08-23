import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
from src.myproject.utils.paths import CONFIG_PATH, FEATURE_DATA, MODELS_DIR, VIS_DIR

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

churn_prediction_folder = VIS_DIR / "churn_prediction"

# ---------- helpers ----------
def ensure_shapes_trim_bias(shap_vals, n_features):
    """Trim bias column so SHAP values align with features."""
    if isinstance(shap_vals, list):
        return [sv[:, :n_features] for sv in shap_vals]

    sv = np.asarray(shap_vals)
    if sv.ndim == 3:
        if sv.shape[0] in (2, 3):        # (C, N, F+1)
            sv = sv[:, :, :n_features]
        elif sv.shape[-1] in (2, 3):     # (N, F+1, C)
            sv = sv[:, :n_features, :]
        return sv
    return sv[:, :n_features]


def pick_class1(shap_vals):
    """Always pick class-1 SHAP values in a consistent shape (N,F)."""
    if isinstance(shap_vals, list):
        return shap_vals[1]
    sv = np.asarray(shap_vals)
    if sv.ndim == 3:
        if sv.shape[0] in (2, 3):        # (C, N, F)
            return sv[1]
        if sv.shape[-1] in (2, 3):       # (N, F, C)
            return sv[..., 1]
        return sv[1]
    return sv


def _get_base_value(explainer, class_idx=1):
    base_arr = np.atleast_1d(explainer.expected_value)
    return base_arr[class_idx] if base_arr.size > 1 else float(base_arr[0])


# ---------- SHAP Summary ----------
def plot_shap_summaries(X_train, rf_clf, xgb_clf, feature_names): 
    S = min(2000, len(X_train))
    X_sample = X_train.sample(S, random_state=config["plot_SHAP_waterfall_plots"]["random_state"])

    # ---------- RF ----------
    expl_rf = shap.TreeExplainer(rf_clf)
    sv_rf = expl_rf.shap_values(X_sample)
    sv_rf = ensure_shapes_trim_bias(sv_rf, X_sample.shape[1])
    sv_rf_cls1 = pick_class1(sv_rf)

    shap.summary_plot(sv_rf_cls1, X_sample,
                      feature_names=feature_names,
                      max_display=15, plot_type="dot", show=False)
    fig = plt.gcf()
    fig.suptitle("Random Forest SHAP Summary Plot")
    plot_path = churn_prediction_folder / "shap_summary_rf.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to:\n- {plot_path}")

    # ---------- XGB ----------
    bg = X_sample.sample(200, random_state=config["plot_SHAP_waterfall_plots"]["random_state"])   # background for interventional
    expl_xgb = shap.TreeExplainer(
        xgb_clf, data=bg,
        feature_perturbation="interventional"
    )
    sv_xgb = expl_xgb.shap_values(X_sample)
    sv_xgb = ensure_shapes_trim_bias(sv_xgb, X_sample.shape[1])
    sv_xgb_cls1 = pick_class1(sv_xgb)

    shap.summary_plot(sv_xgb_cls1, X_sample,
                      feature_names=feature_names,
                      max_display=15, plot_type="dot", show=False)
    fig = plt.gcf()
    fig.suptitle("XGBoost SHAP Summary Plot")
    plot_path = churn_prediction_folder / "shap_summary_xgb.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to:\n- {plot_path}")


# ---------- SHAP Waterfall ----------
def plot_waterfall_for_index(explainer, X, idx, feature_names, model_name, class_idx=1):
    X_one = X.iloc[[idx]]
    sv = explainer.shap_values(X_one)
    sv = ensure_shapes_trim_bias(sv, X_one.shape[1])
    sv_cls1 = pick_class1(sv)
    sv_row = np.asarray(sv_cls1[0]).ravel()

    base = _get_base_value(explainer, class_idx)

    ex = shap.Explanation(
        values=sv_row,
        base_values=base,
        data=X_one.values[0],
        feature_names=feature_names
    )
    shap.plots.waterfall(ex, max_display=15, show=False)
    fig = plt.gcf()
    fig.suptitle(f"SHAP Waterfall (row {idx}, model={model_name})")
    plot_path = churn_prediction_folder / f"shap_waterfall_row{idx}_{model_name}.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to:\n- {plot_path}")


# ---------- Main entry ----------
def plot_shap_waterfall(train_engineered_parquet, rf_clf_path, xgb_clf_path):
    train = pd.read_parquet(train_engineered_parquet)
    target_col = "Churn"
    feature_names = [c for c in train.columns if c != target_col]
    X_train = train[feature_names].copy()

    rf_clf = joblib.load(rf_clf_path)
    xgb_clf = joblib.load(xgb_clf_path)

    plot_shap_summaries(X_train, rf_clf, xgb_clf, feature_names)

    np.random.seed(config["plot_SHAP_waterfall_plots"]["random_state"])
    row_idx = np.random.choice(len(X_train))

    print(f"\nWaterfall for row {row_idx} (Random Forest)")
    expl_rf = shap.TreeExplainer(rf_clf)
    plot_waterfall_for_index(expl_rf, X_train, row_idx, feature_names, model_name="rf")

    print(f"\nWaterfall for row {row_idx} (XGBoost)")
    bg = X_train.sample(200, random_state=config["plot_SHAP_waterfall_plots"]["random_state"])
    expl_xgb = shap.TreeExplainer(
        xgb_clf, data=bg,
        feature_perturbation="interventional",
        model_output="probability"
    )
    plot_waterfall_for_index(expl_xgb, X_train, row_idx, feature_names, model_name="xgb")


if __name__ == "__main__":
    plot_shap_waterfall(
        train_engineered_parquet=FEATURE_DATA / "churn_v1_train_engineered_features.parquet",
        rf_clf_path=MODELS_DIR / "rf_v1.joblib",
        xgb_clf_path=MODELS_DIR / "xgb_v1.joblib"
    )
