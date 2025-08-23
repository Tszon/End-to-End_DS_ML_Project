import sys
from pathlib import Path
import json
import streamlit as st
import pandas as pd
import joblib

# --- Path setup ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.myproject.utils.paths import MODELS_DIR, PREPROCESSED_DATA, VIS_DIR

# --- Page Config ---
st.set_page_config(page_title="Telco Customer Churn Prediction", layout="wide")

# --- Sidebar Navigation (vertical main tabs) ---
page = st.sidebar.radio(
    "Navigate",
    [
        "📖 Introduction",
        "🔎 EDA & Customer Segregation",
        "📊 Churn Prediction & Explainable AI",
        "🧪 Model Demonstration"
    ]
)

# --- TAB 1: INTRO ---
if page == "📖 Introduction":
    st.title("📖 Telco Customer Churn Prediction Project")
    st.markdown("""
                ## End-to-End ML Deployment Project
    **IBM Sample Data sets** from: *https://www.kaggle.com/datasets/blastchar/telco-customer-churn*
                
    Welcome to the **Telco Customer Churn Prediction Dashboard**!  

    The goal of this project is to **predict customer behaviour** so the company can **retain customers**.  
    By analysing customer data, we can develop **targeted retention programs** to reduce churn and increase loyalty.

    ### 📂 Dataset
    - Source: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
    - Each row = 1 customer, columns = customer attributes.
    
    **Key features include:**
    - **Churn** → whether the customer left in the last month.  
    - **Services** → `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV/movies`.  
    - **Account Info** → `tenure`, `Contract`, `PaymentMethod`, `PaperlessBilling`, `MonthlyCharges`.  
    - **Demographics** → `Gender`, `Partner`, `Dependents`.

    ### ⚙️ Tools & Methods
    - **Data Management**: NumPy, Pandas, JSON  
    - **Statistical Tests**: SciPy  
    - **Data Visualisation**: Matplotlib, Seaborn  
    - **Machine Learning**:  
        - Supervised: Logistic Regression, Random Forest, XGBoost  
        - Unsupervised: K-Means Clustering, HDBSCAN, UMAP
    - **Experimental Tracking**: MLflow  
    - **Explainability**: SHAP  
    - **Deployment**: Streamlit, Docker
    - **Version Control**: Git/GitHub  
    """)

    img = VIS_DIR / "Telco-business-models-1.png"
    if img.exists():
        st.image(img, caption="Telco Business Model.", use_container_width=True)

# --- TAB 2: EDA & SEGREGATION ---
elif page == "🔎 EDA & Customer Segregation":
    st.title("🔎 Exploratory Data Analysis & Clustering")

    tab_eda, tab_cluster = st.tabs(["📊 EDA", "👥 Customer Segregation (Clustering)"])

    # --- Sub-tab: EDA ---
    with tab_eda:
        st.subheader("Tenure vs Monthly Charges")

        col1, col2 = st.columns(2)  # split page into 2 equal columns

        plot1 = VIS_DIR / "tenure_vs_monthlycharges_by_churn.png"
        plot2 = VIS_DIR / "tenure_vs_monthlycharges_by_internetservice.png"

        if plot1.exists(): 
            with col1:
                st.image(plot1, caption="Tenure vs Monthly Charges by Churn")
        if plot2.exists(): 
            with col2:
                st.image(plot2, caption="Tenure vs Monthly Charges by Internet Service")

        st.markdown("""
        **Insights:**
        - **Fiber Optic customers churn the most** → pricing/service quality issues.  
        - **DSL customers churn less** → stable pricing or loyalty.  
        """)

        st.subheader("Tenure Distribution by Churn")

        plot3 = VIS_DIR / "tenure_distribution_by_churn.png"
        if plot3.exists(): 
            st.image(plot3, caption="Tenure Distribution by Churn")

        st.markdown("""
        **Insights:**
        - **High churn risk in first 5 months** → onboarding/competitors matter most early.  
        - Customers with **> 24 months tenure** churn less.  
        - **Loyalty peak at 60–70 months** → long-term contracts encourage stability.  
        """)

    # --- Sub-tab: Clustering ---
    with tab_cluster:
        st.subheader("KMeans Results")

        col1, col2, col3 = st.columns(3)
        
        with col1:
            plot1 = VIS_DIR / "kmeans_elbow_v1.png"
            if plot1.exists(): 
                st.image(plot1, caption="KMeans Elbow Plot. Optimal k = 3 where inertia drops much more slower.")
        with col2:
            plot2 = VIS_DIR / "kmeans_clusters_v1.png"
            if plot2.exists(): 
                st.image(plot2, caption="KMeans clusters in original feature space.")
        with col3:
            plot3 = VIS_DIR / "KMeans_boxplots_v1.png"
            if plot3.exists(): 
                st.image(plot3, caption="`tenure` and `MonthlyCharges` boxplots by KMeans clusters.")

        st.markdown("""
        **KMeans Insights:**
        - The elbow plot suggests the optimal number of clusters (i.e **k = 3**) for stable segmentation.  
        - Clusters show clear differences in **tenure, monthly charges, and churn behaviour**.  
        - Some clusters are dominated by **short-tenure, high-risk customers**, while others represent **long-tenure loyal customers**.  
        - Boxplots confirm that **monthly charges** and **contract type** are strong differentiators across groups.  
        - KMeans serves as a **baseline segmentation**, but more advanced clustering (e.g., HDBSCAN) reveals deeper structure.  
        """)
        
    with tab_cluster:
        st.subheader("HDBSCAN Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            for plot_name, caption in zip([
                "hdbscan_clusters_umap_v1.png",
                "hdbscan_MonthlyCharges_distribution.png"
                ], [
                    "HDBSCAN clusters in UMAP space.",
                    "`MonthlyCharges` distribution by HDBSCAN clusters."
                ]):

                path = VIS_DIR / plot_name
                if path.exists(): 
                    st.image(path, caption=caption)
        with col2:
            for plot_name, caption in zip([
                "hdbscan_clusters_v1.png",
                "hdbscan_boxplots.png"
                ], [
                    "HDBSCAN clusters in original feature space.",
                    "`tenure` and `MonthlyCharges` boxplots by HDBSCAN clusters."
                ]):

                path = VIS_DIR / plot_name
                if path.exists(): 
                    st.image(path, caption=caption)
        with col3:
            for plot_name, caption in zip([
                "hdbscan_tenure_distribution.png",
                "hdbscan_cat_by_clusters.png"
                ], [
                    "`tenure` distributions by HDBSCAN clusters.",
                    "Categorical features by HDBSCAN clusters."
                ]):

                path = VIS_DIR / plot_name
                if path.exists(): 
                    st.image(path, caption=caption)

        st.markdown("""
        **HDBSCAN Insights:**
        - **Cluster 0 – Budget Loyalists** → Minimal services, mailed checks → stable & cost-conscious.  
        - **Cluster 1 – At-Risk Premiums** → Fibre Optic + month-to-month + electronic checks → high churn risk.  
        - **Cluster 2 – Balanced Mainstream** → DSL, moderate services, diverse payments → mid-spending & moderately loyal.  
        - **Cluster -1 (Noise/Outliers) – Drifters** → DSL, no phone service → scattered & unstable.  
        """)

# --- TAB 3: PREDICTION ---
elif page == "📊 Churn Prediction & Explainable AI":
    st.title("📊 Churn Prediction Models & Explainable AI")

    tab_logreg, tab_rf, tab_xgb, tab_vote, tab_shap = st.tabs([
        "🔹 Logistic Regression",
        "🌲 Random Forest",
        "⚡ XGBoost",
        "🗳️ Voting Classifier",
        "🔎 SHAP & Waterfall"
    ])

    # --- Logistic Regression ---
    with tab_logreg:
        st.subheader("Baseline Logistic Regression")

        col1, col2 = st.columns(2)

        roc = VIS_DIR / "churn_prediction" / "log_reg_roc_v1.png"
        if roc.exists(): 
            with col1:
                st.image(roc, caption="Logistic Regression ROC")

        report = VIS_DIR / "churn_prediction" / "log_reg_report_v1.txt"
        if report.exists():
            with col2:
                st.subheader("Classification Report")
                st.text(report.read_text())

        st.markdown("""
        - Simple interpretable baseline.  
        - Useful for understanding **key churn drivers**, but may underfit.  
        """)

    # --- Random Forest ---
    with tab_rf:
        st.subheader("Random Forest Classifier")

        col1, col2 = st.columns(2)

        roc = VIS_DIR / "churn_prediction" / "rf_roc_v1.png"
        rpt = VIS_DIR / "churn_prediction" / "rf_report_v1.txt"

        if roc.exists(): 
            with col1:
                st.image(roc, caption="Random Forest ROC")

        if rpt.exists():
            with col2:
                st.subheader("Classification Report")
                st.text(rpt.read_text())

        st.markdown("""
        - Captures nonlinearities and feature interactions.  
        - Provides improved accuracy over Logistic Regression.  
        """)

    # --- XGBoost ---
    with tab_xgb:
        st.subheader("XGBoost Classifier")

        col1, col2 = st.columns(2)

        roc = VIS_DIR / "churn_prediction" / "xgb_roc_v1.png"
        rpt = VIS_DIR / "churn_prediction" / "xgb_report_v1.txt"

        if roc.exists(): 
            with col1:
                st.image(roc, caption="XGBoost ROC")

        if rpt.exists():
            with col2:
                st.subheader("Classification Report")
                st.text(rpt.read_text())

        st.markdown("""
        - Gradient boosting algorithm with strong predictive power.  
        - Handles complex relationships better than Random Forest.  
        """)

    # --- Voting Classifier ---
    with tab_vote:
        st.subheader("Voting Classifier (Ensemble)")

        col1, col2 = st.columns(2)

        roc = VIS_DIR / "churn_prediction" / "votingclassifier_roc_v1.png"
        rpt = VIS_DIR / "churn_prediction" / "voting_class_report_v1.txt"

        if roc.exists(): 
            with col1:
                st.image(roc, caption="Voting Classifier ROC")

        if rpt.exists():
            with col2:
                st.subheader("Classification Report")
                st.text(rpt.read_text())
                
        st.markdown("""
        - Combines Logistic Regression, Random Forest, XGB Classifier.  
        - Aims to balance bias & variance.  
        - Typically achieves the strongest ROC-AUC.  
        - Model **weights**: Logstic Regression=0.25, **Random Forest=0.5**, XGBoost Classifier=0.4.
        """)

    # --- SHAP & Waterfall ---
    with tab_shap:
        st.subheader("Explainability with SHAP")
        col1, col2 = st.columns(2)
        col1.image(VIS_DIR / "churn_prediction" / "shap_summary_rf.png", caption="Random Forest SHAP Beeswarm Plot")
        col2.image(VIS_DIR / "churn_prediction" / "shap_summary_xgb.png", caption="XGBoost SHAP Beeswarm Plot")

        st.markdown("""
        **Insights:**
        - Top churn drivers: `tenure`, `MonthlyCharges`, `Contract`, `InternetService`.  
        - Short `tenure` + high `MonthlyCharges` + month-to-month `Contract` = **highest churn risk**.  
        - SHAP explains feature contributions for each prediction.  
        """)

        st.subheader("Example SHAP Waterfall Explanations (Row 860)")

        col1, col2 = st.columns(2)

        for model, col, model_name in zip(
            ["rf", "xgb"], [col1, col2], ["Random Forest Classifier", "XGBoost Classifier"]
            ):
            wf = VIS_DIR / "churn_prediction" / f"shap_waterfall_row860_{model}.png"
            if wf.exists():
                with col:
                    st.image(wf, caption=f"Waterfall Plot ({model_name})")

# --- TAB 4: Model Demonstration ---
elif page == "🧪 Model Demonstration":
    st.title("🧪 Interactive Model Demonstration")
    st.markdown("""
    Enter customer information below and select a classifier.  
    The app will output the predicted churn probability.
    """)

    # --- Load feature metadata ---
    with open(PREPROCESSED_DATA / "churn_v1_feature_store_meta.json", "r") as f:
        feature_meta = json.load(f)

    num_features = feature_meta["numeric"]        # dict of {col: {min, max, mean, std}}
    cat_features = feature_meta["categorical"]    # dict of {col: [categories...]}

    # --- Load preprocessor + models ---
    preprocessor = joblib.load(MODELS_DIR / "preprocessor_v1.joblib")
    models = {
        "Logistic Regression": joblib.load(MODELS_DIR / "log_reg_v1.joblib"),
        "Random Forest": joblib.load(MODELS_DIR / "rf_v1.joblib"),
        "XGBoost": joblib.load(MODELS_DIR / "xgb_v1.joblib"),
    }

    # --- Sidebar for model choice ---
    clf_choice = st.selectbox("Choose a classifier model:", list(models.keys()))
    model = models[clf_choice]

    # --- Input Form (dynamic from metadata) ---
    with st.form("prediction_form"):
        input_data = {}

        st.subheader("Numeric Features")
        for feat, stats in num_features.items():
            min_val = float(stats["min"])
            max_val = float(stats["max"])
            mean_val = float(stats["mean"])

            if "tenure" in feat.lower():
                input_data[feat] = st.slider(
                    f"{feat} (months)", int(min_val), int(max_val), int(mean_val)
                )
            elif "monthlycharges" in feat.lower():
                input_data[feat] = st.slider(
                    f"{feat}", float(min_val), float(max_val), float(mean_val)
                )
            else:
                input_data[feat] = st.number_input(
                    feat, min_value=min_val, max_value=max_val, value=mean_val
                )

        st.subheader("Categorical Features")
        cat_items = list(cat_features.items())

        # Split into n columns
        num_cols = 8
        cols = st.columns(num_cols)
        for i, (feat, categories) in enumerate(cat_items):
            col = cols[i % num_cols]
            with col:
                input_data[feat] = st.selectbox(feat, categories)

        submitted = st.form_submit_button("Predict Churn")

    # --- Prediction ---
    if submitted:
        raw_input_df = pd.DataFrame([input_data])

        try:
            # Apply preprocessing before feeding to model
            X_proc = preprocessor.transform(raw_input_df)
            proba = model.predict_proba(X_proc)[0, 1]
            pred = model.predict(X_proc)[0]

            st.success(f"**Prediction:** {'Churn' if pred==1 else 'No Churn'}")
            st.info(f"**Churn Probability:** {proba:.2%}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")



