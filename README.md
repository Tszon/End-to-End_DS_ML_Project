# 📊 End-to-End ML Deployment: Telco Customer Churn Project

## 🌐 Live Demo

Click 👉 [![Open in Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://tszontseng-telco-end2end-customer-churn-project.streamlit.app/)

---

## 📖 Project Overview

Customer churn is a major challenge for telecom companies — retaining customers is often more cost-effective than acquiring new ones.
This project builds an **end-to-end machine learning pipeline** to predict churn, explain drivers of churn, and segment customers into actionable groups for better retention strategies.

The project includes:

* **EDA** → Explore churn patterns, tenure, contracts, charges.
* **Customer Segmentation** → KMeans (baseline) vs HDBSCAN (tuned).
* **Churn Prediction** → Logistic Regression baseline vs advanced ensemble models (Random Forest, XGBoost, Voting Classifier).
* **Explainability** → SHAP summary & waterfall plots.
* **Interactive App** → Built with **Streamlit**, deployed on Streamlit Cloud.

---

## 📂 Dataset

Dataset: [WA_Fn-UseC_-Telco-Customer-Churn.csv](https://www.kaggle.com/blastchar/telco-customer-churn)

* **Target**: `"Churn"` (Yes/No)
* **Features include**:

  * **Demographics** → Gender, Senior Citizen, Dependents, Partner
  * **Services** → Phone, Internet, Tech Support, Streaming, Security
  * **Account Info** → Tenure, Contract, Billing, Payment Method
  * **Charges** → Monthly & Total Charges

---

## 🧪 Methods & Models

### 🔎 Exploratory Data Analysis (EDA)

* Customers with **fiber optic internet churn the most** (pricing/service quality issues).
* **DSL customers churn less**, possibly due to stable pricing or loyalty.
* **High churn in first 5 months** → critical onboarding phase.
* Long-tenure customers (>24 months) show **significantly lower churn rates**.

### 👥 Customer Segmentation (Unsupervised Learning)

* **Cluster 0 — Budget Loyalists** → Minimal services, mailed check payments, stable.
* **Cluster 1 — At-Risk Premiums** → Fiber optic, month-to-month, electronic check, highest churn risk.
* **Cluster 2 — Balanced Mainstream** → Moderate DSL usage, mixed services, mid-spenders.
* **Cluster -1 — Drifters** → DSL, no phone, low commitment.

### 📊 Churn Prediction Models

* Logistic Regression (baseline)
* Random Forest (ensemble)
* XGBoost (boosted trees)
* Voting Classifier (combined)

### 🔍 Explainability (SHAP)

* Feature importance ranking.
* SHAP summary plots + waterfall plots for individual predictions.

---

## 🚀 Deployment

* **Streamlit App** for interactive visualization and prediction.
* **Dockerized** for reproducibility.
* **Deployed on Streamlit Cloud** with a public link.

---

## ⚙️ Installation & Usage

### 1. Clone Repo

```bash
git clone https://github.com/<your-username>/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run Locally

```bash
streamlit run scripts/app.py
```

App runs at: [http://localhost:8501](http://localhost:8501)

### 4. Run with Docker

```bash
docker build -t churn-app .
docker run -p 8501:8501 churn-app
```

---

## 📦 Project Structure

```
Customer_Churn_Prediction/
│
├── data/                 # feature store JSON (not raw data)
├── models/               # saved ML models (.joblib)
├── reports_app/          # plots & visualizations
├── scripts/              # Streamlit app (app.py) & utilities
├── src/                  # preprocessing, feature engineering, utils
├── config.json           # config settings
├── requirements.txt      # dependencies
├── Dockerfile            # container setup
└── README.md             # this file
```

---

## 🛠️ Tech Stack

* **Python**: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `hdbscan`, `umap`
* **Visualization**: `matplotlib`, `seaborn`, `streamlit`
* **MLOps Tools**: `Docker`, `GitHub`, `MLflow` (Experimental Tracking)
* **Deployment**: `Streamlit Cloud`

---

## 📌 Next Steps

* Extend segmentation with deep embeddings.
* Add hyperparameter search with Optuna.
* Deploy with a custom domain using Render or Railway.

---

## 👤 Author

Developed by **[Tszon Tseng](https://github.com/Tszontseng)**

* 💼 Passionate about Data Science & AI
* 🚀 Building end-to-end ML pipelines
* 🌐 [LinkedIn Profile](https://www.linkedin.com/in/tszon-tseng-a381aa297/)

---

✨ With this app, telecom providers can **predict churn, understand why customers leave, and design better retention strategies.**
