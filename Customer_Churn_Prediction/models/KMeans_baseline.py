# models/baseline_KMeans.py
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow, mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score,
    calinski_harabasz_score, davies_bouldin_score
)

from src.myproject.utils.paths import (
    CONFIG_PATH, PREPROCESSED_DATA, FEATURE_DATA,
    REPORT_DATA, VIS_DIR
)

# --- Load config ---
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

def run_kmeans(
        train_engineered_parquet,
        train_original_parquet,
        n_clusters=3,
        version="v1",
        experiment_name="KMeans_Clustering"
    ):
    """
    Perform KMeans clustering, log results to MLflow, save segmentation + plots.
    """
    # --- Load ---
    train = pd.read_parquet(train_engineered_parquet)
    target_col = "Churn"
    features = [c for c in train.columns if c != target_col]

    X_train = train[features].copy()
    y_train = train[target_col].copy()

    # Build PCA + KMeans pipeline 
    pipeline = Pipeline([
        ("pca", PCA(n_components=0.95, random_state=config["kmeans"]["random_state"])),
        ("kmeans", KMeans(n_clusters=n_clusters, n_init="auto", random_state=SEED))
    ])

    # Fit pipeline
    pipeline.fit(X_train)
    labels = pipeline["kmeans"].labels_

    # Extract PCA embedding for elbow plot / metrics
    X_emb = pipeline.named_steps["pca"].transform(X_train)


    # --- Metrics across k ---
    def kmeans_metrics(X, k, seed=config["kmeans"]["random_state"]):
        km = KMeans(n_clusters=k, n_init="auto", random_state=seed).fit(X)
        labels = km.labels_
        return {
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
        }

    ks = range(2, 11)
    scores = pd.DataFrame([kmeans_metrics(X_emb, k) for k in ks]).set_index("k")
    print("=== KMeans Clustering ===")
    print("\nCluster quality metrics:\n", scores.head())


    # --- Stability check (ARI) ---
    def stability(X, k, runs=5):
        labs = [
            KMeans(n_clusters=k, n_init="auto", random_state=100 + r).fit(X).labels_
            for r in range(runs)
        ]
        pair_ari = [
            adjusted_rand_score(labs[i], labs[j])
            for i in range(runs)
            for j in range(i + 1, runs)
        ]
        return np.mean(pair_ari)

    stab = {k: stability(X_emb, k) for k in ks}
    print("\nStability (mean ARI across seeds):\n", stab)

    # --- Set experiment + log MLflow run ---
    if mlflow.active_run():
        mlflow.end_run()

    mlflow.set_experiment("Customer_Churn_Project")
    with mlflow.start_run(run_name=f"kmeans_{n_clusters}_clusters"):
        # Set tags
        mlflow.set_tag("experiment", f"{experiment_name}_{version}")
        
        # Log params
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("random_state", config["kmeans"]["random_state"])
        mlflow.log_param("n_features", len(features))

        # Log features used
        with open(FEATURE_DATA / f"features_kmeans_{version}.json", "w") as f:
            json.dump({"features": features}, f, indent=4)
        print(f"\nFeatures used saved to:\n- {FEATURE_DATA / f'features_kmeans_{version}.json'}")
        mlflow.log_dict({"features": features}, "features.json")

        # Log metrics
        mlflow.log_metric("inertia", pipeline["kmeans"].inertia_)
        mlflow.log_metric("silhouette", silhouette_score(X_emb, labels))
        mlflow.log_metric("stability", stability(X_emb, n_clusters))
        mlflow.log_metric("calinski_harabasz", calinski_harabasz_score(X_emb, labels))
        mlflow.log_metric("davies_bouldin", davies_bouldin_score(X_emb, labels))
        
        # --- 1. Elbow plot (Inertia vs k) ---
        plt.figure(figsize=(6, 4))
        plt.plot(scores.index, scores["inertia"], marker="o", color="red")
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Inertia")
        plt.title("KMeans Elbow Plot")
        plt.tight_layout()

        elbow_path = VIS_DIR / f"kmeans_elbow_{version}.png"
        plt.savefig(elbow_path, dpi=300)
        plt.close()
        mlflow.log_artifact(elbow_path, artifact_path="plots")
        print(f"\nElbow plot saved to:\n- {elbow_path}")

        # 2. Segmentation file
        train_original = pd.read_parquet(train_original_parquet)
        X_labeled = train_original.copy()
        X_labeled["cluster"] = labels.astype(int)
        X_labeled["churn"] = y_train
        out_file = REPORT_DATA / f"segmentation_kmeans_{version}_train.parquet"
        X_labeled.to_parquet(out_file, index=False)
        mlflow.log_artifact(out_file, artifact_path="segmentation")

        # 3. Cluster plot in original feature space
        cols = ["MonthlyCharges", "tenure"]
        plt.figure(figsize=(8, 6))
        for c in sorted(np.unique(labels)):
            m = X_labeled["cluster"] == c
            plt.scatter(
                X_labeled.loc[m, cols[0]],
                X_labeled.loc[m, cols[1]],
                s=12, alpha=0.4, label=f"cluster {c}"
            )
        plt.xlabel(cols[0]); plt.ylabel(cols[1])
        plt.legend()
        plt.title("KMeans Clusters (original space)")
        plt.tight_layout()
        
        plot_path = VIS_DIR / f"kmeans_clusters_{version}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        mlflow.log_artifact(plot_path, artifact_path="plots")
        print(f"\nCluster plot saved to:\n- {plot_path}")

        # 4. Boxplots of numeric features by cluster
        order = sorted(X_labeled["cluster"].unique())
        melted = X_labeled.melt(
            id_vars="cluster", value_vars=["MonthlyCharges", "tenure"],
            var_name="feature", value_name="value"
        )
        g = sns.catplot(
            data=melted, x="cluster", y="value", col="feature",
            kind="box", col_wrap=2, order=order,
            sharey=False, height=4, aspect=1.2
        )
        g.set_axis_labels("cluster", "")
        g.set_titles("{col_name} by cluster")
        g.fig.subplots_adjust(top=0.85)
        g.fig.suptitle("Numeric Features by KMeans Cluster", fontsize=14)

        plot_path = VIS_DIR / f"kmeans_boxplots_{version}.png"
        g.savefig(plot_path, dpi=300)
        plt.close()
        mlflow.log_artifact(plot_path, artifact_path="plots")
        print(f"\nBoxplot saved to:\n- {plot_path}")

        # Log pipeline model (PCA + KMeans)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="pipeline_model",
            input_example=X_train.iloc[:1],
            signature=mlflow.models.infer_signature(X_train,  pipeline.predict(X_train))
        )

        print(f"\nMLflow run logged with {n_clusters} clusters")

    # --- Cluster profile ---
    prof = (
        X_labeled.groupby("cluster")
        .agg(n=("churn", "size"),
             churn_rate=("churn", "mean"),
             tenure_med=("tenure", "median"),
             monthly_mean=("MonthlyCharges", "mean"))
        .assign(ratio=lambda d: d.n / d.n.sum())
        .round(3)
    )
    print("\nCluster profiles:\n", prof)

    return X_labeled


if __name__ == "__main__":
    train_engineered_path = FEATURE_DATA / "churn_v1_train_engineered_features.parquet"
    train_original_path = PREPROCESSED_DATA / "churn_v1_train.parquet"
    X_labeled = run_kmeans(
        train_engineered_parquet=train_engineered_path,
        train_original_parquet=train_original_path,
        n_clusters=3,
        version="v1"
    )
