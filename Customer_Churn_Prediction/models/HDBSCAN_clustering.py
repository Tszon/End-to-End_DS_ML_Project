# models/HDBSCAN_clustering.py
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
import umap, hdbscan
import mlflow, mlflow.sklearn

from src.myproject.utils.paths import CONFIG_PATH, PREPROCESSED_DATA, FEATURE_DATA, VIS_DIR

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

def run_hdbscan(
        train_engineered_parquet, 
        train_original_parquet,
        n_clusters=3,
        version="v1",
        experiment_name="HDBSCAN_Clustering"
    ):
    """Perform HDBSCAN clustering, log results to MLflow, save segmentation + plots."""
    # --- Load ---
    train = pd.read_parquet(train_engineered_parquet)
    target_col = "Churn"
    features = [c for c in train.columns if c != target_col]

    X_train = train[features].copy()
    y_train = train[target_col].copy()

    # Setup UMAP, HDBSCAN with the optimal set of hyperparameters
    emb_best = umap.UMAP(
        n_neighbors=config["UMAP"]["n_neighbors"],
        min_dist=config["UMAP"]["min_dist"],
        n_components=config["UMAP"]["n_components"],
        metric=config["UMAP"]["metric"],
        random_state=config["UMAP"]["random_state"]
    ).fit_transform(X_train)

    clusterer_best = hdbscan.HDBSCAN(
        min_cluster_size=config["HDBSCAN"]["min_cluster_size"],
        cluster_selection_method=config["HDBSCAN"]["cluster_selection_method"],
        prediction_data=config["HDBSCAN"]["prediction_data"]
    ).fit(emb_best)

    # Cluster evaluation
    labels = clusterer_best.labels_
    mask = labels != -1
    n_clusters = np.unique(labels[mask]).size

    noise_frac = (labels == -1).mean()
    noise_n = int((labels == -1).sum())

    # --- Set experiment + log MLflow run ---
    if mlflow.active_run():
        mlflow.end_run()

    mlflow.set_experiment("Customer_Churn_Project")
    with mlflow.start_run(run_name=f"hdbscan_{n_clusters}_clusters"):
        
        # Set tags
        mlflow.set_tag("experiment", f"{experiment_name}_{version}")

        # Log params
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("random_state", config["UMAP"]["random_state"])
        mlflow.log_param("n_features", len(features))

        # Log features used
        with open(FEATURE_DATA / f"features_hdbscan_{version}.json", "w") as f:
            json.dump({"features": features}, f, indent=4)
            print(f"\nFeatures used saved to:\n- {FEATURE_DATA / f'features_hdbscan_{version}.json'}")
        mlflow.log_dict({"features": features}, "features.json")
        
        # Log metrics
        sil = silhouette_score(emb_best[mask], labels[mask]) if mask.sum() > n_clusters else np.nan
        mlflow.log_metric("silhouette", sil)
        mlflow.log_metric("noise_frac", noise_frac)
        mlflow.log_metric("noise_n", noise_n)
        mlflow.log_metric("cluster_persistence_mean", clusterer_best.cluster_persistence_.mean())

        print(f"\n=== HDBSCAN results ===\n")
        print(f"n_clusters={n_clusters}, noise={noise_frac:.2%} ({noise_n}), silhouette={sil:.3f}, cluster_pers={clusterer_best.cluster_persistence_}")

        # 1. Cluster plot in UMAP space
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=emb_best[:, 0], y=emb_best[:, 1], hue=labels, s=8, palette="tab10", alpha=0.8)
        plt.title("UMAP + HDBSCAN clusters (-1 = noise)")
        plt.legend()
        plt.tight_layout()

        plot_path = VIS_DIR / f"hdbscan_clusters_umap_{version}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        mlflow.log_artifact(plot_path, artifact_path="plots")
        print(f"\nCluster plot in UMAP space saved to:\n- {plot_path}")

        # 2. Cluster plot in original feature space
        labels = clusterer_best.labels_
        probs  = clusterer_best.probabilities_

        train_original = pd.read_parquet(train_original_parquet)
        X_labeled = train_original.copy()
        X_labeled["cluster"] = labels
        X_labeled["prob"] = probs

        cols = ["MonthlyCharges", "tenure"]
        u = sorted(X_labeled["cluster"].unique())

        plt.figure(figsize=(8, 6))
        for c in u:
            m = X_labeled["cluster"] == c
            plt.scatter(X_labeled.loc[m, cols[0]],
                        X_labeled.loc[m, cols[1]],
                        s=12, alpha=0.6, label=f"cluster {c}")
        plt.xlabel(cols[0]); plt.ylabel(cols[1])
        plt.title("HDBSCAN Clusters in Original Feature Space")
        plt.legend()
        plt.tight_layout()

        plot_path = VIS_DIR / f"hdbscan_clusters_{version}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        mlflow.log_artifact(plot_path, artifact_path="plots")
        print(f"\nCluster plot saved to:\n- {plot_path}")

        # 3. Distribution plots of "tenure", "MonthlyCharges" by cluster
        plot_cols = ["tenure", "MonthlyCharges"]
        x_labels = ["Tenure", "Monthly Charges"]

        for (col, label) in zip(plot_cols, x_labels):
            plt.figure(figsize=(8, 6))
            sns.histplot(x=X_labeled[col], hue=X_labeled['cluster'], palette="tab10", kde=True, alpha=0.3, bins=20)
            plt.title(f"{label} Distribution Across HDBSCAN Clusters")
            plt.xlabel(label)
            plot_path = VIS_DIR / f"hdbscan_{col}_distribution.png"
            plt.savefig(plot_path, dpi=300)
            plt.close()
            mlflow.log_artifact(plot_path, artifact_path="plots")
            print(f"\nDistribution plots in saved to:\n- {plot_path}")

        # Check how confident the assigned clusters are
        bins = np.arange(0, 1.2, 0.2)
        bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
        X_labeled['prob_label'] = pd.cut(X_labeled['prob'], bins=bins, labels=bin_labels)

        # Name the clusters based on their characteristics
        X_labeled['cluster_label'] = X_labeled['cluster'].map({
            -1: "Drifters", 0: "Budget Loyalists", 
            1: "At-Risk Premiums", 2: "Balanced Mainstream"
            })

        # 4. Boxplots of numeric features by cluster
        order = sorted(X_labeled["cluster"].unique())
        melted = X_labeled.melt(id_vars="cluster", value_vars=["MonthlyCharges", "tenure"], var_name="feature", value_name="value")
        g = sns.catplot(
            data=melted, x="cluster", y="value", col="feature",
            kind="box", col_wrap=2, order=order,
            sharey=False, height=3.5, aspect=1.2
        )
        g.set_axis_labels("cluster", "")
        g.set_titles("{col_name} by cluster")
        g.fig.suptitle("Numeric Features by HDBSCAN Clusters", y=1.02)
        
        plot_path = VIS_DIR / "hdbscan_boxplots.png"
        g.savefig(plot_path, dpi=300)
        plt.close()
        mlflow.log_artifact(plot_path, artifact_path="plots")
        print(f"\nBoxplot saved to:\n- {plot_path}")


        # 5. Stacked bar plots of categorical features by cluster
        def plot_cat_stacked_grid(df, cat_cols, cluster_col="cluster",
                                ncols=3, top_k=None, title="Categoricals by HDBSCAN Clusters"):
            """
            df must contain `cluster_col` and the categorical columns in `cat_cols`.
            """
            n = len(cat_cols)
            nrows = n // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 3.6*nrows), squeeze=False)

            for i, col in enumerate(cat_cols):
                ax = axes[i // ncols, i % ncols]

                # robust proportion table: crosstab + normalize
                ct = pd.crosstab(
                    df[cluster_col].astype(str),
                    df[col].astype("string").fillna("<NA>"),
                    normalize="index"
                )

                # keep only top_k levels and bucket the rest
                if top_k is not None and ct.shape[1] > top_k:
                    overall = df[col].astype("string").fillna("<NA>").value_counts()
                    keep = list(overall.index[:top_k].astype(str))
                    other = ct.drop(columns=[c for c in ct.columns if c in keep], errors="ignore")
                    ct = ct.reindex(columns=keep, fill_value=0.0)
                    if other.shape[1] > 0:
                        ct["Other"] = other.sum(axis=1)

                ct.plot(kind="bar", stacked=True, ax=ax, legend=True)
                ax.set_title(col, fontsize=20)
                ax.set_xlabel("cluster")
                ax.set_ylabel("proportion")
                ax.set_ylim(0, 1)
                ax.tick_params(axis="x", rotation=0)
                ax.legend(fontsize=12, bbox_to_anchor=(1.02, 1), loc="upper left",)

            # remove any empty panels
            for j in range(i+1, nrows*ncols):
                fig.delaxes(axes[j // ncols, j % ncols])

            fig.suptitle(title, y=1.02, fontsize=25)
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            plot_path = VIS_DIR / f"hdbscan_cat_by_clusters.png"
            plt.savefig(plot_path, dpi=300)
            plt.close()
            mlflow.log_artifact(plot_path, artifact_path="plots")
            print(f"\nBar plots saved to:\n- {plot_path}")

        # choose as many categoricals as you want
        cols_to_plot = [
            "Gender",
            "SeniorCitizen", "Partner", "Dependents",
            "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
            "Contract", "PaperlessBilling", "PaymentMethod"
            # add as many as you like…
        ]
        plot_cat_stacked_grid(X_labeled, cols_to_plot, ncols=3, top_k=8)

        # Log HDBSCAN model
        mlflow.sklearn.log_model(
            sk_model=clusterer_best,
            name="hdbscan_model"
        )

        print(f"\nMLflow run logged with {n_clusters} clusters")

    # --- Cluster profile ---
    prof = (X_labeled.assign(churn=y_train)
            .groupby("cluster")
            .agg(n=("churn","size"),
                churn_rate=("churn","mean"),
                tenure_med=("tenure","median"),
                monthly_mean=("MonthlyCharges","mean"))
            .assign(prop_pct=lambda d: round(d.n/d.n.sum() * 100, 1))
            .round(3))
    print("\nCluster profiles:\n", prof)

    return X_labeled


if __name__ == "__main__":
    train_engineered_path = FEATURE_DATA / "churn_v1_train_engineered_features.parquet"
    train_original_path = PREPROCESSED_DATA / "churn_v1_train.parquet"
    X_labeled = run_hdbscan(
        train_engineered_parquet=train_engineered_path,
        train_original_parquet=train_original_path,
        n_clusters=3,
        version="v1"
    )