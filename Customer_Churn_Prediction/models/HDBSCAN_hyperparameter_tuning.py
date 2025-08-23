# models/HDBSCAN_tuning.py
import numpy as np
import pandas as pd
import json
from itertools import product

from sklearn.metrics import silhouette_score
import umap, hdbscan

from src.myproject.utils.paths import (
    CONFIG_PATH, FEATURE_DATA,
    REPORT_DATA
)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

SEED = config["UMAP"]["random_state"]

def hyperparameter_tuning(train_engineered_parquet):
    """HDBSCAN hyperparameter tuning for k≈3"""
    
    # Load data
    train = pd.read_parquet(train_engineered_parquet)
    target_col = "Churn"
    features = [c for c in train.columns if c != target_col]
    X_train = train[features]

    # Define a smaller, coarse grid
    param_grid = {
        "n_neighbors": [10, 15],        # Controls the balance b/w LOCAL & GLOBAL structure
        "min_dist": [0.0, 0.1],         # tight vs looser embedding
        "n_components": [10, 20],        # compact vs richer space
        "min_cluster_size": [500, 600],  # Controls the MIN. granularity of a cluster.
        "cluster_selection_method": ["eom"],  # only eom first, then try leaf later
        "metric": ["cosine", "euclidean"]
    }

    # Collect results
    results = []

    # Total number of combinations
    param_combinations = list(product(*param_grid.values()))
    n_search = len(param_combinations)

    for i, params in enumerate(param_combinations, 1):
        print(f"\n{i}/{n_search} search in progress...\n")

        p = dict(zip(param_grid.keys(), params))

        # UMAP
        emb = umap.UMAP(
            n_neighbors=p["n_neighbors"],
            min_dist=p["min_dist"],
            n_components=p["n_components"],
            metric=p["metric"],
            random_state=SEED
        ).fit_transform(X_train)

        # HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=p["min_cluster_size"],
            cluster_selection_method=p["cluster_selection_method"],
            prediction_data=True
        ).fit(emb)

        labels = clusterer.labels_
        mask = labels != -1
        n_clusters = np.unique(labels[mask]).size
        noise_frac = 1 - mask.mean()
        sil = silhouette_score(emb[mask], labels[mask]) if n_clusters > 1 else np.nan
        pers = clusterer.cluster_persistence_

        results.append({
            **p,
            "clusters": n_clusters,
            "noise%": round(noise_frac*100, 2), # 5 ~ 20 %
            "silhouette": round(sil, 3), # > 0.5
            "persistence": np.round(pers, 3) # individually > 0.5
        })

    # Save to CSV
    res_df = pd.DataFrame(results).sort_values("silhouette", ascending=False)
    print(res_df[res_df['clusters'] == 3].head())
    res_df.to_csv(REPORT_DATA / "HDBSCAN_tuning.csv", index=False)
    print("Searches completed.")
    print("\nSave tuned HDBSCAN hyperparameters to:\n -", REPORT_DATA / "HDBSCAN_tuning.csv")


if __name__ == "__main__":
    hyperparameter_tuning(
        train_engineered_parquet=FEATURE_DATA / "churn_v1_train_engineered_features.parquet",
    )