# ============================================================
# M4 - UNSUPERVISED LEARNING
# ============================================================
#
# Algorithms:
#   1. K-Means
#   2. K-Means++
#   3. Hierarchical Clustering
#   4. DBSCAN
#   5. PCA
#
# Windows-safe version
# ============================================================

import os
import sys
import json
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"

if sys.platform == "win32":

    try:

        sys.stdout.reconfigure(
            encoding="utf-8",
            errors="replace"
        )

        sys.stderr.reconfigure(
            encoding="utf-8",
            errors="replace"
        )

    except Exception:
        pass


import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import joblib

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN
)

from sklearn.decomposition import PCA

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    __file__
).resolve().parents[2]

DATA_DIR = ROOT / "data"

M1_REPORT_DIR = (
    ROOT / "reports" / "M1"
)

REPORT_DIR = (
    ROOT / "reports" / "M4"
)

MODEL_DIR = (
    ROOT / "models" / "M4"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TARGETS TO EXCLUDE
# ============================================================

EXCLUDED_COLUMNS = [

    "PlacementStatus",

    "Salary Package"
]


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    engineered_file = (
        M1_REPORT_DIR
        / "m1_engineered_dataset.csv"
    )

    original_file = (
        DATA_DIR
        / "placement_predict_50k Dataset (2).csv"
    )


    if engineered_file.exists():

        print(
            "Loading M1 engineered dataset..."
        )

        return pd.read_csv(
            engineered_file
        )


    if original_file.exists():

        print(
            "Loading original dataset..."
        )

        return pd.read_csv(
            original_file
        )


    csv_files = list(
        DATA_DIR.glob("*.csv")
    )


    if not csv_files:

        raise FileNotFoundError(
            "No dataset found."
        )


    return pd.read_csv(
        csv_files[0]
    )


# ============================================================
# PREPARE NUMERIC DATA
# ============================================================

def prepare_features(df):

    X = df.drop(

        columns=EXCLUDED_COLUMNS,

        errors="ignore"
    ).copy()


    # Numeric features only
    X = X.select_dtypes(
        include=[np.number]
    )


    # Remove empty columns
    X = X.dropna(
        axis=1,
        how="all"
    )


    if X.empty:

        raise ValueError(
            "No numeric features available "
            "for M4."
        )


    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )


    # Median imputation
    imputer = SimpleImputer(
        strategy="median"
    )


    X_imputed = (
        imputer.fit_transform(
            X
        )
    )


    # Standard scaling
    scaler = StandardScaler()


    X_scaled = (
        scaler.fit_transform(
            X_imputed
        )
    )


    return (
        X,
        X_scaled,
        imputer,
        scaler
    )


# ============================================================
# SAFE SILHOUETTE
# ============================================================

def get_cluster_metrics(
    X,
    labels
):

    unique = np.unique(
        labels
    )


    # Remove DBSCAN noise
    mask = (
        labels != -1
    )


    if (
        -1 in unique
        and np.sum(mask) > 0
    ):

        X_valid = X[mask]

        labels_valid = labels[
            mask
        ]

    else:

        X_valid = X

        labels_valid = labels


    unique_valid = np.unique(
        labels_valid
    )


    if len(unique_valid) < 2:

        return {

            "silhouette_score":
                0.0,

            "calinski_harabasz_score":
                0.0
        }


    try:

        silhouette = silhouette_score(

            X_valid,

            labels_valid
        )

    except Exception:

        silhouette = 0.0


    try:

        calinski = (
            calinski_harabasz_score(

                X_valid,

                labels_valid
            )
        )

    except Exception:

        calinski = 0.0


    return {

        "silhouette_score":
            float(silhouette),

        "calinski_harabasz_score":
            float(calinski)
    }


# ============================================================
# SAVE DISTRIBUTION
# ============================================================

def save_distribution(
    labels,
    name
):

    unique, counts = np.unique(
        labels,
        return_counts=True
    )


    df = pd.DataFrame({

        "Cluster":
            unique,

        "Count":
            counts
    })


    safe_name = (
        name
        .lower()
        .replace(" ", "_")
        .replace("+", "plus")
    )


    df.to_csv(

        REPORT_DIR
        / f"{safe_name}_cluster_distribution.csv",

        index=False
    )


    plt.figure(
        figsize=(8, 5)
    )


    plt.bar(

        [
            str(value)
            for value in unique
        ],

        counts
    )


    plt.xlabel(
        "Cluster"
    )


    plt.ylabel(
        "Number of Records"
    )


    plt.title(
        f"{name} - Cluster Distribution"
    )


    plt.tight_layout()


    plt.savefig(

        REPORT_DIR
        / f"{safe_name}_cluster_distribution.png",

        dpi=150
    )


    plt.close()


# ============================================================
# PCA
# ============================================================

def run_pca(
    X_scaled,
    feature_names
):

    print()
    print(
        "Running PCA..."
    )


    pca = PCA(
        n_components=0.95
    )


    X_pca = (
        pca.fit_transform(
            X_scaled
        )
    )


    explained = (
        pca.explained_variance_ratio_
    )


    cumulative = np.cumsum(
        explained
    )


    pca_results = {

        "algorithm":
            "PCA",

        "original_features":
            int(X_scaled.shape[1]),

        "reduced_features":
            int(X_pca.shape[1]),

        "explained_variance":
            float(
                explained.sum()
            )
    }


    # --------------------------------------------------------
    # Variance CSV
    # --------------------------------------------------------

    variance_df = pd.DataFrame({

        "Component": [

            f"PC{i + 1}"

            for i in range(
                len(explained)
            )
        ],

        "ExplainedVariance":
            explained,

        "CumulativeVariance":
            cumulative
    })


    variance_df.to_csv(

        REPORT_DIR
        / "pca_explained_variance.csv",

        index=False
    )


    # --------------------------------------------------------
    # Loadings
    # --------------------------------------------------------

    loading_df = pd.DataFrame(

        pca.components_.T,

        index=feature_names,

        columns=[
            f"PC{i + 1}"
            for i in range(
                X_pca.shape[1]
            )
        ]
    )


    loading_df.to_csv(

        REPORT_DIR
        / "pca_loadings.csv"
    )


    # --------------------------------------------------------
    # Save transformed
    # --------------------------------------------------------

    np.save(

        REPORT_DIR
        / "pca_transformed.npy",

        X_pca
    )


    # --------------------------------------------------------
    # Variance chart
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(

        range(
            1,
            len(cumulative) + 1
        ),

        cumulative,

        marker="o"
    )


    plt.axhline(
        0.95,
        linestyle="--"
    )


    plt.xlabel(
        "Principal Components"
    )


    plt.ylabel(
        "Cumulative Explained Variance"
    )


    plt.title(
        "PCA Explained Variance"
    )


    plt.grid(
        alpha=0.3
    )


    plt.tight_layout()


    plt.savefig(

        REPORT_DIR
        / "pca_explained_variance.png",

        dpi=150
    )


    plt.close()


    # --------------------------------------------------------
    # 2D projection
    # --------------------------------------------------------

    if X_pca.shape[1] >= 2:

        plt.figure(
            figsize=(9, 7)
        )


        plt.scatter(

            X_pca[:, 0],

            X_pca[:, 1],

            s=10,

            alpha=0.5
        )


        plt.xlabel(
            "PC1"
        )


        plt.ylabel(
            "PC2"
        )


        plt.title(
            "PCA 2D Projection"
        )


        plt.grid(
            alpha=0.3
        )


        plt.tight_layout()


        plt.savefig(

            REPORT_DIR
            / "pca_2d_projection.png",

            dpi=150
        )


        plt.close()


    joblib.dump(

        pca,

        MODEL_DIR
        / "pca_model.joblib"
    )


    return (
        pca,
        X_pca,
        pca_results
    )


# ============================================================
# MAIN
# ============================================================

def run_m4():

    print()
    print("=" * 70)
    print("M4 - UNSUPERVISED LEARNING")
    print("=" * 70)


    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print()
    print("[1/5] Loading dataset...")


    df = load_dataset()


    print(
        f"Rows: {len(df):,}"
    )


    print(
        f"Columns: {len(df.columns)}"
    )


    # --------------------------------------------------------
    # PREPARE
    # --------------------------------------------------------

    print()
    print(
        "[2/5] Preparing unsupervised features..."
    )


    (
        X_df,
        X_scaled,
        imputer,
        scaler
    ) = prepare_features(
        df
    )


    feature_names = (
        X_df.columns.tolist()
    )


    print(
        f"Numeric features: "
        f"{len(feature_names)}"
    )


    print(
        f"Excluded targets: "
        f"{', '.join(EXCLUDED_COLUMNS)}"
    )


    joblib.dump(

        imputer,

        MODEL_DIR
        / "m4_imputer.joblib"
    )


    joblib.dump(

        scaler,

        MODEL_DIR
        / "m4_scaler.joblib"
    )


    joblib.dump(

        feature_names,

        MODEL_DIR
        / "m4_feature_names.joblib"
    )


    pd.DataFrame({

        "Feature":
            feature_names

    }).to_csv(

        REPORT_DIR
        / "m4_feature_names.csv",

        index=False
    )


    # ========================================================
    # PCA
    # ========================================================

    print()
    print(
        "[3/5] PCA..."
    )


    (
        pca_model,
        X_pca,
        pca_results
    ) = run_pca(

        X_scaled,

        feature_names
    )


    print(
        f"Original dimensions: "
        f"{pca_results['original_features']}"
    )


    print(
        f"PCA dimensions: "
        f"{pca_results['reduced_features']}"
    )


    print(
        f"Explained variance: "
        f"{pca_results['explained_variance']:.4f}"
    )


    # ========================================================
    # CLUSTERING
    # ========================================================

    print()
    print(
        "[4/5] Running clustering algorithms..."
    )


    results = []


    # --------------------------------------------------------
    # K-MEANS
    # --------------------------------------------------------

    print()
    print(
        "Training K-Means..."
    )


    kmeans = KMeans(

        n_clusters=3,

        init="random",

        n_init=10,

        max_iter=300,

        random_state=42
    )


    kmeans_labels = (
        kmeans.fit_predict(
            X_scaled
        )
    )


    kmeans_metrics = (
        get_cluster_metrics(

            X_scaled,

            kmeans_labels
        )
    )


    kmeans_result = {

        "Algorithm":
            "K-Means",

        "Clusters":
            3,

        "Inertia":
            float(
                kmeans.inertia_
            ),

        **kmeans_metrics
    }


    results.append(
        kmeans_result
    )


    joblib.dump(

        kmeans,

        MODEL_DIR
        / "kmeans_model.joblib"
    )


    save_distribution(

        kmeans_labels,

        "K-Means"
    )


    # PCA visualization
    if X_pca.shape[1] >= 2:

        plt.figure(
            figsize=(9, 7)
        )


        plt.scatter(

            X_pca[:, 0],

            X_pca[:, 1],

            c=kmeans_labels,

            s=10,

            alpha=0.5
        )


        plt.xlabel(
            "PC1"
        )


        plt.ylabel(
            "PC2"
        )


        plt.title(
            "K-Means Clusters - PCA Projection"
        )


        plt.tight_layout()


        plt.savefig(

            REPORT_DIR
            / "kmeans_pca_projection.png",

            dpi=150
        )


        plt.close()


    print(
        f"  Silhouette: "
        f"{kmeans_metrics['silhouette_score']:.4f}"
    )


    # --------------------------------------------------------
    # K-MEANS++
    # --------------------------------------------------------

    print()
    print(
        "Training K-Means++..."
    )


    kpp = KMeans(

        n_clusters=3,

        init="k-means++",

        n_init=10,

        max_iter=300,

        random_state=42
    )


    kpp_labels = (
        kpp.fit_predict(
            X_scaled
        )
    )


    kpp_metrics = (
        get_cluster_metrics(

            X_scaled,

            kpp_labels
        )
    )


    kpp_result = {

        "Algorithm":
            "K-Means++",

        "Clusters":
            3,

        "Inertia":
            float(
                kpp.inertia_
            ),

        **kpp_metrics
    }


    results.append(
        kpp_result
    )


    joblib.dump(

        kpp,

        MODEL_DIR
        / "kmeans_plus_plus_model.joblib"
    )


    save_distribution(

        kpp_labels,

        "K-Means++"
    )


    print(
        f"  Silhouette: "
        f"{kpp_metrics['silhouette_score']:.4f}"
    )


    # --------------------------------------------------------
    # HIERARCHICAL
    # --------------------------------------------------------

    print()
    print(
        "Training Hierarchical Clustering..."
    )


    MAX_HIERARCHICAL_ROWS = 5000


    if len(X_scaled) > MAX_HIERARCHICAL_ROWS:

        rng = np.random.default_rng(
            42
        )


        indices = rng.choice(

            len(X_scaled),

            size=MAX_HIERARCHICAL_ROWS,

            replace=False
        )


        X_hierarchical = (
            X_scaled[indices]
        )


        X_pca_hierarchical = (
            X_pca[indices]
        )

    else:

        X_hierarchical = X_scaled

        X_pca_hierarchical = X_pca


    hierarchical = AgglomerativeClustering(

        n_clusters=3,

        linkage="ward"
    )


    hierarchical_labels = (
        hierarchical.fit_predict(
            X_hierarchical
        )
    )


    hierarchical_metrics = (
        get_cluster_metrics(

            X_hierarchical,

            hierarchical_labels
        )
    )


    hierarchical_result = {

        "Algorithm":
            "Hierarchical Clustering",

        "Clusters":
            3,

        "SampleSize":
            int(
                len(X_hierarchical)
            ),

        **hierarchical_metrics
    }


    results.append(
        hierarchical_result
    )


    joblib.dump(

        hierarchical,

        MODEL_DIR
        / "hierarchical_model.joblib"
    )


    save_distribution(

        hierarchical_labels,

        "Hierarchical Clustering"
    )


    print(
        f"  Sample size: "
        f"{len(X_hierarchical)}"
    )


    print(
        f"  Silhouette: "
        f"{hierarchical_metrics['silhouette_score']:.4f}"
    )


    # --------------------------------------------------------
    # DBSCAN
    # --------------------------------------------------------

    print()
    print(
        "Training DBSCAN..."
    )


    MAX_DBSCAN_ROWS = 10000


    if len(X_scaled) > MAX_DBSCAN_ROWS:

        rng = np.random.default_rng(
            42
        )


        indices = rng.choice(

            len(X_scaled),

            size=MAX_DBSCAN_ROWS,

            replace=False
        )


        X_dbscan = (
            X_scaled[indices]
        )

    else:

        X_dbscan = X_scaled


    dbscan = DBSCAN(

        eps=0.8,

        min_samples=10,

        n_jobs=-1
    )


    dbscan_labels = (
        dbscan.fit_predict(
            X_dbscan
        )
    )


    dbscan_metrics = (
        get_cluster_metrics(

            X_dbscan,

            dbscan_labels
        )
    )


    cluster_count = len(

        set(
            dbscan_labels
        ) - {-1}

    )


    noise_count = int(

        np.sum(
            dbscan_labels == -1
        )

    )


    dbscan_result = {

        "Algorithm":
            "DBSCAN",

        "Clusters":
            int(cluster_count),

        "NoisePoints":
            noise_count,

        "SampleSize":
            int(len(X_dbscan)),

        **dbscan_metrics
    }


    results.append(
        dbscan_result
    )


    joblib.dump(

        dbscan,

        MODEL_DIR
        / "dbscan_model.joblib"
    )


    save_distribution(

        dbscan_labels,

        "DBSCAN"
    )


    print(
        f"  Clusters: "
        f"{cluster_count}"
    )


    print(
        f"  Noise points: "
        f"{noise_count}"
    )


    print(
        f"  Silhouette: "
        f"{dbscan_metrics['silhouette_score']:.4f}"
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    print()
    print(
        "[5/5] Saving M4 results..."
    )


    results_df = pd.DataFrame(
        results
    )


    results_df.to_csv(

        REPORT_DIR
        / "m4_summary.csv",

        index=False
    )


    results_df.to_csv(

        REPORT_DIR
        / "m4_results.csv",

        index=False
    )


    results_json = {

        "module":
            "M4",

        "module_name":
            "Unsupervised Learning",

        "status":
            "completed",

        "dataset_rows":
            int(len(df)),

        "numeric_features":
            int(len(feature_names)),

        "excluded_targets":
            EXCLUDED_COLUMNS,

        "pca":
            pca_results,

        "clustering":
            results
    }


    with open(

        REPORT_DIR
        / "m4_results.json",

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            results_json,

            file,

            indent=4
        )


    # ========================================================
    # SUCCESS
    # ========================================================

    print()
    print("=" * 70)
    print("M4 COMPLETED SUCCESSFULLY")
    print("=" * 70)


    print(
        f"Result JSON : "
        f"{REPORT_DIR / 'm4_results.json'}"
    )


    print(
        f"Charts      : "
        f"{REPORT_DIR}"
    )


    print(
        f"Models      : "
        f"{MODEL_DIR}"
    )


    print()
    print(
        "M4 pipeline:"
    )


    print(
        "Load -> Prepare -> PCA -> "
        "K-Means -> K-Means++ -> "
        "Hierarchical -> DBSCAN -> Evaluation"
    )


    print()
    print(
        "M4 completed. Full ML pipeline is ready."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        run_m4()

    except Exception as error:

        print()
        print("=" * 70)
        print("M4 TRAINING FAILED")
        print("=" * 70)

        print(
            f"Error: {error}"
        )

        print("=" * 70)

        import traceback

        traceback.print_exc()

        raise