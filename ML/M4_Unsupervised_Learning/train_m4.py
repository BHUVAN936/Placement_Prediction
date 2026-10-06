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
#   6. UMAP
#   7. t-SNE
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
from sklearn.manifold import TSNE

try:
    import umap.umap_ as umap
except ImportError:
    umap = None

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
# UMAP + t-SNE DIMENSIONALITY REDUCTION
# ============================================================

def run_umap_tsne(X_scaled, cluster_labels=None, sample_size=5000):
    """Generate UMAP and t-SNE 2D + 3D graphs independently.

    The same professor-provided preprocessing approach is retained:
    deterministic sampling, PCA pre-reduction, UMAP neighbourhood
    parameters and t-SNE parameters. Only the practical sample size and
    the additional 3D projections are extended here.
    """

    print()
    print("Running UMAP + t-SNE...")

    n_rows = len(X_scaled)
    actual_size = min(sample_size, n_rows)

    rng = np.random.default_rng(42)

    if n_rows > actual_size:
        indices = np.sort(
            rng.choice(n_rows, size=actual_size, replace=False)
        )
    else:
        indices = np.arange(n_rows)

    X_sample = X_scaled[indices]

    # PCA pre-reduction for faster/stabler manifold learning.
    pca_dim = min(30, X_sample.shape[1], X_sample.shape[0] - 1)

    if pca_dim >= 2 and X_sample.shape[1] > pca_dim:
        manifold_input = PCA(
            n_components=pca_dim,
            random_state=42
        ).fit_transform(X_sample)
    else:
        manifold_input = X_sample

    labels_sample = None
    if cluster_labels is not None:
        labels_sample = np.asarray(cluster_labels)[indices]

    results = {
        "sample_size": int(actual_size),
        "source_rows": int(n_rows),
        "pre_reduction_dimensions": int(manifold_input.shape[1]),
        "umap": {"status": "not_generated"},
        "tsne": {"status": "not_generated"}
    }

    # ========================================================
    # UMAP - 2D + 3D
    # ========================================================

    if umap is None:
        print("  UMAP: NOT GENERATED")
        print("  Install with: pip install umap-learn")
        results["umap"] = {
            "status": "not_generated",
            "reason": "umap-learn is not installed"
        }
    else:
        try:
            # ------------------------------------------------
            # UMAP 2D
            # ------------------------------------------------
            reducer_2d = umap.UMAP(
                n_components=2,
                n_neighbors=min(15, max(2, actual_size - 1)),
                min_dist=0.10,
                metric="euclidean",
                random_state=42,
                transform_seed=42
            )

            X_umap_2d = reducer_2d.fit_transform(manifold_input)

            umap_2d_df = pd.DataFrame({
                "SourceIndex": indices,
                "UMAP1": X_umap_2d[:, 0],
                "UMAP2": X_umap_2d[:, 1]
            })

            if labels_sample is not None:
                umap_2d_df["Cluster"] = labels_sample

            umap_2d_df.to_csv(
                REPORT_DIR / "umap_2d_embedding.csv",
                index=False
            )

            np.save(
                REPORT_DIR / "umap_2d_transformed.npy",
                X_umap_2d
            )

            joblib.dump(
                reducer_2d,
                MODEL_DIR / "umap_2d_model.joblib"
            )

            plt.figure(figsize=(10, 7))

            if labels_sample is not None:
                plt.scatter(
                    X_umap_2d[:, 0],
                    X_umap_2d[:, 1],
                    c=labels_sample,
                    s=12,
                    alpha=0.65
                )
                plt.colorbar(label="K-Means Cluster")
            else:
                plt.scatter(
                    X_umap_2d[:, 0],
                    X_umap_2d[:, 1],
                    s=12,
                    alpha=0.65
                )

            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
            plt.title(
                f"UMAP 2D Projection ({actual_size:,} sampled students)"
            )
            plt.grid(alpha=0.25)
            plt.tight_layout()
            plt.savefig(
                REPORT_DIR / "umap_2d_projection.png",
                dpi=180,
                bbox_inches="tight"
            )
            plt.close()

            # ------------------------------------------------
            # UMAP neighbourhood graph
            # ------------------------------------------------
            graph = reducer_2d.graph_.tocsr()
            graph_nodes = min(400, actual_size)
            graph_positions = X_umap_2d[:graph_nodes]

            plt.figure(figsize=(11, 8))

            for source in range(graph_nodes):
                edge_start = graph.indptr[source]
                edge_end = graph.indptr[source + 1]

                for edge_pos in range(edge_start, edge_end):
                    target = graph.indices[edge_pos]

                    if target >= graph_nodes or target <= source:
                        continue

                    weight = float(graph.data[edge_pos])

                    if weight < 0.20:
                        continue

                    plt.plot(
                        [
                            graph_positions[source, 0],
                            graph_positions[target, 0]
                        ],
                        [
                            graph_positions[source, 1],
                            graph_positions[target, 1]
                        ],
                        alpha=min(
                            0.35,
                            max(0.05, weight * 0.35)
                        ),
                        linewidth=0.5
                    )

            if labels_sample is not None:
                plt.scatter(
                    graph_positions[:, 0],
                    graph_positions[:, 1],
                    c=labels_sample[:graph_nodes],
                    s=18,
                    alpha=0.85
                )
                plt.colorbar(label="K-Means Cluster")
            else:
                plt.scatter(
                    graph_positions[:, 0],
                    graph_positions[:, 1],
                    s=18,
                    alpha=0.85
                )

            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
            plt.title(
                "UMAP Neighborhood Graph (First 400 sampled nodes)"
            )
            plt.grid(alpha=0.20)
            plt.tight_layout()
            plt.savefig(
                REPORT_DIR / "umap_neighborhood_graph.png",
                dpi=180,
                bbox_inches="tight"
            )
            plt.close()

            # ------------------------------------------------
            # UMAP 3D
            # ------------------------------------------------
            reducer_3d = umap.UMAP(
                n_components=3,
                n_neighbors=min(15, max(2, actual_size - 1)),
                min_dist=0.10,
                metric="euclidean",
                random_state=42,
                transform_seed=42
            )

            X_umap_3d = reducer_3d.fit_transform(manifold_input)

            umap_3d_df = pd.DataFrame({
                "SourceIndex": indices,
                "UMAP1": X_umap_3d[:, 0],
                "UMAP2": X_umap_3d[:, 1],
                "UMAP3": X_umap_3d[:, 2]
            })

            if labels_sample is not None:
                umap_3d_df["Cluster"] = labels_sample

            umap_3d_df.to_csv(
                REPORT_DIR / "umap_3d_embedding.csv",
                index=False
            )

            np.save(
                REPORT_DIR / "umap_3d_transformed.npy",
                X_umap_3d
            )

            joblib.dump(
                reducer_3d,
                MODEL_DIR / "umap_3d_model.joblib"
            )

            figure = plt.figure(figsize=(10, 8))
            axis = figure.add_subplot(111, projection="3d")

            if labels_sample is not None:
                scatter = axis.scatter(
                    X_umap_3d[:, 0],
                    X_umap_3d[:, 1],
                    X_umap_3d[:, 2],
                    c=labels_sample,
                    s=10,
                    alpha=0.65
                )
                figure.colorbar(
                    scatter,
                    ax=axis,
                    label="K-Means Cluster",
                    pad=0.10
                )
            else:
                axis.scatter(
                    X_umap_3d[:, 0],
                    X_umap_3d[:, 1],
                    X_umap_3d[:, 2],
                    s=10,
                    alpha=0.65
                )

            axis.set_xlabel("UMAP 1")
            axis.set_ylabel("UMAP 2")
            axis.set_zlabel("UMAP 3")
            axis.set_title(
                f"UMAP 3D Projection ({actual_size:,} sampled students)"
            )
            figure.tight_layout()
            figure.savefig(
                REPORT_DIR / "umap_3d_projection.png",
                dpi=180,
                bbox_inches="tight"
            )
            plt.close(figure)

            results["umap"] = {
                "status": "completed",
                "sample_size": int(actual_size),
                "input_dimensions": int(manifold_input.shape[1]),
                "output_dimensions": [2, 3],
                "n_neighbors": min(15, max(2, actual_size - 1)),
                "min_dist": 0.10,
                "metric": "euclidean"
            }

            print(
                f"  UMAP: 5,000 rows -> 2D + 3D"
                if actual_size == 5000
                else f"  UMAP: {actual_size:,} rows -> 2D + 3D"
            )

        except Exception as error:
            print(f"  UMAP: FAILED - {error}")
            results["umap"] = {
                "status": "failed",
                "reason": str(error)
            }

    # ========================================================
    # t-SNE - 2D + 3D
    # ========================================================

    try:
        perplexity = min(
            30.0,
            max(5.0, (actual_size - 1) / 3.0)
        )

        # ------------------------------------------------
        # t-SNE 2D
        # ------------------------------------------------
        tsne_2d = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            max_iter=1000,
            random_state=42
        )

        X_tsne_2d = tsne_2d.fit_transform(manifold_input)

        tsne_2d_df = pd.DataFrame({
            "SourceIndex": indices,
            "tSNE1": X_tsne_2d[:, 0],
            "tSNE2": X_tsne_2d[:, 1]
        })

        if labels_sample is not None:
            tsne_2d_df["Cluster"] = labels_sample

        tsne_2d_df.to_csv(
            REPORT_DIR / "tsne_2d_embedding.csv",
            index=False
        )

        np.save(
            REPORT_DIR / "tsne_2d_transformed.npy",
            X_tsne_2d
        )

        plt.figure(figsize=(10, 7))

        if labels_sample is not None:
            plt.scatter(
                X_tsne_2d[:, 0],
                X_tsne_2d[:, 1],
                c=labels_sample,
                s=12,
                alpha=0.65
            )
            plt.colorbar(label="K-Means Cluster")
        else:
            plt.scatter(
                X_tsne_2d[:, 0],
                X_tsne_2d[:, 1],
                s=12,
                alpha=0.65
            )

        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title(
            f"t-SNE 2D Projection ({actual_size:,} sampled students)"
        )
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(
            REPORT_DIR / "tsne_2d_projection.png",
            dpi=180,
            bbox_inches="tight"
        )
        plt.close()

        # ------------------------------------------------
        # t-SNE 3D
        # ------------------------------------------------
        tsne_3d = TSNE(
            n_components=3,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            max_iter=1000,
            random_state=42
        )

        X_tsne_3d = tsne_3d.fit_transform(manifold_input)

        tsne_3d_df = pd.DataFrame({
            "SourceIndex": indices,
            "tSNE1": X_tsne_3d[:, 0],
            "tSNE2": X_tsne_3d[:, 1],
            "tSNE3": X_tsne_3d[:, 2]
        })

        if labels_sample is not None:
            tsne_3d_df["Cluster"] = labels_sample

        tsne_3d_df.to_csv(
            REPORT_DIR / "tsne_3d_embedding.csv",
            index=False
        )

        np.save(
            REPORT_DIR / "tsne_3d_transformed.npy",
            X_tsne_3d
        )

        figure = plt.figure(figsize=(10, 8))
        axis = figure.add_subplot(111, projection="3d")

        if labels_sample is not None:
            scatter = axis.scatter(
                X_tsne_3d[:, 0],
                X_tsne_3d[:, 1],
                X_tsne_3d[:, 2],
                c=labels_sample,
                s=10,
                alpha=0.65
            )
            figure.colorbar(
                scatter,
                ax=axis,
                label="K-Means Cluster",
                pad=0.10
            )
        else:
            axis.scatter(
                X_tsne_3d[:, 0],
                X_tsne_3d[:, 1],
                X_tsne_3d[:, 2],
                s=10,
                alpha=0.65
            )

        axis.set_xlabel("t-SNE 1")
        axis.set_ylabel("t-SNE 2")
        axis.set_zlabel("t-SNE 3")
        axis.set_title(
            f"t-SNE 3D Projection ({actual_size:,} sampled students)"
        )
        figure.tight_layout()
        figure.savefig(
            REPORT_DIR / "tsne_3d_projection.png",
            dpi=180,
            bbox_inches="tight"
        )
        plt.close(figure)

        results["tsne"] = {
            "status": "completed",
            "sample_size": int(actual_size),
            "input_dimensions": int(manifold_input.shape[1]),
            "output_dimensions": [2, 3],
            "perplexity": float(perplexity),
            "iterations": 1000,
            "learning_rate": "auto",
            "init": "pca"
        }

        print(
            f"  t-SNE: 5,000 rows -> 2D + 3D"
            if actual_size == 5000
            else f"  t-SNE: {actual_size:,} rows -> 2D + 3D"
        )

    except Exception as error:
        print(f"  t-SNE: FAILED - {error}")
        results["tsne"] = {
            "status": "failed",
            "reason": str(error)
        }

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_rows = []

    summary_rows.append({
        "Algorithm": "UMAP",
        "SampleSize": int(actual_size),
        "InputDimensions": int(manifold_input.shape[1]),
        "OutputDimensions": "2D + 3D",
        "n_neighbors": 15,
        "min_dist": 0.10,
        "Metric": "euclidean",
        "Status": results["umap"].get(
            "status",
            "not_generated"
        )
    })

    summary_rows.append({
        "Algorithm": "t-SNE",
        "SampleSize": int(actual_size),
        "InputDimensions": int(manifold_input.shape[1]),
        "OutputDimensions": "2D + 3D",
        "Perplexity": results["tsne"].get(
            "perplexity",
            float(perplexity)
        ),
        "Metric": "euclidean",
        "Status": results["tsne"].get(
            "status",
            "not_generated"
        )
    })

    pd.DataFrame(summary_rows).to_csv(
        REPORT_DIR / "m4_dimensionality_reduction.csv",
        index=False
    )

    return results

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


    # ========================================================
    # UMAP + t-SNE
    # ========================================================

    print()
    print(
        "[5/5] UMAP + t-SNE dimensionality reduction..."
    )

    # Run dimensionality reduction independently so a missing optional
    # UMAP package does not prevent the remaining M4 algorithms/graphs
    # from being generated.
    try:
        manifold_results = run_umap_tsne(
            X_scaled,
            cluster_labels=kmeans_labels,
            sample_size=5000
        )
    except ImportError as manifold_error:
        print(f"  UMAP unavailable: {manifold_error}")
        print("  Continuing with the remaining M4 algorithms.")
        manifold_results = {
            "umap": {
                "status": "not_generated",
                "reason": str(manifold_error)
            }
        }
    except Exception as manifold_error:
        print(f"  UMAP/t-SNE generation failed: {manifold_error}")
        print("  Continuing with the remaining M4 algorithms.")
        manifold_results = {
            "status": "not_generated",
            "reason": str(manifold_error)
        }


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

    # PCA visualization for K-Means++
    if X_pca.shape[1] >= 2:

        plt.figure(figsize=(9, 7))
        plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=kpp_labels,
            s=10,
            alpha=0.5
        )
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("K-Means++ Clusters - PCA Projection")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(
            REPORT_DIR / "kmeans_plus_plus_pca_projection.png",
            dpi=160
        )
        plt.close()


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

    # Hierarchical dendrogram using a readable 120-row sample.
    # The clustering result above remains based on the configured sample.
    try:
        from scipy.cluster.hierarchy import linkage, dendrogram

        dendro_n = min(120, len(X_hierarchical))
        dendro_X = X_hierarchical[:dendro_n]
        linkage_matrix = linkage(dendro_X, method="ward")

        plt.figure(figsize=(12, 7))
        dendrogram(
            linkage_matrix,
            no_labels=True,
            color_threshold=None
        )
        plt.xlabel("Student observations")
        plt.ylabel("Ward linkage distance")
        plt.title("Hierarchical Clustering Dendrogram")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(
            REPORT_DIR / "hierarchical_dendrogram.png",
            dpi=170
        )
        plt.close()
    except Exception as dendrogram_error:
        print(f"  Dendrogram generation skipped: {dendrogram_error}")


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

    # DBSCAN 2D visualization using the first two PCA dimensions.
    if X_pca.shape[1] >= 2:

        if len(X_dbscan) == len(X_scaled):
            db_pca = X_pca
        else:
            # Recover the same deterministic sample for the PCA coordinates.
            rng = np.random.default_rng(42)
            db_indices = rng.choice(
                len(X_scaled),
                size=len(X_dbscan),
                replace=False
            )
            db_pca = X_pca[db_indices]

        plt.figure(figsize=(9, 7))
        plt.scatter(
            db_pca[:, 0],
            db_pca[:, 1],
            c=dbscan_labels,
            s=12,
            alpha=0.65
        )
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("DBSCAN Clusters - PCA Projection")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(
            REPORT_DIR / "dbscan_2d_projection.png",
            dpi=160
        )
        plt.close()

    # Euclidean-distance visualization for the first source observation.
    try:
        first_point = X_scaled[0]
        center_distances = [
            float(np.linalg.norm(first_point - center))
            for center in kmeans.cluster_centers_
        ]

        plt.figure(figsize=(8, 5))
        plt.bar(
            [f"Cluster {i}" for i in range(len(center_distances))],
            center_distances
        )
        plt.xlabel("K-Means centroid")
        plt.ylabel("Euclidean distance")
        plt.title("Euclidean Distance from First Student to K-Means Centroids")
        plt.grid(axis="y", alpha=0.2)
        plt.tight_layout()
        plt.savefig(
            REPORT_DIR / "euclidean_distance_example.png",
            dpi=160
        )
        plt.close()
    except Exception as distance_error:
        print(f"  Euclidean-distance graph skipped: {distance_error}")


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
        "[6/6] Saving M4 results..."
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

        "dimensionality_reduction":
            manifold_results,

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
        "K-Means -> UMAP -> t-SNE -> "
        "K-Means++ -> Hierarchical -> DBSCAN -> Evaluation"
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