from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_pca(
    X,
    feature_names=None,
    output_dir=None,
    variance_threshold=0.95
):
    """
    PCA dimensionality reduction.

    Keeps enough components to explain approximately
    the requested amount of variance.
    """

    X = np.asarray(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(
        n_components=variance_threshold,
        random_state=42
    )

    X_pca = pca.fit_transform(X_scaled)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(
        explained_variance
    )

    n_components = X_pca.shape[1]

    if feature_names is None:
        feature_names = [
            f"Feature_{i + 1}"
            for i in range(X.shape[1])
        ]

    component_names = [
        f"PC{i + 1}"
        for i in range(n_components)
    ]

    explained_df = pd.DataFrame({
        "Component": component_names,
        "ExplainedVarianceRatio": explained_variance,
        "CumulativeVariance": cumulative_variance
    })

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=component_names
    )

    results = {
        "algorithm": "PCA",
        "original_features": int(X.shape[1]),
        "reduced_features": int(n_components),
        "variance_threshold": float(
            variance_threshold
        ),
        "total_explained_variance": float(
            np.sum(explained_variance)
        )
    }

    if output_dir is not None:

        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        explained_df.to_csv(
            output_dir / "pca_explained_variance.csv",
            index=False
        )

        loadings.to_csv(
            output_dir / "pca_loadings.csv"
        )

        np.save(
            output_dir / "pca_transformed.npy",
            X_pca
        )

        # Variance chart
        plt.figure(figsize=(10, 6))

        plt.plot(
            range(1, len(explained_variance) + 1),
            cumulative_variance,
            marker="o"
        )

        plt.axhline(
            variance_threshold,
            linestyle="--"
        )

        plt.xlabel("Number of Principal Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("PCA Cumulative Explained Variance")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        plt.savefig(
            output_dir / "pca_explained_variance.png",
            dpi=150
        )

        plt.close()

        # 2D PCA projection
        if X_pca.shape[1] >= 2:

            plt.figure(figsize=(9, 7))

            plt.scatter(
                X_pca[:, 0],
                X_pca[:, 1],
                s=10,
                alpha=0.5
            )

            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title("PCA 2D Projection")
            plt.grid(True, alpha=0.3)

            plt.tight_layout()

            plt.savefig(
                output_dir / "pca_2d_projection.png",
                dpi=150
            )

            plt.close()

    return (
        pca,
        scaler,
        X_pca,
        explained_df,
        loadings,
        results
    )