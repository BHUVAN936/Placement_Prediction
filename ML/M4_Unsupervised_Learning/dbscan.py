import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score
)


def run_dbscan(
    X,
    eps=0.8,
    min_samples=10
):
    """
    DBSCAN clustering.

    Noise points receive label -1.
    """

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean",
        n_jobs=-1
    )

    labels = model.fit_predict(X)

    unique_labels = np.unique(labels)

    n_clusters = len(
        set(labels) - {-1}
    )

    n_noise = int(
        np.sum(labels == -1)
    )

    metrics = {
        "algorithm": "DBSCAN",
        "eps": float(eps),
        "min_samples": int(min_samples),
        "n_clusters": int(n_clusters),
        "noise_points": n_noise
    }

    # Calculate clustering metrics only for non-noise points
    mask = labels != -1

    if (
        n_clusters > 1
        and np.sum(mask) > n_clusters
    ):
        X_valid = X[mask]
        labels_valid = labels[mask]

        metrics["silhouette_score"] = float(
            silhouette_score(
                X_valid,
                labels_valid
            )
        )

        metrics["calinski_harabasz_score"] = float(
            calinski_harabasz_score(
                X_valid,
                labels_valid
            )
        )
    else:
        metrics["silhouette_score"] = 0.0
        metrics["calinski_harabasz_score"] = 0.0

    return model, labels, metrics