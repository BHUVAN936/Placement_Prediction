import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score
)


def run_hierarchical_clustering(
    X,
    n_clusters=3,
    linkage="ward"
):
    """
    Agglomerative Hierarchical Clustering.

    Ward linkage is used by default.
    """

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage
    )

    labels = model.fit_predict(X)

    metrics = {
        "algorithm": "Hierarchical Clustering",
        "n_clusters": int(n_clusters),
        "linkage": linkage
    }

    unique_labels = np.unique(labels)

    if len(unique_labels) > 1:
        metrics["silhouette_score"] = float(
            silhouette_score(X, labels)
        )

        metrics["calinski_harabasz_score"] = float(
            calinski_harabasz_score(X, labels)
        )
    else:
        metrics["silhouette_score"] = 0.0
        metrics["calinski_harabasz_score"] = 0.0

    return model, labels, metrics