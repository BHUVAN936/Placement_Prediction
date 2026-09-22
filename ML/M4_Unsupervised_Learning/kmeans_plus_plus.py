import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score


def run_kmeans_plus_plus(X, n_clusters=3, random_state=42):
    """
    K-Means++ clustering.

    K-Means++ improves the initialization of cluster centroids
    compared with random initialization.
    """

    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=random_state
    )

    labels = model.fit_predict(X)

    metrics = {
        "algorithm": "K-Means++",
        "n_clusters": int(n_clusters),
        "inertia": float(model.inertia_)
    }

    if len(np.unique(labels)) > 1:
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