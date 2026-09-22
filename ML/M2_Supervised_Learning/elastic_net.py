from __future__ import annotations

import numpy as np

from sklearn.linear_model import ElasticNet

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def train_elastic_net(
    X_train,
    X_test,
    y_train,
    y_test,
    alpha=0.001,
    l1_ratio=0.5,
):
    """
    Train Elastic Net Regression.
    """

    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=10000,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    mse = mean_squared_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions,
    )

    metrics = {

        "model":
            "Elastic Net",

        "alpha":
            float(alpha),

        "l1_ratio":
            float(l1_ratio),

        "MAE":
            float(mae),

        "MSE":
            float(mse),

        "RMSE":
            float(rmse),

        "R2":
            float(r2),

    }

    return model, predictions, metrics