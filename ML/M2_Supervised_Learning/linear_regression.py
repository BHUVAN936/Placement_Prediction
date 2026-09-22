from __future__ import annotations

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def train_linear_regression(
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Train Linear Regression and return
    model + regression metrics.
    """

    model = LinearRegression()

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
            "Linear Regression",

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