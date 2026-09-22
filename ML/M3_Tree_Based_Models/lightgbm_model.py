from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def train_lightgbm(
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Train LightGBM Classifier.
    """

    try:

        from lightgbm import LGBMClassifier

    except ImportError as exc:

        raise ImportError(
            "LightGBM is not installed.\n"
            "Install it using:\n"
            "pip install lightgbm"
        ) from exc

    model = LGBMClassifier(

        n_estimators=200,

        learning_rate=0.05,

        max_depth=6,

        num_leaves=31,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="binary",

        random_state=42,

        n_jobs=-1,

        verbosity=-1,

    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    metrics = {
        "model": "LightGBM",

        "Accuracy": float(
            accuracy_score(
                y_test,
                predictions,
            )
        ),

        "Precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),

        "Recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),

        "F1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),

        "ROC_AUC": float(
            roc_auc_score(
                y_test,
                probabilities,
            )
        ),

        "Confusion_Matrix":
            confusion_matrix(
                y_test,
                predictions,
            ).tolist(),
    }

    return (
        model,
        predictions,
        probabilities,
        metrics,
    )