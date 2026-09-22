from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def train_gradient_boosting(
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Train Gradient Boosting Classifier.
    """

    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
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
        "model": "Gradient Boosting",

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