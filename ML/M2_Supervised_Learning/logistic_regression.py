from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def train_logistic_regression(
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Train standard Logistic Regression.
    """

    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=3000,
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

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    try:

        roc_auc = roc_auc_score(
            y_test,
            probabilities,
        )

    except ValueError:

        roc_auc = 0.0

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    metrics = {

        "model":
            "Logistic Regression",

        "Accuracy":
            float(accuracy),

        "Precision":
            float(precision),

        "Recall":
            float(recall),

        "F1":
            float(f1),

        "ROC_AUC":
            float(roc_auc),

        "Confusion_Matrix":
            cm.tolist(),

    }

    return (
        model,
        predictions,
        probabilities,
        metrics,
    )