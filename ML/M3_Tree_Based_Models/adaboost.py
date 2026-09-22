from __future__ import annotations

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def train_adaboost(
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Train AdaBoost Classifier.
    """

    base_estimator = DecisionTreeClassifier(
        max_depth=2,
        random_state=42,
    )

    try:

        model = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=150,
            learning_rate=0.8,
            random_state=42,
        )

    except TypeError:

        # Compatibility with older sklearn.
        model = AdaBoostClassifier(
            base_estimator=base_estimator,
            n_estimators=150,
            learning_rate=0.8,
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
        "model": "AdaBoost",

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