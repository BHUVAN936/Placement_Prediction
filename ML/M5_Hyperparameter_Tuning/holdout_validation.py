from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def run_holdout_validation(X: pd.DataFrame, y: pd.Series, test_size: float = 0.20, seed: int = 42) -> dict:
    """
    Performs Holdout Validation (80% Train, 20% Test) on 50 dataset sample records.
    Calculates Training Accuracy, Testing Accuracy, Precision, Recall, F1, and ROC-AUC.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y if len(np.unique(y)) > 1 else None
    )

    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=seed)
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else test_preds

    train_acc = float(accuracy_score(y_train, train_preds))
    test_acc = float(accuracy_score(y_test, test_preds))
    prec = float(precision_score(y_test, test_preds, zero_division=0))
    rec = float(recall_score(y_test, test_preds, zero_division=0))
    f1 = float(f1_score(y_test, test_preds, zero_division=0))
    auc = float(roc_auc_score(y_test, test_probs)) if len(np.unique(y_test)) > 1 else 1.0

    return {
        "algorithm": "Holdout Validation",
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_accuracy": round(train_acc * 100, 2),
        "test_accuracy": round(test_acc * 100, 2),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "formula": "Training Data = 80% N | Testing Data = 20% N; Test Accuracy = Correct / Total_Test",
        "explanation": f"Holdout Validation splits the dataset into {len(X_train)} training rows (80%) and {len(X_test)} validation rows (20%). It achieves {round(test_acc*100, 2)}% accuracy on unseen test data."
    }
