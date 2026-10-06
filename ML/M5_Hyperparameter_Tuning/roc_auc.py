from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def run_roc_auc_analysis(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> dict:
    """
    Performs ROC-AUC Analysis on dataset sample records.
    Calculates FPR, TPR, optimal decision threshold, Confusion Matrix, and ROC-AUC score.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=seed)

    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=seed)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
    preds = model.predict(X_test)

    auc_score = float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else 1.0
    fpr, tpr, thresholds = roc_curve(y_test, probs)

    # Youden's J statistic for optimal threshold selection
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = float(thresholds[best_idx]) if len(thresholds) > best_idx else 0.5

    cm = confusion_matrix(y_test, preds).tolist()

    roc_curve_samples = []
    for i in range(min(8, len(fpr))):
        roc_curve_samples.append({
            "Threshold": round(float(thresholds[i]), 4) if i < len(thresholds) else 0.0,
            "FPR": round(float(fpr[i]), 4),
            "TPR": round(float(tpr[i]), 4)
        })

    return {
        "algorithm": "ROC-AUC Analysis",
        "roc_auc_score": round(auc_score, 4),
        "optimal_threshold": round(optimal_threshold, 4),
        "confusion_matrix": cm,
        "roc_curve_samples": roc_curve_samples,
        "formula": "FPR = FP / (FP + TN); TPR = TP / (TP + FN); AUC = Area under ROC Curve",
        "explanation": f"ROC-AUC evaluates classification performance across all probability thresholds. Achieved ROC-AUC score: {round(auc_score, 4)} with optimal decision threshold at {round(optimal_threshold, 4)}."
    }
