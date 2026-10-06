from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def run_kfold_cross_validation(X: pd.DataFrame, y: pd.Series, k: int = 5, seed: int = 42) -> dict:
    """
    Performs 5-Fold Cross Validation (k=5) on dataset sample records.
    Calculates individual fold accuracy scores S1, S2, S3, S4, S5, mean score, and std dev.
    """
    cv = KFold(n_splits=k, shuffle=True, random_state=seed)
    fold_scores = []
    fold_details = []

    fold_idx = 1
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=seed)
        model.fit(X_train, y_train)

        val_preds = model.predict(X_val)
        acc = float(accuracy_score(y_val, val_preds))
        fold_scores.append(acc)

        fold_details.append({
            "Fold": fold_idx,
            "Train_Samples": len(X_train),
            "Val_Samples": len(X_val),
            "Accuracy_Score": round(acc * 100, 2)
        })
        fold_idx += 1

    mean_acc = float(np.mean(fold_scores))
    std_acc = float(np.std(fold_scores))

    return {
        "algorithm": "5-Fold Cross Validation",
        "k": k,
        "fold_scores": [round(s * 100, 2) for s in fold_scores],
        "mean_accuracy": round(mean_acc * 100, 2),
        "std_deviation": round(std_acc * 100, 2),
        "fold_details": fold_details,
        "formula": "Mean CV Score = (1/k) * Sum(Fold_i Score); Std Dev = sqrt((1/k) * Sum((S_i - Mean)^2))",
        "explanation": f"5-Fold Cross-Validation (k={k}) partitions data into {k} equal folds. Each fold acts as validation while remaining folds train the model. Mean CV score: {round(mean_acc*100, 2)}% +/- {round(std_acc*100, 2)}%."
    }
