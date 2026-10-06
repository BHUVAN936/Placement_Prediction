from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def run_probability_calibration(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> dict:
    """
    Performs Probability Calibration (Isotonic / Sigmoid Platt Scaling) on dataset sample records.
    Calculates Brier Score before and after probability calibration.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=seed)

    base_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=seed)
    base_model.fit(X_train, y_train)

    uncalibrated_probs = base_model.predict_proba(X_test)[:, 1]
    brier_before = float(brier_score_loss(y_test, uncalibrated_probs))

    calibrated_model = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=5)
    calibrated_model.fit(X_train, y_train)

    calibrated_probs = calibrated_model.predict_proba(X_test)[:, 1]
    brier_after = float(brier_score_loss(y_test, calibrated_probs))

    calibration_details = []
    for i in range(min(10, len(y_test))):
        calibration_details.append({
            "Sample": i + 1,
            "Actual_Class": int(y_test.iloc[i]),
            "Uncalibrated_Prob": round(float(uncalibrated_probs[i]), 4),
            "Calibrated_Prob": round(float(calibrated_probs[i]), 4),
            "Diff": round(float(calibrated_probs[i] - uncalibrated_probs[i]), 4)
        })

    return {
        "algorithm": "Probability Calibration",
        "brier_score_before": round(brier_before, 4),
        "brier_score_after": round(brier_after, 4),
        "brier_improvement": round(brier_before - brier_after, 4),
        "calibration_method": "Sigmoid Platt Scaling",
        "calibration_samples": len(y_test),
        "calibration_details": calibration_details,
        "formula": "Brier Score = (1/N) * Sum((p_i - y_i)^2)",
        "explanation": f"Probability Calibration refines raw classifier outputs into reliable probabilities. Brier Loss improved from {round(brier_before, 4)} to {round(brier_after, 4)}."
    }
