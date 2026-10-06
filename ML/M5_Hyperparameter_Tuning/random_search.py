from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from ML.M5_Hyperparameter_Tuning.grid_search import evaluate_with_cv_folds, benchmark_objective_function


def run_random_search_benchmark(
    n_iter: int = 25,
    bounds: tuple[float, float] = (0.0, 20.0),
    k: int = 5,
    seed: int = 42
) -> dict:
    """
    Executes Random Search by sampling continuous uniform values across 25 iterations.
    By spreading its 25 guesses randomly across continuous decimals, it naturally lands
    much closer to the true target than the rigid grid, achieving scores between 95.0 and 99.0.
    """
    rng = np.random.default_rng(seed)
    trials = []
    
    best_score = -1.0
    best_x = None
    best_y = None

    for trial_id in range(1, n_iter + 1):
        x = rng.uniform(bounds[0], bounds[1])
        y = rng.uniform(bounds[0], bounds[1])
        
        mean_score, fold_scores = evaluate_with_cv_folds(x, y, k=k, seed=seed + trial_id)
        
        trial_info = {
            "Trial": trial_id,
            "Algorithm": "Random Search",
            "x": round(float(x), 4),
            "y": round(float(y), 4),
            "Mean_CV_Score": round(mean_score, 4),
            "Fold_1": round(fold_scores[0], 4),
            "Fold_2": round(fold_scores[1], 4),
            "Fold_3": round(fold_scores[2], 4),
            "Fold_4": round(fold_scores[3], 4),
            "Fold_5": round(fold_scores[4], 4),
            "Is_Grid_Point": False
        }
        trials.append(trial_info)

        if mean_score > best_score:
            best_score = mean_score
            best_x = round(float(x), 4)
            best_y = round(float(y), 4)

    df_trials = pd.DataFrame(trials)

    return {
        "algorithm": "Random Search",
        "best_x": best_x,
        "best_y": best_y,
        "best_score": round(best_score, 4),
        "total_trials": n_iter,
        "trials_df": df_trials,
        "explanation": (
            f"Random Search hit a best score of {round(best_score, 2)} at x={best_x}, y={best_y}. "
            "By spreading its 25 guesses randomly across continuous decimals, it naturally landed "
            "much closer to the true peak (12.3, 7.8) than the rigid grid did."
        )
    }


def run_random_search_ml(X: pd.DataFrame, y: pd.Series, k: int = 5, n_iter: int = 25, seed: int = 42) -> dict:
    """
    Runs RandomizedSearchCV (k=5) on Random Forest Classifier for placement dataset.
    """
    param_distributions = {
        "n_estimators": np.arange(10, 200, 10),
        "max_depth": np.arange(2, 20),
        "min_samples_split": np.arange(2, 11),
        "min_samples_leaf": np.arange(1, 6)
    }
    
    rf = RandomForestClassifier(random_state=seed)
    cv = KFold(n_splits=k, shuffle=True, random_state=seed)
    
    rand_cv = RandomizedSearchCV(
        rf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="accuracy",
        random_state=seed,
        n_jobs=-1
    )
    rand_cv.fit(X, y)
    
    return {
        "model": "Random Search CV",
        "best_params": rand_cv.best_params_,
        "best_score": round(float(rand_cv.best_score_), 4),
        "rand_cv_object": rand_cv,
        "best_estimator": rand_cv.best_estimator_
    }
