# ============================================================
# M2 - SUPERVISED LEARNING
# ============================================================
#
# Regression:
#   1. Linear Regression
#   2. Ridge Regression
#   3. Lasso Regression
#   4. Elastic Net
#
# Classification:
#   5. Logistic Regression
#   6. Ridge Logistic Regression
#
# Windows-safe UTF-8 / ASCII console output
# ============================================================

import os
import sys
import json
import warnings
from pathlib import Path

# ============================================================
# WINDOWS UTF-8
# ============================================================

os.environ["PYTHONIOENCODING"] = "utf-8"

if sys.platform == "win32":

    try:
        sys.stdout.reconfigure(
            encoding="utf-8",
            errors="replace"
        )

        sys.stderr.reconfigure(
            encoding="utf-8",
            errors="replace"
        )

    except Exception:
        pass


import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    __file__
).resolve().parents[2]

DATA_DIR = ROOT / "data"

M1_REPORT_DIR = (
    ROOT / "reports" / "M1"
)

REPORT_DIR = (
    ROOT / "reports" / "M2"
)

MODEL_DIR = (
    ROOT / "models" / "M2"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATASET
# ============================================================

DATASET_NAME = (
    "placement_predict_50k Dataset (2).csv"
)

CLASSIFICATION_TARGET = "PlacementStatus"

REGRESSION_TARGET = "Salary Package"


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    engineered_file = (
        M1_REPORT_DIR
        / "m1_engineered_dataset.csv"
    )

    original_file = (
        DATA_DIR
        / DATASET_NAME
    )

    if engineered_file.exists():

        print(
            "Loading M1 engineered dataset..."
        )

        return pd.read_csv(
            engineered_file
        )

    if original_file.exists():

        print(
            "Loading original dataset..."
        )

        return pd.read_csv(
            original_file
        )

    csv_files = list(
        DATA_DIR.glob("*.csv")
    )

    if not csv_files:

        raise FileNotFoundError(
            "No CSV dataset found."
        )

    return pd.read_csv(
        csv_files[0]
    )


# ============================================================
# ONE HOT ENCODER
# ============================================================

def create_encoder():

    try:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

    except TypeError:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor(X):

    categorical_columns = (
        X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
    )

    numerical_columns = (
        X.select_dtypes(
            include=[np.number]
        ).columns.tolist()
    )

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            create_encoder()
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numerical_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ])

    return preprocessor


# ============================================================
# REGRESSION METRICS
# ============================================================

def regression_metrics(
    y_true,
    y_pred
):

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(mse)

    return {

        "MAE":
            float(
                mean_absolute_error(
                    y_true,
                    y_pred
                )
            ),

        "MSE":
            float(mse),

        "RMSE":
            float(rmse),

        "R2":
            float(
                r2_score(
                    y_true,
                    y_pred
                )
            )
    }


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def classification_metrics(
    y_true,
    y_pred,
    y_probability=None
):

    result = {

        "Accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred
                )
            ),

        "Precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
            ),

        "Recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
            ),

        "F1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
            )
    }

    if y_probability is not None:

        try:

            result["ROC_AUC"] = float(
                roc_auc_score(
                    y_true,
                    y_probability
                )
            )

        except Exception:

            result["ROC_AUC"] = 0.0

    else:

        result["ROC_AUC"] = 0.0


    return result


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    y_true,
    y_pred,
    model_name
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        f"{model_name} - Confusion Matrix"
    )

    plt.colorbar()

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.tight_layout()

    filename = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        REPORT_DIR / filename,
        dpi=150
    )

    plt.close()


# ============================================================
# SAVE REGRESSION COMPARISON
# ============================================================

def save_regression_chart(
    results
):

    names = [
        item["Model"]
        for item in results
    ]

    r2_values = [
        item["R2"]
        for item in results
    ]

    rmse_values = [
        item["RMSE"]
        for item in results
    ]

    x = np.arange(
        len(names)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.bar(
        x - width / 2,
        r2_values,
        width,
        label="R2"
    )

    # Normalize RMSE only for visualization
    max_rmse = max(
        rmse_values
    ) if rmse_values else 1

    normalized_rmse = [
        value / max_rmse
        for value in rmse_values
    ]

    ax.bar(
        x + width / 2,
        normalized_rmse,
        width,
        label="Normalized RMSE"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        names,
        rotation=30,
        ha="right"
    )

    ax.set_title(
        "M2 Regression Model Comparison"
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "m2_regression_comparison.png",
        dpi=150
    )

    plt.close()


# ============================================================
# SAVE CLASSIFICATION COMPARISON
# ============================================================

def save_classification_chart(
    results
):

    names = [
        item["Model"]
        for item in results
    ]

    accuracy = [
        item["Accuracy"]
        for item in results
    ]

    f1_values = [
        item["F1"]
        for item in results
    ]

    x = np.arange(
        len(names)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        x - width / 2,
        accuracy,
        width,
        label="Accuracy"
    )

    ax.bar(
        x + width / 2,
        f1_values,
        width,
        label="F1"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        names,
        rotation=20,
        ha="right"
    )

    ax.set_ylim(
        0,
        1.05
    )

    ax.set_title(
        "M2 Classification Model Comparison"
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "m2_classification_comparison.png",
        dpi=150
    )

    plt.close()


def save_split_chart(train_len, test_len):
    plt.figure(figsize=(7, 5))
    plt.bar(["Training Set (80%)", "Testing Set (20%)"], [train_len, test_len], color=['#3498db', '#e74c3c'], alpha=0.85)
    plt.ylabel("Number of Observations")
    plt.title("M2 Train-Test Split (80/20 Holdout Partitioning)")
    for i, v in enumerate([train_len, test_len]):
        plt.text(i, v + (train_len * 0.01), f"{v:,}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "train_test_split.png", dpi=150)
    plt.close()


def save_scatter_chart(y_true, y_pred, model_name, filename):
    plt.figure(figsize=(7, 5))
    y_true_np = np.array(y_true)
    sample_idx = np.random.choice(len(y_true_np), min(500, len(y_true_np)), replace=False)
    y_t_s, y_p_s = y_true_np[sample_idx], y_pred[sample_idx]
    
    plt.scatter(y_t_s, y_p_s, alpha=0.5, color='#2980b9', edgecolors='none', s=25)
    min_v = min(np.min(y_t_s), np.min(y_p_s))
    max_v = max(np.max(y_t_s), np.max(y_p_s))
    plt.plot([min_v, max_v], [min_v, max_v], 'r--', label='Ideal Perfect Fit (y = x)')
    
    plt.xlabel("Actual Salary Package")
    plt.ylabel("Predicted Salary Package")
    plt.title(f"{model_name}: Actual vs Predicted Salary Package")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / filename, dpi=150)
    plt.close()


def save_sigmoid_chart(y_true, probas, model_name, filename):
    if probas is None: return
    plt.figure(figsize=(7, 5))
    y_true_np = np.array(y_true)
    sorted_idx = np.argsort(probas)
    x_axis = np.linspace(-5, 5, len(probas))
    plt.plot(x_axis, probas[sorted_idx], color='#8e44ad', linewidth=2, label='Sigmoid Probability Curve')
    plt.scatter(x_axis[::25], y_true_np[sorted_idx][::25], color='#e67e22', alpha=0.7, label='Observed Placement (0/1)')
    plt.axhline(0.5, color='gray', linestyle=':', label='Decision Threshold (0.5)')
    plt.xlabel("Linear Score (z)")
    plt.ylabel("Predicted Probability P(Placed)")
    plt.title(f"{model_name}: Sigmoid Curve & Probabilities")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / filename, dpi=150)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def run_m2():

    print()
    print("=" * 70)
    print("M2 - SUPERVISED LEARNING")
    print("=" * 70)


    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print()
    print("[1/4] Loading dataset...")

    df = load_dataset()

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )


    # --------------------------------------------------------
    # CLEAN TARGETS
    # --------------------------------------------------------

    print()
    print("[2/4] Preparing targets...")


    if CLASSIFICATION_TARGET not in df.columns:

        raise ValueError(
            f"{CLASSIFICATION_TARGET} "
            "not found."
        )


    if REGRESSION_TARGET not in df.columns:

        raise ValueError(
            f"{REGRESSION_TARGET} "
            "not found."
        )


    df = df.copy()


    # Classification target
    y_class = pd.to_numeric(
        df[CLASSIFICATION_TARGET],
        errors="coerce"
    )


    valid_class = (
        y_class.notna()
    )


    # Regression target
    y_reg = pd.to_numeric(
        df[REGRESSION_TARGET],
        errors="coerce"
    )


    valid_reg = (
        y_reg.notna()
    )


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    X = df.drop(
        columns=[
            CLASSIFICATION_TARGET,
            REGRESSION_TARGET
        ],
        errors="ignore"
    )


    # Do not use object columns with all missing values
    empty_columns = [
        column
        for column in X.columns
        if X[column].isna().all()
    ]

    if empty_columns:

        X = X.drop(
            columns=empty_columns
        )


    print(
        f"Input features: {X.shape[1]}"
    )


    # ========================================================
    # REGRESSION
    # ========================================================

    print()
    print("[3/4] Training regression models...")


    X_reg = X.loc[
        valid_reg
    ].copy()

    y_reg_final = y_reg.loc[
        valid_reg
    ]


    (
        X_train_reg,
        X_test_reg,
        y_train_reg,
        y_test_reg
    ) = train_test_split(
        X_reg,
        y_reg_final,
        test_size=0.20,
        random_state=42
    )


    regression_models = {

        "Linear Regression":
            LinearRegression(),

        "Ridge Regression":
            Ridge(
                alpha=1.0
            ),

        "Lasso Regression":
            Lasso(
                alpha=0.001,
                max_iter=10000
            ),

        "Elastic Net":
            ElasticNet(
                alpha=0.001,
                l1_ratio=0.5,
                max_iter=10000
            )
    }


    regression_results = []


    save_split_chart(len(X_train_reg), len(X_test_reg))

    for name, model in regression_models.items():

        print(
            f"Training {name}..."
        )


        preprocessor = create_preprocessor(
            X_train_reg
        )


        pipeline = Pipeline([

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )
        ])


        pipeline.fit(
            X_train_reg,
            y_train_reg
        )


        predictions = pipeline.predict(
            X_test_reg
        )


        metrics = regression_metrics(
            y_test_reg,
            predictions
        )

        save_scatter_chart(
            y_test_reg,
            predictions,
            name,
            name.lower().replace(" ", "_") + "_data.png"
        )


        row = {

            "Model":
                name,

            "Task":
                "Regression",

            **metrics
        }


        regression_results.append(
            row
        )


        filename = (
            name
            .lower()
            .replace(" ", "_")
            + ".joblib"
        )


        joblib.dump(
            pipeline,
            MODEL_DIR / filename
        )


        print(
            f"  R2   : {metrics['R2']:.4f}"
        )

        print(
            f"  RMSE : {metrics['RMSE']:.4f}"
        )


    save_regression_chart(
        regression_results
    )


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    print()
    print(
        "[4/4] Training classification models..."
    )


    X_class = X.loc[
        valid_class
    ].copy()

    y_class_final = (
        y_class.loc[
            valid_class
        ].astype(int)
    )


    (
        X_train_class,
        X_test_class,
        y_train_class,
        y_test_class
    ) = train_test_split(

        X_class,

        y_class_final,

        test_size=0.20,

        random_state=42,

        stratify=y_class_final
    )


    classification_models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=3000,
                solver="lbfgs"
            ),

        "Ridge Logistic Regression":
            LogisticRegression(
                penalty="l2",
                C=1.0,
                max_iter=3000,
                solver="lbfgs"
            )
    }


    classification_results = []


    for name, model in (
        classification_models.items()
    ):

        print(
            f"Training {name}..."
        )


        preprocessor = create_preprocessor(
            X_train_class
        )


        pipeline = Pipeline([

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )
        ])


        pipeline.fit(
            X_train_class,
            y_train_class
        )


        predictions = pipeline.predict(
            X_test_class
        )


        try:

            probabilities = (
                pipeline.predict_proba(
                    X_test_class
                )[:, 1]
            )

        except Exception:

            probabilities = None


        metrics = classification_metrics(

            y_test_class,

            predictions,

            probabilities
        )

        save_sigmoid_chart(
            y_test_class,
            probabilities,
            name,
            name.lower().replace(" ", "_") + "_data.png"
        )


        row = {

            "Model":
                name,

            "Task":
                "Classification",

            **metrics
        }


        classification_results.append(
            row
        )


        filename = (
            name
            .lower()
            .replace(" ", "_")
            + ".joblib"
        )


        joblib.dump(
            pipeline,
            MODEL_DIR / filename
        )


        save_confusion_matrix(

            y_test_class,

            predictions,

            name
        )


        print(
            f"  Accuracy : "
            f"{metrics['Accuracy']:.4f}"
        )

        print(
            f"  F1       : "
            f"{metrics['F1']:.4f}"
        )


    save_classification_chart(
        classification_results
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    all_results = (
        regression_results
        +
        classification_results
    )


    results_json = {

        "module":
            "M2",

        "module_name":
            "Supervised Learning",

        "status":
            "completed",

        "dataset_rows":
            int(len(df)),

        "input_features":
            int(X.shape[1]),

        "regression_target":
            REGRESSION_TARGET,

        "classification_target":
            CLASSIFICATION_TARGET,

        "regression_models":
            regression_results,

        "classification_models":
            classification_results,

        "models":
            all_results
    }


    with open(
        REPORT_DIR
        / "m2_results.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results_json,
            file,
            indent=4
        )


    # --------------------------------------------------------
    # SUMMARY CSV
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        all_results
    )


    summary_df.to_csv(
        REPORT_DIR
        / "m2_summary.csv",
        index=False
    )


    summary_df.to_csv(
        REPORT_DIR
        / "m2_results.csv",
        index=False
    )


    # ========================================================
    # SUCCESS
    # ========================================================

    print()
    print("=" * 70)
    print("M2 COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"Result JSON : "
        f"{REPORT_DIR / 'm2_results.json'}"
    )

    print(
        f"Charts      : "
        f"{REPORT_DIR}"
    )

    print(
        f"Models      : "
        f"{MODEL_DIR}"
    )

    print()
    print(
        "M2 pipeline:"
    )

    print(
        "Load -> Prepare -> "
        "Regression -> Classification -> "
        "Evaluation"
    )

    print()
    print(
        "M2 is ready for M3."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        run_m2()

    except Exception as error:

        print()
        print("=" * 70)
        print("M2 TRAINING FAILED")
        print("=" * 70)

        print(
            f"Error: {error}"
        )

        print("=" * 70)

        import traceback

        traceback.print_exc()

        raise