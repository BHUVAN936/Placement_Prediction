# ============================================================
# M3 - TREE BASED MODELS
# ============================================================
#
# Algorithms:
#   1. Decision Tree
#   2. Random Forest
#   3. AdaBoost
#   4. Gradient Boosting
#   5. XGBoost
#   6. LightGBM
#
# Windows-safe version
# ============================================================

import os
import sys
import json
import warnings
from pathlib import Path

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

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

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
    ROOT / "reports" / "M3"
)

MODEL_DIR = (
    ROOT / "models" / "M3"
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
# TARGET
# ============================================================

TARGET = "PlacementStatus"

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
        / "placement_predict_50k Dataset (2).csv"
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
            "No dataset found."
        )


    return pd.read_csv(
        csv_files[0]
    )


# ============================================================
# ENCODER
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

    categorical = (
        X.select_dtypes(
            include=[
                "object",
                "category",
                "bool"
            ]
        ).columns.tolist()
    )


    numerical = (
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


    return ColumnTransformer([

        (
            "numeric",
            numeric_pipeline,
            numerical
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical
        )
    ])


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    probabilities=None
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


    if probabilities is not None:

        try:

            result["ROC_AUC"] = float(
                roc_auc_score(
                    y_true,
                    probabilities
                )
            )

        except Exception:

            result["ROC_AUC"] = 0.0

    else:

        result["ROC_AUC"] = 0.0


    return result


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    y_true,
    y_pred,
    name
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
        f"{name} - Confusion Matrix"
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
        name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        +
        "_confusion_matrix.png"
    )


    plt.savefig(
        REPORT_DIR / filename,
        dpi=150
    )


    plt.close()


# ============================================================
# MODEL COMPARISON
# ============================================================

def save_comparison_chart(
    results
):

    names = [
        row["Model"]
        for row in results
    ]


    accuracy = [
        row["Accuracy"]
        for row in results
    ]


    f1_values = [
        row["F1"]
        for row in results
    ]


    x = np.arange(
        len(names)
    )


    width = 0.35


    fig, ax = plt.subplots(
        figsize=(12, 6)
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
        rotation=25,
        ha="right"
    )


    ax.set_ylim(
        0,
        1.05
    )


    ax.set_title(
        "M3 Tree-Based Model Comparison"
    )


    ax.legend()


    plt.tight_layout()


    plt.savefig(
        REPORT_DIR
        / "m3_model_comparison.png",
        dpi=150
    )


    plt.close()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(
    pipeline,
    model_name
):

    try:

        model = (
            pipeline
            .named_steps[
                "model"
            ]
        )


        if not hasattr(
            model,
            "feature_importances_"
        ):

            return


        preprocessor = (
            pipeline
            .named_steps[
                "preprocessor"
            ]
        )


        names = (
            preprocessor
            .get_feature_names_out()
        )


        importances = (
            model
            .feature_importances_
        )


        if len(names) != len(
            importances
        ):

            names = [
                f"Feature_{i + 1}"
                for i in range(
                    len(importances)
                )
            ]


        importance_df = pd.DataFrame({

            "Feature":
                names,

            "Importance":
                importances

        })


        importance_df = (
            importance_df
            .sort_values(
                "Importance",
                ascending=False
            )
        )


        safe_name = (
            model_name
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )


        importance_df.to_csv(

            REPORT_DIR
            / f"{safe_name}_feature_importance.csv",

            index=False
        )


        top = (
            importance_df
            .head(20)
            .sort_values(
                "Importance"
            )
        )


        plt.figure(
            figsize=(10, 8)
        )


        plt.barh(
            top["Feature"],
            top["Importance"]
        )


        plt.xlabel(
            "Importance"
        )


        plt.title(
            f"{model_name} - Top Features"
        )


        plt.tight_layout()


        plt.savefig(

            REPORT_DIR
            / f"{safe_name}_feature_importance.png",

            dpi=150
        )


        plt.close()


    except Exception as error:

        print(
            f"Feature importance warning "
            f"for {model_name}: {error}"
        )


# ============================================================
# MAIN
# ============================================================

def run_m3():

    print()
    print("=" * 70)
    print("M3 - TREE BASED MODELS")
    print("=" * 70)


    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print()
    print("[1/5] Loading dataset...")


    df = load_dataset()


    print(
        f"Rows: {len(df):,}"
    )


    print(
        f"Columns: {len(df.columns)}"
    )


    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    print()
    print("[2/5] Preparing classification data...")


    if TARGET not in df.columns:

        raise ValueError(
            f"{TARGET} not found."
        )


    y = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )


    valid = y.notna()


    df = df.loc[
        valid
    ].copy()


    y = y.loc[
        valid
    ].astype(int)


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    X = df.drop(

        columns=[
            TARGET,
            REGRESSION_TARGET
        ],

        errors="ignore"
    )


    # Remove empty columns
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


    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )


    # --------------------------------------------------------
    # PREPROCESSOR
    # --------------------------------------------------------

    print()
    print("[3/5] Preparing preprocessing...")


    preprocessor = create_preprocessor(
        X_train
    )


    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    print()
    print("[4/5] Training tree-based models...")


    models = {

        "Decision Tree":
            DecisionTreeClassifier(
                criterion="gini",
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ),


        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),


        "AdaBoost":
            AdaBoostClassifier(
                n_estimators=150,
                learning_rate=0.8,
                random_state=42
            ),


        "Gradient Boosting":
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ),


        "XGBoost":
            XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            ),


        "LightGBM":
            LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary",
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )
    }


    results = []


    for name, model in models.items():

        print()
        print(
            f"Training {name}..."
        )


        pipeline = Pipeline([

            (
                "preprocessor",
                create_preprocessor(
                    X_train
                )
            ),

            (
                "model",
                model
            )
        ])


        pipeline.fit(
            X_train,
            y_train
        )


        predictions = (
            pipeline.predict(
                X_test
            )
        )


        try:

            probabilities = (
                pipeline
                .predict_proba(
                    X_test
                )[:, 1]
            )

        except Exception:

            probabilities = None


        metrics = calculate_metrics(

            y_test,

            predictions,

            probabilities
        )


        row = {

            "Model":
                name,

            "Task":
                "Classification",

            **metrics
        }


        results.append(
            row
        )


        safe_name = (
            name
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )


        joblib.dump(

            pipeline,

            MODEL_DIR
            / f"{safe_name}.joblib"
        )


        save_confusion_matrix(

            y_test,

            predictions,

            name
        )


        save_feature_importance(

            pipeline,

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

        print(
            f"  ROC-AUC  : "
            f"{metrics['ROC_AUC']:.4f}"
        )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("[5/5] Saving M3 results...")


    results_df = pd.DataFrame(
        results
    )


    results_df.to_csv(

        REPORT_DIR
        / "m3_summary.csv",

        index=False
    )


    results_df.to_csv(

        REPORT_DIR
        / "m3_results.csv",

        index=False
    )


    save_comparison_chart(
        results
    )


    results_json = {

        "module":
            "M3",

        "module_name":
            "Tree-Based Models",

        "status":
            "completed",

        "dataset_rows":
            int(len(df)),

        "input_features":
            int(X.shape[1]),

        "models":
            results
    }


    with open(

        REPORT_DIR
        / "m3_results.json",

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            results_json,

            file,

            indent=4
        )


    # ========================================================
    # SUCCESS
    # ========================================================

    print()
    print("=" * 70)
    print("M3 COMPLETED SUCCESSFULLY")
    print("=" * 70)


    print(
        f"Result JSON : "
        f"{REPORT_DIR / 'm3_results.json'}"
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
        "M3 pipeline:"
    )


    print(
        "Load -> Prepare -> "
        "Decision Tree -> Random Forest -> "
        "AdaBoost -> Gradient Boosting -> "
        "XGBoost -> LightGBM -> Evaluation"
    )


    print()
    print(
        "M3 is ready for M4."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        run_m3()

    except Exception as error:

        print()
        print("=" * 70)
        print("M3 TRAINING FAILED")
        print("=" * 70)

        print(
            f"Error: {error}"
        )

        print("=" * 70)

        import traceback

        traceback.print_exc()

        raise