from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    ROOT
    / "models"
    / "M1"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "M1"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# TARGET COLUMNS
# ============================================================

CLASSIFICATION_TARGET = (
    "PlacementStatus"
)

REGRESSION_TARGET = (
    "Salary Package"
)


# ============================================================
# CATEGORICAL COLUMNS
# ============================================================

CATEGORICAL_COLUMNS = [

    "Gender",

    "City",

    "CollegeTier",

    "Stream",

    "Specialisation",

    "Hostel",

    "HistoryOfBacklogs",

    "CGPA_Tier",

]


# ============================================================
# FEATURE LIST
# ============================================================

BASE_FEATURES = [

    "Gender",

    "City",

    "CollegeTier",

    "Stream",

    "Specialisation",

    "Hostel",

    "HistoryOfBacklogs",

    "SGPA_Sem1",

    "SGPA_Sem2",

    "SGPA_Sem3",

    "SGPA_Sem4",

    "SGPA_Sem5",

    "SGPA_Sem6",

    "SGPA_Sem7",

    "SGPA_Sem8",

    "CGPA",

    "AttendancePercent",

    "Internships",

    "Projects",

    "Workshops",

    "Certifications",

    "Publications",

    "AptitudeTestScore",

    "SoftSkillsRating",

    "CodingTestScore",

    "MockInterviewScore",

    "ExtraCurricular",

    "CGPA_Tier",

]


# ============================================================
# GET AVAILABLE FEATURES
# ============================================================

def get_feature_columns(
    df: pd.DataFrame,
    target: str | None = None,
):

    features = [

        column

        for column in BASE_FEATURES

        if column in df.columns

    ]

    if target in features:

        features.remove(
            target
        )

    return features


# ============================================================
# GET CATEGORICAL FEATURES
# ============================================================

def get_categorical_features(
    features,
):

    return [

        column

        for column in features

        if column
        in CATEGORICAL_COLUMNS

    ]


# ============================================================
# GET NUMERICAL FEATURES
# ============================================================

def get_numeric_features(
    features,
):

    return [

        column

        for column in features

        if column
        not in CATEGORICAL_COLUMNS

    ]


# ============================================================
# CREATE ONE-HOT ENCODER
# ============================================================

def create_one_hot_encoder():

    try:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )

    except TypeError:

        # Compatibility with older sklearn.
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


# ============================================================
# CREATE PREPROCESSOR
# ============================================================

def create_preprocessor(
    features,
):

    categorical_features = (
        get_categorical_features(
            features
        )
    )

    numeric_features = (
        get_numeric_features(
            features
        )
    )

    numeric_pipeline = Pipeline(

        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),

            (
                "scaler",
                StandardScaler(),
            ),

        ]

    )

    categorical_pipeline = Pipeline(

        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),

            (
                "encoder",
                create_one_hot_encoder(),
            ),

        ]

    )

    transformers = []

    if numeric_features:

        transformers.append(

            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )

        )

    if categorical_features:

        transformers.append(

            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )

        )

    if not transformers:

        raise ValueError(
            "No valid features were found "
            "for preprocessing."
        )

    preprocessor = ColumnTransformer(

        transformers=transformers,

        remainder="drop",

    )

    return preprocessor


# ============================================================
# FIT PREPROCESSOR
# ============================================================

def fit_preprocessor(
    df: pd.DataFrame,
    target: str | None = None,
):

    features = get_feature_columns(
        df,
        target,
    )

    if not features:

        raise ValueError(
            "No features available "
            "for preprocessing."
        )

    preprocessor = create_preprocessor(
        features
    )

    X = df[
        features
    ].copy()

    X_transformed = (
        preprocessor
        .fit_transform(X)
    )

    return (
        preprocessor,
        X_transformed,
        features,
    )


# ============================================================
# TRANSFORM DATA
# ============================================================

def transform_data(
    df: pd.DataFrame,
    preprocessor,
    features,
):

    missing_features = [

        column

        for column in features

        if column not in df.columns

    ]

    if missing_features:

        raise ValueError(

            "Missing preprocessing "
            "features: "

            + ", ".join(
                missing_features
            )

        )

    X = df[
        features
    ].copy()

    return (
        preprocessor
        .transform(X)
    )


# ============================================================
# GET FEATURE NAMES
# ============================================================

def get_feature_names(
    preprocessor,
):

    try:

        names = (
            preprocessor
            .get_feature_names_out()
        )

        cleaned = []

        for name in names:

            name = str(name)

            name = (
                name
                .replace(
                    "numeric__",
                    "",
                )
                .replace(
                    "categorical__",
                    "",
                )
            )

            cleaned.append(
                name
            )

        return cleaned

    except Exception:

        return []


# ============================================================
# FIT AND SAVE
# ============================================================

def fit_and_save_preprocessor(
    df: pd.DataFrame,
    target: str | None = None,
):

    preprocessor, X, features = (
        fit_preprocessor(
            df,
            target,
        )
    )

    model_path = (
        MODEL_DIR
        / "m1_preprocessor.joblib"
    )

    feature_path = (
        REPORT_DIR
        / "m1_feature_names.csv"
    )

    joblib.dump(
        preprocessor,
        model_path,
    )

    feature_names = (
        get_feature_names(
            preprocessor
        )
    )

    pd.DataFrame({

        "feature_name":
            feature_names,

    }).to_csv(
        feature_path,
        index=False,
    )

    return {

        "preprocessor":
            preprocessor,

        "transformed_data":
            X,

        "features":
            features,

        "feature_names":
            feature_names,

        "model_path":
            str(model_path),

    }


# ============================================================
# SCALE NUMERICAL MATRIX
# ============================================================

def scale_matrix(
    X,
):

    scaler = StandardScaler()

    X_scaled = (
        scaler.fit_transform(X)
    )

    return (
        scaler,
        X_scaled,
    )


# ============================================================
# SCRIPT TEST
# ============================================================

if __name__ == "__main__":

    from data_loader import load_data
    from data_cleaning import clean_data

    raw = load_data()

    cleaned = clean_data(
        raw
    )

    result = (
        fit_and_save_preprocessor(
            cleaned,
            CLASSIFICATION_TARGET,
        )
    )

    print()
    print(
        "Preprocessing completed."
    )

    print(
        "Original features:",
        len(
            result["features"]
        ),
    )

    print(
        "Transformed features:",
        len(
            result["feature_names"]
        ),
    )

    print(
        "Saved:",
        result["model_path"]
    )