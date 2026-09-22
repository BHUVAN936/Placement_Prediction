# ============================================================
# M1 - PLACEMENTPREDICT AS A SYSTEM
# ============================================================
#
# Pipeline:
#   1. Load Dataset
#   2. Clean Dataset
#   3. Feature Engineering
#   4. EDA
#   5. Preprocessing
#
# Outputs:
#   reports/M1/
#   models/M1/
#
# Windows-safe:
#   No Unicode arrows/checkmarks are used in console output.
# ============================================================


import os
import sys
import json
import warnings
from pathlib import Path

# ============================================================
# WINDOWS UTF-8 SUPPORT
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


# ============================================================
# THIRD-PARTY IMPORTS
# ============================================================

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)


warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(
    __file__
).resolve().parents[2]

DATA_DIR = ROOT / "data"

REPORT_DIR = (
    ROOT
    / "reports"
    / "M1"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "M1"
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

TARGET_COLUMN = "PlacementStatus"

REGRESSION_TARGET = "Salary Package"


# ============================================================
# ORIGINAL MODEL FEATURES
# ============================================================

MODEL_FEATURES = [

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
    "CGPA_Tier"
]


CATEGORICAL_FEATURES = [

    "Gender",
    "City",
    "CollegeTier",
    "Stream",
    "Specialisation",
    "Hostel",
    "HistoryOfBacklogs",
    "CGPA_Tier"
]


NUMERICAL_FEATURES = [

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

    "ExtraCurricular"
]


ENGINEERED_FEATURES = [

    "AverageSGPA",
    "AcademicConsistency",
    "AcademicScore",
    "SkillScore",
    "InterviewScore",
    "ExperienceScore"
]


# ============================================================
# UTILITY
# ============================================================

def print_header(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# FIND DATASET
# ============================================================

def get_dataset_path():

    exact_path = (
        DATA_DIR
        / DATASET_NAME
    )

    if exact_path.exists():

        return exact_path


    csv_files = list(
        DATA_DIR.glob("*.csv")
    )

    if not csv_files:

        raise FileNotFoundError(
            "No CSV dataset found in data folder."
        )


    return csv_files[0]


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    dataset_path = get_dataset_path()

    print(
        f"Dataset: {dataset_path}"
    )

    df = pd.read_csv(
        dataset_path
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Columns loaded: {len(df.columns)}"
    )

    return df


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

def normalize_columns(df):

    df = df.copy()

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


# ============================================================
# CLEAN TARGET
# ============================================================

def clean_target(df):

    df = df.copy()

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "not found."
        )


    # Convert common text labels
    if not pd.api.types.is_numeric_dtype(
        df[TARGET_COLUMN]
    ):

        mapping = {

            "placed": 1,
            "not placed": 0,

            "yes": 1,
            "no": 0,

            "true": 1,
            "false": 0,

            "1": 1,
            "0": 0
        }

        values = (
            df[TARGET_COLUMN]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        mapped = values.map(mapping)

        numeric = pd.to_numeric(
            df[TARGET_COLUMN],
            errors="coerce"
        )

        df[TARGET_COLUMN] = (
            mapped
            .fillna(numeric)
        )

    else:

        df[TARGET_COLUMN] = pd.to_numeric(
            df[TARGET_COLUMN],
            errors="coerce"
        )


    # Remove rows without target
    before = len(df)

    df = df.dropna(
        subset=[TARGET_COLUMN]
    )

    removed = before - len(df)

    if removed > 0:

        print(
            f"Rows removed because of missing "
            f"target: {removed}"
        )


    # Force binary target when applicable
    unique_values = sorted(
        df[TARGET_COLUMN]
        .dropna()
        .unique()
        .tolist()
    )

    if set(unique_values).issubset(
        {0, 1}
    ):

        df[TARGET_COLUMN] = (
            df[TARGET_COLUMN]
            .astype(int)
        )

    else:

        # Convert values to 0/1 if there
        # are exactly two classes
        if len(unique_values) == 2:

            mapping = {
                unique_values[0]: 0,
                unique_values[1]: 1
            }

            df[TARGET_COLUMN] = (
                df[TARGET_COLUMN]
                .map(mapping)
                .astype(int)
            )


    return df


# ============================================================
# REMOVE EMPTY COLUMNS
# ============================================================

def remove_empty_columns(df):

    df = df.copy()

    empty_columns = [
        column
        for column in df.columns
        if df[column].isna().all()
    ]

    if empty_columns:

        df = df.drop(
            columns=empty_columns
        )

        print(
            f"Empty columns removed: "
            f"{len(empty_columns)}"
        )


    return df


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(df):

    df = df.copy()

    before = len(df)

    df = df.drop_duplicates()

    removed = (
        before - len(df)
    )

    print(
        f"Duplicate rows removed: "
        f"{removed}"
    )

    return df


# ============================================================
# REPLACE INFINITE VALUES
# ============================================================

def replace_infinite_values(df):

    df = df.copy()

    numeric_columns = (
        df.select_dtypes(
            include=[np.number]
        ).columns
    )

    df[numeric_columns] = (
        df[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    return df


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

def fill_missing_values(df):

    df = df.copy()

    numeric_columns = (
        df.select_dtypes(
            include=[np.number]
        ).columns
    )

    categorical_columns = (
        df.select_dtypes(
            exclude=[np.number]
        ).columns
    )


    # Numeric -> median
    for column in numeric_columns:

        if column == TARGET_COLUMN:

            continue

        if df[column].isna().any():

            median_value = (
                df[column].median()
            )

            df[column] = (
                df[column]
                .fillna(median_value)
            )


    # Categorical -> mode
    for column in categorical_columns:

        if df[column].isna().any():

            mode_values = (
                df[column]
                .mode()
            )

            if not mode_values.empty:

                df[column] = (
                    df[column]
                    .fillna(mode_values.iloc[0])
                )

            else:

                df[column] = (
                    df[column]
                    .fillna("Unknown")
                )


    return df


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(df):

    print_header(
        "[2/5] Cleaning dataset..."
    )

    df = normalize_columns(
        df
    )

    df = remove_empty_columns(
        df
    )

    df = replace_infinite_values(
        df
    )

    df = clean_target(
        df
    )

    df = remove_duplicates(
        df
    )

    df = fill_missing_values(
        df
    )


    print(
        f"Rows after cleaning: "
        f"{len(df):,}"
    )

    print(
        f"Columns after cleaning: "
        f"{len(df.columns)}"
    )

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def safe_numeric(
    df,
    column,
    default=0
):

    if column in df.columns:

        return pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(default)

    return pd.Series(
        default,
        index=df.index,
        dtype=float
    )


def feature_engineering(df):

    print_header(
        "[3/5] Engineering features..."
    )

    df = df.copy()


    # --------------------------------------------------------
    # SGPA FEATURES
    # --------------------------------------------------------

    sgpa_columns = [

        column

        for column in [
            "SGPA_Sem1",
            "SGPA_Sem2",
            "SGPA_Sem3",
            "SGPA_Sem4",
            "SGPA_Sem5",
            "SGPA_Sem6",
            "SGPA_Sem7",
            "SGPA_Sem8"
        ]

        if column in df.columns
    ]


    if sgpa_columns:

        sgpa_data = df[
            sgpa_columns
        ].apply(
            pd.to_numeric,
            errors="coerce"
        )

        df[
            "AverageSGPA"
        ] = sgpa_data.mean(
            axis=1
        )

        df[
            "AcademicConsistency"
        ] = sgpa_data.std(
            axis=1
        ).fillna(0)

    else:

        df[
            "AverageSGPA"
        ] = 0.0

        df[
            "AcademicConsistency"
        ] = 0.0


    # --------------------------------------------------------
    # ACADEMIC SCORE
    # --------------------------------------------------------

    cgpa = safe_numeric(
        df,
        "CGPA"
    )

    attendance = safe_numeric(
        df,
        "AttendancePercent"
    )

    df[
        "AcademicScore"
    ] = (
        0.70 * cgpa
        +
        0.30 * (
            attendance / 10.0
        )
    )


    # --------------------------------------------------------
    # SKILL SCORE
    # --------------------------------------------------------

    aptitude = safe_numeric(
        df,
        "AptitudeTestScore"
    )

    coding = safe_numeric(
        df,
        "CodingTestScore"
    )

    soft_skills = safe_numeric(
        df,
        "SoftSkillsRating"
    )

    df[
        "SkillScore"
    ] = (
        0.40 * aptitude
        +
        0.40 * coding
        +
        0.20 * (
            soft_skills * 10.0
        )
    )


    # --------------------------------------------------------
    # INTERVIEW SCORE
    # --------------------------------------------------------

    mock_interview = safe_numeric(
        df,
        "MockInterviewScore"
    )

    df[
        "InterviewScore"
    ] = (
        0.60 * mock_interview
        +
        0.40 * (
            soft_skills * 10.0
        )
    )


    # --------------------------------------------------------
    # EXPERIENCE SCORE
    # --------------------------------------------------------

    internships = safe_numeric(
        df,
        "Internships"
    )

    projects = safe_numeric(
        df,
        "Projects"
    )

    workshops = safe_numeric(
        df,
        "Workshops"
    )

    certifications = safe_numeric(
        df,
        "Certifications"
    )

    publications = safe_numeric(
        df,
        "Publications"
    )

    df[
        "ExperienceScore"
    ] = (
        0.30 * internships
        +
        0.30 * projects
        +
        0.15 * workshops
        +
        0.15 * certifications
        +
        0.10 * publications
    )


    print("New features:")

    print("  + AverageSGPA")
    print("  + AcademicConsistency")
    print("  + AcademicScore")
    print("  + SkillScore")
    print("  + InterviewScore")
    print("  + ExperienceScore")


    return df


# ============================================================
# SAVE ENGINEERED DATASET
# ============================================================

def save_engineered_dataset(df):

    output_file = (
        REPORT_DIR
        / "m1_engineered_dataset.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    return output_file


# ============================================================
# EDA
# ============================================================

def run_eda(df):

    print_header(
        "[4/5] Running EDA..."
    )

    chart_count = 0


    # --------------------------------------------------------
    # 1. PLACEMENT BAR
    # --------------------------------------------------------

    if TARGET_COLUMN in df.columns:

        counts = (
            df[TARGET_COLUMN]
            .value_counts()
            .sort_index()
        )

        plt.figure(
            figsize=(8, 5)
        )

        plt.bar(
            [
                str(x)
                for x in counts.index
            ],
            counts.values
        )

        plt.xlabel(
            "Placement Status"
        )

        plt.ylabel(
            "Number of Students"
        )

        plt.title(
            "Placement Status Distribution"
        )

        plt.tight_layout()

        plt.savefig(
            REPORT_DIR
            / "01_placement_distribution.png",
            dpi=150
        )

        plt.close()

        chart_count += 1


    # --------------------------------------------------------
    # 2. PLACEMENT PIE
    # --------------------------------------------------------

    if TARGET_COLUMN in df.columns:

        counts = (
            df[TARGET_COLUMN]
            .value_counts()
        )

        plt.figure(
            figsize=(7, 7)
        )

        plt.pie(
            counts.values,
            labels=[
                str(x)
                for x in counts.index
            ],
            autopct="%1.1f%%"
        )

        plt.title(
            "Placement Status Percentage"
        )

        plt.tight_layout()

        plt.savefig(
            REPORT_DIR
            / "02_placement_pie.png",
            dpi=150
        )

        plt.close()

        chart_count += 1


    # --------------------------------------------------------
    # 3. CGPA DISTRIBUTION
    # --------------------------------------------------------

    if "CGPA" in df.columns:

        plt.figure(
            figsize=(8, 5)
        )

        plt.hist(
            df["CGPA"].dropna(),
            bins=30
        )

        plt.xlabel(
            "CGPA"
        )

        plt.ylabel(
            "Students"
        )

        plt.title(
            "CGPA Distribution"
        )

        plt.tight_layout()

        plt.savefig(
            REPORT_DIR
            / "03_cgpa_distribution.png",
            dpi=150
        )

        plt.close()

        chart_count += 1


    # --------------------------------------------------------
    # 4. ATTENDANCE DISTRIBUTION
    # --------------------------------------------------------

    if "AttendancePercent" in df.columns:

        plt.figure(
            figsize=(8, 5)
        )

        plt.hist(
            df[
                "AttendancePercent"
            ].dropna(),
            bins=30
        )

        plt.xlabel(
            "Attendance Percentage"
        )

        plt.ylabel(
            "Students"
        )

        plt.title(
            "Attendance Distribution"
        )

        plt.tight_layout()

        plt.savefig(
            REPORT_DIR
            / "04_attendance_distribution.png",
            dpi=150
        )

        plt.close()

        chart_count += 1


    # --------------------------------------------------------
    # 5. INTERNSHIPS
    # --------------------------------------------------------

    if "Internships" in df.columns:

        counts = (
            df["Internships"]
            .value_counts()
            .sort_index()
        )

        plt.figure(
            figsize=(8, 5)
        )

        plt.bar(
            [
                str(x)
                for x in counts.index
            ],
            counts.values
        )

        plt.xlabel(
            "Number of Internships"
        )

        plt.ylabel(
            "Students"
        )

        plt.title(
            "Internship Distribution"
        )

        plt.tight_layout()

        plt.savefig(
            REPORT_DIR
            / "05_internship_distribution.png",
            dpi=150
        )

        plt.close()

        chart_count += 1


    # --------------------------------------------------------
    # 6. SGPA
    # --------------------------------------------------------

    sgpa_columns = [

        column

        for column in [
            "SGPA_Sem1",
            "SGPA_Sem2",
            "SGPA_Sem3",
            "SGPA_Sem4",
            "SGPA_Sem5",
            "SGPA_Sem6",
            "SGPA_Sem7",
            "SGPA_Sem8"
        ]

        if column in df.columns
    ]


    if sgpa_columns:

        means = (
            df[sgpa_columns]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .mean()
        )

        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            means.index,
            means.values,
            marker="o"
        )

        plt.xlabel(
            "Semester"
        )

        plt.ylabel(
            "Average SGPA"
        )

        plt.title(
            "Average SGPA by Semester"
        )

        plt.xticks(
            rotation=45
        )

        plt.tight_layout()

        plt.savefig(
            REPORT_DIR
            / "06_sgpa_by_semester.png",
            dpi=150
        )

        plt.close()

        chart_count += 1


    # --------------------------------------------------------
    # 7. ACADEMIC SCORE
    # --------------------------------------------------------

    if (
        "AcademicScore" in df.columns
        and TARGET_COLUMN in df.columns
    ):

        plt.figure(
            figsize=(8, 5)
        )

        for value in sorted(
            df[TARGET_COLUMN]
            .dropna()
            .unique()
        ):

            subset = df[
                df[TARGET_COLUMN] == value
            ]

            plt.hist(
                subset["AcademicScore"],
                bins=30,
                alpha=0.5,
                label=f"Status {value}"
            )

        plt.xlabel(
            "Academic Score"
        )

        plt.ylabel(
            "Students"
        )

        plt.title(
            "Academic Score by Placement Status"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            REPORT_DIR
            / "07_academic_score.png",
            dpi=150
        )

        plt.close()

        chart_count += 1


    # --------------------------------------------------------
    # 8. SKILL SCORE
    # --------------------------------------------------------

    if (
        "SkillScore" in df.columns
        and TARGET_COLUMN in df.columns
    ):

        plt.figure(
            figsize=(8, 5)
        )

        for value in sorted(
            df[TARGET_COLUMN]
            .dropna()
            .unique()
        ):

            subset = df[
                df[TARGET_COLUMN] == value
            ]

            plt.hist(
                subset["SkillScore"],
                bins=30,
                alpha=0.5,
                label=f"Status {value}"
            )

        plt.xlabel(
            "Skill Score"
        )

        plt.ylabel(
            "Students"
        )

        plt.title(
            "Skill Score by Placement Status"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            REPORT_DIR
            / "08_skill_score.png",
            dpi=150
        )

        plt.close()

        chart_count += 1


    # --------------------------------------------------------
    # 9. CORRELATION
    # --------------------------------------------------------

    numeric_df = (
        df.select_dtypes(
            include=[np.number]
        )
    )

    if numeric_df.shape[1] >= 2:

        correlation = (
            numeric_df
            .corr()
        )

        correlation.to_csv(
            REPORT_DIR
            / "m1_correlation_matrix.csv"
        )

        plt.figure(
            figsize=(12, 10)
        )

        plt.imshow(
            correlation,
            aspect="auto"
        )

        plt.colorbar()

        plt.title(
            "Numerical Feature Correlation"
        )

        plt.xticks(
            range(
                len(correlation.columns)
            ),
            correlation.columns,
            rotation=90,
            fontsize=7
        )

        plt.yticks(
            range(
                len(correlation.columns)
            ),
            correlation.columns,
            fontsize=7
        )

        plt.tight_layout()

        plt.savefig(
            REPORT_DIR
            / "09_correlation_matrix.png",
            dpi=150
        )

        plt.close()

        chart_count += 1


    # --------------------------------------------------------
    # 10. CGPA VS PLACEMENT
    # --------------------------------------------------------

    if (
        "CGPA" in df.columns
        and TARGET_COLUMN in df.columns
    ):

        grouped = (
            df.groupby(
                TARGET_COLUMN
            )["CGPA"]
            .mean()
        )

        plt.figure(
            figsize=(8, 5)
        )

        plt.bar(
            [
                str(x)
                for x in grouped.index
            ],
            grouped.values
        )

        plt.xlabel(
            "Placement Status"
        )

        plt.ylabel(
            "Average CGPA"
        )

        plt.title(
            "Average CGPA by Placement Status"
        )

        plt.tight_layout()

        plt.savefig(
            REPORT_DIR
            / "10_cgpa_vs_placement.png",
            dpi=150
        )

        plt.close()

        chart_count += 1


    print(
        f"Charts generated: {chart_count}"
    )

    return chart_count


# ============================================================
# DATASET REPORTS
# ============================================================

def save_dataset_reports(df):

    # --------------------------------------------------------
    # NUMERICAL SUMMARY
    # --------------------------------------------------------

    numeric_df = (
        df.select_dtypes(
            include=[np.number]
        )
    )

    if not numeric_df.empty:

        numeric_summary = (
            numeric_df
            .describe()
            .T
            .reset_index()
        )

        numeric_summary = (
            numeric_summary
            .rename(
                columns={
                    "index": "Feature"
                }
            )
        )

        numeric_summary.to_csv(
            REPORT_DIR
            / "m1_numerical_summary.csv",
            index=False
        )


    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    missing = pd.DataFrame({
        "Column": df.columns,
        "MissingCount": [
            int(df[column].isna().sum())
            for column in df.columns
        ]
    })

    missing[
        "MissingPercentage"
    ] = (
        missing["MissingCount"]
        / len(df)
        * 100
    )

    missing.to_csv(
        REPORT_DIR
        / "m1_missing_values.csv",
        index=False
    )


    # --------------------------------------------------------
    # CATEGORICAL SUMMARY
    # --------------------------------------------------------

    categorical_columns = (
        df.select_dtypes(
            exclude=[np.number]
        ).columns
    )

    categorical_rows = []

    for column in categorical_columns:

        categorical_rows.append({

            "Column": column,

            "UniqueValues": int(
                df[column]
                .nunique(
                    dropna=True
                )
            ),

            "MissingValues": int(
                df[column]
                .isna()
                .sum()
            )
        })


    pd.DataFrame(
        categorical_rows
    ).to_csv(
        REPORT_DIR
        / "m1_categorical_summary.csv",
        index=False
    )


    # --------------------------------------------------------
    # DATASET INFO
    # --------------------------------------------------------

    dataset_info = {

        "rows": int(
            df.shape[0]
        ),

        "columns": int(
            df.shape[1]
        ),

        "numeric_columns": int(
            len(numeric_df.columns)
        ),

        "categorical_columns": int(
            len(categorical_columns)
        ),

        "missing_values": int(
            df.isna()
            .sum()
            .sum()
        ),

        "duplicate_rows": int(
            df.duplicated()
            .sum()
        )
    }


    with open(
        REPORT_DIR
        / "m1_dataset_info.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            dataset_info,
            file,
            indent=4
        )


# ============================================================
# PREPROCESSOR
# ============================================================

def create_one_hot_encoder():

    try:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

    except TypeError:

        # Compatibility with older sklearn
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


def build_preprocessor(
    available_features
):

    categorical = [
        column

        for column
        in CATEGORICAL_FEATURES

        if column
        in available_features
    ]


    numerical = [
        column

        for column
        in NUMERICAL_FEATURES

        if column
        in available_features
    ]


    numeric_pipeline = [

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
    ]


    categorical_pipeline = [

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",
            create_one_hot_encoder()
        )
    ]


    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numeric",
                __import__(
                    "sklearn.pipeline",
                    fromlist=[
                        "Pipeline"
                    ]
                ).Pipeline(
                    numeric_pipeline
                ),
                numerical
            ),

            (
                "categorical",
                __import__(
                    "sklearn.pipeline",
                    fromlist=[
                        "Pipeline"
                    ]
                ).Pipeline(
                    categorical_pipeline
                ),
                categorical
            )
        ],

        remainder="drop"
    )


    return (
        preprocessor,
        numerical,
        categorical
    )


# ============================================================
# GET FEATURE NAMES
# ============================================================

def get_transformed_feature_names(
    preprocessor,
    numerical,
    categorical
):

    feature_names = []


    # Numerical names
    feature_names.extend(
        numerical
    )


    # Categorical names
    if categorical:

        try:

            categorical_pipeline = (
                preprocessor
                .named_transformers_[
                    "categorical"
                ]
            )

            encoder = (
                categorical_pipeline
                .named_steps[
                    "encoder"
                ]
            )

            names = (
                encoder
                .get_feature_names_out(
                    categorical
                )
            )

            feature_names.extend(
                names.tolist()
            )

        except Exception:

            for column in categorical:

                feature_names.append(
                    column
                )


    return feature_names


# ============================================================
# RUN PREPROCESSING
# ============================================================

def run_preprocessing(df):

    print_header(
        "[5/5] Running preprocessing..."
    )


    # --------------------------------------------------------
    # Select model features
    # --------------------------------------------------------

    available_features = [

        column

        for column in MODEL_FEATURES

        if column in df.columns
    ]


    missing_features = [

        column

        for column in MODEL_FEATURES

        if column not in df.columns
    ]


    if missing_features:

        print(
            "Missing expected features:"
        )

        for column in missing_features:

            print(
                f"  - {column}"
            )


    print(
        f"Original model features: "
        f"{len(available_features)}"
    )


    X = df[
        available_features
    ].copy()


    # --------------------------------------------------------
    # Build preprocessor
    # --------------------------------------------------------

    (
        preprocessor,
        numerical,
        categorical
    ) = build_preprocessor(
        available_features
    )


    # --------------------------------------------------------
    # Fit transform
    # --------------------------------------------------------

    X_processed = (
        preprocessor
        .fit_transform(X)
    )


    X_processed = np.asarray(
        X_processed,
        dtype=np.float64
    )


    print(
        f"Transformed features: "
        f"{X_processed.shape[1]}"
    )


    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    transformed_names = (
        get_transformed_feature_names(
            preprocessor,
            numerical,
            categorical
        )
    )


    # Fallback
    if len(
        transformed_names
    ) != X_processed.shape[1]:

        transformed_names = [

            f"Feature_{i + 1}"

            for i in range(
                X_processed.shape[1]
            )
        ]


    # --------------------------------------------------------
    # Save preprocessor
    # --------------------------------------------------------

    import joblib


    preprocessor_file = (
        MODEL_DIR
        / "m1_preprocessor.joblib"
    )


    joblib.dump(
        preprocessor,
        preprocessor_file
    )


    # --------------------------------------------------------
    # Save feature names
    # --------------------------------------------------------

    feature_names_df = pd.DataFrame({

        "FeatureIndex": range(
            len(transformed_names)
        ),

        "FeatureName":
            transformed_names
    })


    feature_names_df.to_csv(
        REPORT_DIR
        / "m1_feature_names.csv",
        index=False
    )


    # --------------------------------------------------------
    # Save processed matrix
    # --------------------------------------------------------

    processed_df = pd.DataFrame(
        X_processed,
        columns=transformed_names
    )


    processed_df.to_csv(
        REPORT_DIR
        / "m1_processed_features.csv",
        index=False
    )


    # --------------------------------------------------------
    # Save model input feature names
    # --------------------------------------------------------

    model_input_df = pd.DataFrame({

        "OriginalFeature":
            available_features

    })


    model_input_df.to_csv(
        REPORT_DIR
        / "m1_model_features.csv",
        index=False
    )


    return {

        "preprocessor":
            preprocessor,

        "X_processed":
            X_processed,

        "feature_names":
            transformed_names,

        "original_features":
            available_features,

        "numerical_features":
            numerical,

        "categorical_features":
            categorical,

        "preprocessor_file":
            preprocessor_file
    }


# ============================================================
# SAVE FINAL M1 RESULTS
# ============================================================

def save_results(
    df,
    preprocessing_result,
    chart_count
):

    target_distribution = {}

    if TARGET_COLUMN in df.columns:

        counts = (
            df[TARGET_COLUMN]
            .value_counts()
            .sort_index()
        )

        target_distribution = {

            str(key):
                int(value)

            for key, value
            in counts.items()
        }


    results = {

        "module": "M1",

        "module_name":
            "PlacementPredict as a System",

        "status":
            "completed",

        "dataset": {

            "rows":
                int(df.shape[0]),

            "columns":
                int(df.shape[1])
        },

        "target": {

            "classification":
                TARGET_COLUMN,

            "regression":
                REGRESSION_TARGET,

            "distribution":
                target_distribution
        },

        "features": {

            "original_model_features":
                len(
                    preprocessing_result[
                        "original_features"
                    ]
                ),

            "transformed_features":
                len(
                    preprocessing_result[
                        "feature_names"
                    ]
                ),

            "engineered_features":
                ENGINEERED_FEATURES
        },

        "preprocessing": {

            "numerical_features":
                len(
                    preprocessing_result[
                        "numerical_features"
                    ]
                ),

            "categorical_features":
                len(
                    preprocessing_result[
                        "categorical_features"
                    ]
                )
        },

        "eda": {

            "charts_generated":
                int(chart_count)
        },

        "files": {

            "preprocessor":
                str(
                    preprocessing_result[
                        "preprocessor_file"
                    ]
                ),

            "report_directory":
                str(REPORT_DIR),

            "model_directory":
                str(MODEL_DIR)
        }
    }


    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    with open(
        REPORT_DIR
        / "m1_results.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )


    # --------------------------------------------------------
    # SUMMARY CSV
    # --------------------------------------------------------

    summary = pd.DataFrame([

        {

            "Module":
                "M1",

            "Rows":
                int(df.shape[0]),

            "Columns":
                int(df.shape[1]),

            "Original Features":
                len(
                    preprocessing_result[
                        "original_features"
                    ]
                ),

            "Transformed Features":
                len(
                    preprocessing_result[
                        "feature_names"
                    ]
                ),

            "Engineered Features":
                len(
                    ENGINEERED_FEATURES
                ),

            "Charts":
                int(chart_count),

            "Status":
                "Completed"
        }
    ])


    summary.to_csv(
        REPORT_DIR
        / "m1_summary.csv",
        index=False
    )


    return results


# ============================================================
# FINAL PIPELINE
# ============================================================

def run_m1():

    print()
    print("=" * 70)
    print("M1 - PLACEMENTPREDICT AS A SYSTEM")
    print("=" * 70)

    print(
        f"Project root: {ROOT}"
    )

    print(
        f"Reports: {REPORT_DIR}"
    )

    print(
        f"Models: {MODEL_DIR}"
    )


    try:

        # ====================================================
        # 1. LOAD
        # ====================================================

        print()
        print("[1/5] Loading dataset...")

        df = load_dataset()


        # ====================================================
        # 2. CLEAN
        # ====================================================

        df = clean_data(
            df
        )


        # ====================================================
        # 3. FEATURE ENGINEERING
        # ====================================================

        df = feature_engineering(
            df
        )


        # Save engineered dataset
        engineered_file = (
            save_engineered_dataset(
                df
            )
        )

        print(
            f"Engineered dataset saved: "
            f"{engineered_file}"
        )


        # ====================================================
        # DATASET REPORTS
        # ====================================================

        save_dataset_reports(
            df
        )


        # ====================================================
        # 4. EDA
        # ====================================================

        chart_count = run_eda(
            df
        )


        # ====================================================
        # 5. PREPROCESSING
        # ====================================================

        preprocessing_result = (
            run_preprocessing(
                df
            )
        )


        # ====================================================
        # SAVE FINAL RESULTS
        # ====================================================

        results = save_results(

            df,

            preprocessing_result,

            chart_count
        )


        # ====================================================
        # SUCCESS
        # ====================================================

        print()
        print("=" * 70)
        print("M1 COMPLETED SUCCESSFULLY")
        print("=" * 70)

        print(
            f"Result JSON : "
            f"{REPORT_DIR / 'm1_results.json'}"
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
        print("M1 pipeline:")
        print(
            "Load -> Clean -> "
            "Feature Engineering -> "
            "EDA -> Preprocessing"
        )

        print()
        print(
            "M1 is ready for M2."
        )

        return results


    except Exception as error:

        # ====================================================
        # FAILURE
        # ====================================================

        print()
        print("=" * 70)
        print("M1 TRAINING FAILED")
        print("=" * 70)

        print(
            f"Error: {error}"
        )

        print("=" * 70)

        import traceback

        traceback.print_exc()

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_m1()