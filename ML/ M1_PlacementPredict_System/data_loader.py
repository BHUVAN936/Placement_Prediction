from pathlib import Path

import pandas as pd


# =========================================================
# PROJECT PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"

DATA_FILE = DATA_DIR / "placement_predict_50k Dataset (2).csv"


# =========================================================
# LOAD DATASET
# =========================================================

def load_data():

    """
    Load the PlacementPredict dataset.

    Returns
    -------
    pandas.DataFrame
        Complete placement prediction dataset.
    """

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    return df


# =========================================================
# DATASET INFORMATION
# =========================================================

def get_dataset_info(df):

    """
    Return basic information about the dataset.
    """

    info = {

        "rows": int(df.shape[0]),

        "columns": int(df.shape[1]),

        "column_names": list(df.columns),

        "missing_values": int(
            df.isnull().sum().sum()
        ),

        "duplicate_rows": int(
            df.duplicated().sum()
        ),

        "numeric_columns": list(
            df.select_dtypes(
                include="number"
            ).columns
        ),

        "categorical_columns": list(
            df.select_dtypes(
                exclude="number"
            ).columns
        )

    }

    return info


# =========================================================
# FEATURE COLUMNS
# =========================================================

CATEGORICAL_COLUMNS = [

    "Gender",

    "City",

    "CollegeTier",

    "Stream",

    "Specialisation",

    "Hostel",

    "HistoryOfBacklogs",

    "CGPA_Tier"

]


NUMERICAL_COLUMNS = [

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


CLASSIFICATION_TARGET = "PlacementStatus"


REGRESSION_TARGET = "Salary Package"


# =========================================================
# FEATURE LIST
# =========================================================

BASE_FEATURES = (

    CATEGORICAL_COLUMNS

    + NUMERICAL_COLUMNS

)


# =========================================================
# VALIDATE DATASET
# =========================================================

def validate_dataset(df):

    """
    Check whether the required columns are present.
    """

    required_columns = (

        BASE_FEATURES

        + [

            CLASSIFICATION_TARGET,

            REGRESSION_TARGET

        ]

    )

    missing_columns = [

        column

        for column in required_columns

        if column not in df.columns

    ]


    if missing_columns:

        raise ValueError(

            "The following required columns "
            "are missing from the dataset:\n"

            + "\n".join(
                missing_columns
            )

        )


    return True


# =========================================================
# LOAD + VALIDATE
# =========================================================

def load_and_validate():

    """
    Load the dataset and validate its structure.
    """

    df = load_data()

    validate_dataset(df)

    return df


# =========================================================
# MAIN TEST
# =========================================================

if __name__ == "__main__":

    df = load_and_validate()

    info = get_dataset_info(df)


    print(
        "PlacementPredict Dataset Loaded"
    )

    print(
        f"Rows: {info['rows']}"
    )

    print(
        f"Columns: {info['columns']}"
    )

    print(
        f"Missing Values: "
        f"{info['missing_values']}"
    )

    print(
        f"Duplicate Rows: "
        f"{info['duplicate_rows']}"
    )

    print(
        "\nColumns:"
    )

    for column in info["column_names"]:

        print(
            f"- {column}"
        )