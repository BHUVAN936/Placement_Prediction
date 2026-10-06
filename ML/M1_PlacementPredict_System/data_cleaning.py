from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# REQUIRED TARGET
# ============================================================

TARGET_COLUMN = "PlacementStatus"


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_column_names(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean column names without changing their meaning.
    """

    data = df.copy()

    data.columns = [

        str(column)
        .strip()
        .replace("\n", " ")

        for column in data.columns

    ]

    return data


# ============================================================
# REMOVE EMPTY COLUMNS
# ============================================================

def remove_empty_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove columns containing only missing values.
    """

    data = df.copy()

    return data.dropna(
        axis=1,
        how="all",
    )


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove completely duplicated rows.
    """

    data = df.copy()

    return data.drop_duplicates(
        keep="first"
    ).reset_index(
        drop=True
    )


# ============================================================
# HANDLE NUMERIC MISSING VALUES
# ============================================================

def fill_numeric_missing_values(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fill missing numeric values with the median.
    """

    data = df.copy()

    numeric_columns = (
        data.select_dtypes(
            include=np.number
        ).columns
    )

    for column in numeric_columns:

        median = data[
            column
        ].median()

        if pd.isna(median):

            median = 0

        data[column] = (
            data[column]
            .fillna(median)
        )

    return data


# ============================================================
# HANDLE CATEGORICAL MISSING VALUES
# ============================================================

def fill_categorical_missing_values(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fill missing categorical values
    using the mode.
    """

    data = df.copy()

    categorical_columns = (
        data.select_dtypes(
            include=[
                "object",
                "category",
            ]
        ).columns
    )

    for column in categorical_columns:

        mode = data[
            column
        ].mode(
            dropna=True
        )

        if len(mode) > 0:

            replacement = mode.iloc[0]

        else:

            replacement = "Unknown"

        data[column] = (
            data[column]
            .fillna(replacement)
        )

    return data


# ============================================================
# CLEAN TARGET
# ============================================================

def clean_target(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate and clean PlacementStatus.
    """

    data = df.copy()

    if TARGET_COLUMN not in data.columns:

        raise ValueError(

            f"Required target column "
            f"'{TARGET_COLUMN}' is missing."

        )

    target = data[
        TARGET_COLUMN
    ]

    # --------------------------------------------------------
    # Convert common text labels to 0/1.
    # --------------------------------------------------------

    if not pd.api.types.is_numeric_dtype(
        target
    ):

        normalized = (
            target
            .astype(str)
            .str.strip()
            .str.lower()
        )

        mapping = {

            "placed": 1,

            "yes": 1,

            "true": 1,

            "1": 1,

            "not placed": 0,

            "not_placed": 0,

            "no": 0,

            "false": 0,

            "0": 0,

        }

        converted = (
            normalized.map(mapping)
        )

        # If values are already numeric-looking,
        # convert them directly.
        numeric_fallback = pd.to_numeric(
            target,
            errors="coerce",
        )

        converted = converted.fillna(
            numeric_fallback
        )

        data[
            TARGET_COLUMN
        ] = converted

    else:

        data[
            TARGET_COLUMN
        ] = pd.to_numeric(
            target,
            errors="coerce",
        )

    # --------------------------------------------------------
    # Remove rows where target is unavailable.
    # --------------------------------------------------------

    data = data.dropna(
        subset=[
            TARGET_COLUMN
        ]
    )

    # --------------------------------------------------------
    # Force integer binary target when possible.
    # --------------------------------------------------------

    unique_values = sorted(
        data[
            TARGET_COLUMN
        ].unique()
        .tolist()
    )

    if set(unique_values).issubset(
        {0, 1}
    ):

        data[
            TARGET_COLUMN
        ] = (
            data[
                TARGET_COLUMN
            ]
            .astype(int)
        )

    return data


# ============================================================
# CLEAN COMPLETE DATASET
# ============================================================

def clean_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Complete M1 cleaning pipeline.

    Steps:
        1. Normalize column names
        2. Remove empty columns
        3. Remove duplicate rows
        4. Clean target
        5. Fill numerical missing values
        6. Fill categorical missing values
        7. Replace infinite values
    """

    if not isinstance(
        df,
        pd.DataFrame,
    ):

        raise TypeError(
            "clean_data() expects "
            "a pandas DataFrame."
        )

    data = df.copy()

    original_rows = len(data)

    # --------------------------------------------------------
    # Step 1
    # --------------------------------------------------------

    data = normalize_column_names(
        data
    )

    # --------------------------------------------------------
    # Step 2
    # --------------------------------------------------------

    data = remove_empty_columns(
        data
    )

    # --------------------------------------------------------
    # Step 3
    # --------------------------------------------------------

    data = remove_duplicates(
        data
    )

    # --------------------------------------------------------
    # Step 4
    # --------------------------------------------------------

    data = clean_target(
        data
    )

    # --------------------------------------------------------
    # Step 5
    # --------------------------------------------------------

    data = fill_numeric_missing_values(
        data
    )

    # --------------------------------------------------------
    # Step 6
    # --------------------------------------------------------

    data = fill_categorical_missing_values(
        data
    )

    # --------------------------------------------------------
    # Step 7
    # --------------------------------------------------------

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # After replacing infinity,
    # fill any new missing numeric values.
    data = fill_numeric_missing_values(
        data
    )

    data = fill_categorical_missing_values(
        data
    )

    data = data.reset_index(
        drop=True
    )

    if data.empty:

        raise ValueError(
            "No rows remain after data cleaning."
        )

    print()
    print("=" * 60)
    print("DATA CLEANING")
    print("=" * 60)

    print(
        f"Original rows : {original_rows:,}"
    )

    print(
        f"Final rows    : {len(data):,}"
    )

    print(
        f"Columns       : {len(data.columns)}"
    )

    print(
        f"Duplicates removed : "
        f"{original_rows - len(data):,}"
    )

    print("=" * 60)
    print()

    return data


# ============================================================
# CLEANING REPORT
# ============================================================

def cleaning_report(
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> dict:
    """
    Return a structured cleaning report.
    """

    return {

        "rows_before":
            int(
                len(before)
            ),

        "rows_after":
            int(
                len(after)
            ),

        "columns_before":
            int(
                len(before.columns)
            ),

        "columns_after":
            int(
                len(after.columns)
            ),

        "duplicates_removed":
            int(
                before.duplicated().sum()
            ),

        "missing_before":
            int(
                before.isna()
                .sum()
                .sum()
            ),

        "missing_after":
            int(
                after.isna()
                .sum()
                .sum()
            ),

        "target":
            TARGET_COLUMN,

        "target_values":
            (
                after[
                    TARGET_COLUMN
                ]
                .value_counts()
                .sort_index()
                .to_dict()
            ),

    }


# ============================================================
# DATASET SUMMARY
# ============================================================

def dataset_summary(
    df: pd.DataFrame,
) -> dict:
    """
    Return useful dataset statistics.
    """

    numeric = (
        df.select_dtypes(
            include=np.number
        )
    )

    categorical = (
        df.select_dtypes(
            include=[
                "object",
                "category",
            ]
        )
    )

    return {

        "rows":
            int(
                len(df)
            ),

        "columns":
            int(
                len(df.columns)
            ),

        "numeric_columns":
            int(
                len(numeric.columns)
            ),

        "categorical_columns":
            int(
                len(categorical.columns)
            ),

        "missing_values":
            int(
                df.isna()
                .sum()
                .sum()
            ),

        "duplicate_rows":
            int(
                df.duplicated()
                .sum()
            ),

    }


# ============================================================
# SCRIPT TEST
# ============================================================

if __name__ == "__main__":

    from data_loader import (
        load_data,
    )

    raw = load_data()

    cleaned = clean_data(
        raw
    )

    report = cleaning_report(
        raw,
        cleaned,
    )

    print(
        report
    )