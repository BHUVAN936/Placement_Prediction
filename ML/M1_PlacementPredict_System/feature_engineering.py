from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = (
    ROOT
    / "reports"
    / "M1"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HELPER
# ============================================================

def safe_numeric(
    df: pd.DataFrame,
    column: str,
):

    if column not in df.columns:

        return None

    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================
# AVERAGE SGPA
# ============================================================

def add_average_sgpa(
    df: pd.DataFrame,
):

    data = df.copy()

    sgpa_columns = [

        column

        for column in data.columns

        if column.startswith(
            "SGPA_Sem"
        )

    ]

    if sgpa_columns:

        data[
            "AverageSGPA"
        ] = (
            data[
                sgpa_columns
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .mean(
                axis=1
            )
        )

    return data


# ============================================================
# SKILL SCORE
# ============================================================

def add_skill_score(
    df: pd.DataFrame,
):

    data = df.copy()

    components = []

    for column in [

        "AptitudeTestScore",

        "SoftSkillsRating",

        "CodingTestScore",

        "MockInterviewScore",

    ]:

        values = safe_numeric(
            data,
            column,
        )

        if values is not None:

            # Normalize each feature to 0-100
            minimum = values.min()
            maximum = values.max()

            if (
                pd.notna(minimum)
                and pd.notna(maximum)
                and maximum != minimum
            ):

                normalized = (
                    (
                        values
                        - minimum
                    )
                    /
                    (
                        maximum
                        - minimum
                    )
                    * 100
                )

            else:

                normalized = values

            components.append(
                normalized
            )

    if components:

        data[
            "SkillScore"
        ] = pd.concat(
            components,
            axis=1,
        ).mean(
            axis=1
        )

    return data


# ============================================================
# EXPERIENCE SCORE
# ============================================================

def add_experience_score(
    df: pd.DataFrame,
):

    data = df.copy()

    components = []

    for column in [

        "Internships",

        "Projects",

        "Workshops",

        "Certifications",

        "Publications",

    ]:

        values = safe_numeric(
            data,
            column,
        )

        if values is not None:

            components.append(
                values.fillna(0)
            )

    if components:

        data[
            "ExperienceScore"
        ] = pd.concat(
            components,
            axis=1,
        ).sum(
            axis=1
        )

    return data


# ============================================================
# ACADEMIC CONSISTENCY
# ============================================================

def add_academic_consistency(
    df: pd.DataFrame,
):

    data = df.copy()

    sgpa_columns = [

        column

        for column in data.columns

        if column.startswith(
            "SGPA_Sem"
        )

    ]

    if len(sgpa_columns) >= 2:

        sgpa = (
            data[
                sgpa_columns
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
        )

        data[
            "AcademicConsistency"
        ] = (
            sgpa
            .std(
                axis=1
            )
            .fillna(0)
        )

    return data


# ============================================================
# ACADEMIC SCORE
# ============================================================

def add_academic_score(
    df: pd.DataFrame,
):

    data = df.copy()

    components = []

    if "CGPA" in data.columns:

        cgpa = safe_numeric(
            data,
            "CGPA",
        )

        components.append(
            cgpa
        )

    if "AttendancePercent" in data.columns:

        attendance = safe_numeric(
            data,
            "AttendancePercent",
        )

        components.append(
            attendance
        )

    if "AverageSGPA" in data.columns:

        average_sgpa = safe_numeric(
            data,
            "AverageSGPA",
        )

        components.append(
            average_sgpa
        )

    if components:

        normalized_components = []

        for values in components:

            minimum = values.min()
            maximum = values.max()

            if (
                pd.notna(minimum)
                and pd.notna(maximum)
                and maximum != minimum
            ):

                normalized = (
                    (
                        values
                        - minimum
                    )
                    /
                    (
                        maximum
                        - minimum
                    )
                    * 100
                )

            else:

                normalized = values

            normalized_components.append(
                normalized
            )

        data[
            "AcademicScore"
        ] = pd.concat(
            normalized_components,
            axis=1,
        ).mean(
            axis=1
        )

    return data


# ============================================================
# INTERVIEW SCORE
# ============================================================

def add_interview_score(
    df: pd.DataFrame,
):

    data = df.copy()

    components = []

    for column in [

        "AptitudeTestScore",

        "CodingTestScore",

        "MockInterviewScore",

        "SoftSkillsRating",

    ]:

        values = safe_numeric(
            data,
            column,
        )

        if values is not None:

            components.append(
                values
            )

    if components:

        normalized_components = []

        for values in components:

            minimum = values.min()
            maximum = values.max()

            if (
                pd.notna(minimum)
                and pd.notna(maximum)
                and maximum != minimum
            ):

                normalized = (
                    (
                        values
                        - minimum
                    )
                    /
                    (
                        maximum
                        - minimum
                    )
                    * 100
                )

            else:

                normalized = values

            normalized_components.append(
                normalized
            )

        data[
            "InterviewScore"
        ] = pd.concat(
            normalized_components,
            axis=1,
        ).mean(
            axis=1
        )

    return data


# ============================================================
# COMPLETE FEATURE ENGINEERING
# ============================================================

def engineer_features(
    df: pd.DataFrame,
):

    if not isinstance(
        df,
        pd.DataFrame,
    ):

        raise TypeError(
            "engineer_features() "
            "expects a pandas DataFrame."
        )

    data = df.copy()

    # --------------------------------------------------------
    # Academic features
    # --------------------------------------------------------

    data = add_average_sgpa(
        data
    )

    data = add_academic_consistency(
        data
    )

    data = add_academic_score(
        data
    )

    # --------------------------------------------------------
    # Skill features
    # --------------------------------------------------------

    data = add_skill_score(
        data
    )

    data = add_interview_score(
        data
    )

    # --------------------------------------------------------
    # Experience
    # --------------------------------------------------------

    data = add_experience_score(
        data
    )

    # --------------------------------------------------------
    # Final cleanup
    # --------------------------------------------------------

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    engineered_columns = [

        "AverageSGPA",

        "SkillScore",

        "ExperienceScore",

        "AcademicConsistency",

        "AcademicScore",

        "InterviewScore",

    ]

    for column in engineered_columns:

        if column in data.columns:

            data[column] = (
                pd.to_numeric(
                    data[column],
                    errors="coerce",
                )
                .fillna(0)
            )

    return data


# ============================================================
# FEATURE ENGINEERING REPORT
# ============================================================

def get_feature_engineering_report(
    before: pd.DataFrame,
    after: pd.DataFrame,
):

    engineered = [

        column

        for column in after.columns

        if column not in before.columns

    ]

    return {

        "original_feature_count":
            int(
                len(before.columns)
            ),

        "final_feature_count":
            int(
                len(after.columns)
            ),

        "new_feature_count":
            int(
                len(engineered)
            ),

        "new_features":
            engineered,

    }


# ============================================================
# SAVE FEATURE DATA
# ============================================================

def save_engineered_data(
    df: pd.DataFrame,
    filename="m1_engineered_dataset.csv",
):

    output_path = (
        REPORT_DIR
        / filename
    )

    df.to_csv(
        output_path,
        index=False,
    )

    return output_path


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

    engineered = engineer_features(
        cleaned
    )

    report = (
        get_feature_engineering_report(
            cleaned,
            engineered,
        )
    )

    output = save_engineered_data(
        engineered
    )

    print()
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    print(
        "Original features:",
        report[
            "original_feature_count"
        ],
    )

    print(
        "Final features:",
        report[
            "final_feature_count"
        ],
    )

    print(
        "New features:",
        report[
            "new_features"
        ],
    )

    print(
        "Saved:",
        output,
    )

    print("=" * 60)