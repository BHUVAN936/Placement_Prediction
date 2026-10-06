from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = ROOT / "reports" / "M1"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# EDA CLASS
# ============================================================

class EDAAnalyzer:
    """
    Exploratory Data Analysis for PlacementPredict.

    Generates:
        - dataset statistics
        - missing-value report
        - numerical summary
        - categorical summary
        - placement distribution
        - CGPA distribution
        - attendance distribution
        - internships distribution
        - correlation heatmap
        - academic feature comparison
    """

    def __init__(
        self,
        df: pd.DataFrame,
        output_dir: Path | None = None,
    ):

        self.df = df.copy()

        self.output_dir = (
            output_dir
            if output_dir is not None
            else REPORT_DIR
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # BASIC INFORMATION
    # ========================================================

    def basic_information(self):

        numeric_columns = (
            self.df
            .select_dtypes(
                include=np.number
            )
            .columns
            .tolist()
        )

        categorical_columns = (
            self.df
            .select_dtypes(
                include=[
                    "object",
                    "category",
                ]
            )
            .columns
            .tolist()
        )

        return {

            "rows":
                int(len(self.df)),

            "columns":
                int(len(self.df.columns)),

            "numeric_columns":
                numeric_columns,

            "categorical_columns":
                categorical_columns,

            "missing_values":
                int(
                    self.df.isna()
                    .sum()
                    .sum()
                ),

            "duplicate_rows":
                int(
                    self.df.duplicated()
                    .sum()
                ),

        }

    # ========================================================
    # NUMERICAL SUMMARY
    # ========================================================

    def numerical_summary(self):

        numeric = (
            self.df
            .select_dtypes(
                include=np.number
            )
        )

        if numeric.empty:
            return []

        summary = (
            numeric
            .describe()
            .T
            .reset_index()
            .rename(
                columns={
                    "index": "feature"
                }
            )
        )

        return (
            summary
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0)
            .to_dict(
                orient="records"
            )
        )

    # ========================================================
    # MISSING VALUE SUMMARY
    # ========================================================

    def missing_value_summary(self):

        missing = (
            self.df
            .isna()
            .sum()
            .reset_index()
        )

        missing.columns = [
            "feature",
            "missing_count",
        ]

        missing = missing[
            missing[
                "missing_count"
            ] > 0
        ]

        return (
            missing
            .sort_values(
                "missing_count",
                ascending=False,
            )
            .to_dict(
                orient="records"
            )
        )

    # ========================================================
    # CATEGORICAL SUMMARY
    # ========================================================

    def categorical_summary(self):

        categorical = (
            self.df
            .select_dtypes(
                include=[
                    "object",
                    "category",
                ]
            )
        )

        result = []

        for column in categorical.columns:

            counts = (
                self.df[column]
                .astype(str)
                .value_counts()
                .head(15)
            )

            result.append({

                "feature":
                    column,

                "unique_values":
                    int(
                        self.df[
                            column
                        ].nunique(
                            dropna=True
                        )
                    ),

                "values":
                    counts.to_dict(),

            })

        return result

    # ========================================================
    # PLACEMENT DISTRIBUTION
    # ========================================================

    def placement_distribution(self):

        if (
            "PlacementStatus"
            not in self.df.columns
        ):

            return {}

        values = (
            self.df[
                "PlacementStatus"
            ]
            .value_counts()
            .sort_index()
        )

        return {
            str(key): int(value)
            for key, value
            in values.items()
        }

    # ========================================================
    # SAVE CSV
    # ========================================================

    def save_csv_reports(self):

        # Numerical summary
        numeric = (
            self.df
            .select_dtypes(
                include=np.number
            )
        )

        if not numeric.empty:

            (
                numeric
                .describe()
                .T
                .to_csv(
                    self.output_dir
                    / "m1_numerical_summary.csv"
                )
            )

        # Missing values
        missing = (
            self.df
            .isna()
            .sum()
            .reset_index()
        )

        missing.columns = [
            "feature",
            "missing_count",
        ]

        missing.to_csv(
            self.output_dir
            / "m1_missing_values.csv",
            index=False,
        )

        # Categorical summary
        categorical = (
            self.df
            .select_dtypes(
                include=[
                    "object",
                    "category",
                ]
            )
        )

        rows = []

        for column in categorical.columns:

            counts = (
                self.df[column]
                .astype(str)
                .value_counts()
            )

            for value, count in (
                counts.items()
            ):

                rows.append({

                    "feature":
                        column,

                    "value":
                        value,

                    "count":
                        int(count),

                })

        if rows:

            pd.DataFrame(
                rows
            ).to_csv(
                self.output_dir
                / "m1_categorical_summary.csv",
                index=False,
            )

    # ========================================================
    # CHART HELPER
    # ========================================================

    def _save_chart(
        self,
        filename: str,
    ):

        path = (
            self.output_dir
            / filename
        )

        plt.tight_layout()

        plt.savefig(
            path,
            dpi=160,
            bbox_inches="tight",
        )

        plt.close()

        return filename

    # ========================================================
    # PLACEMENT CHART
    # ========================================================

    def plot_placement_distribution(self):

        if (
            "PlacementStatus"
            not in self.df.columns
        ):
            return None

        counts = (
            self.df[
                "PlacementStatus"
            ]
            .value_counts()
            .sort_index()
        )

        plt.figure(
            figsize=(7, 5)
        )

        counts.plot(
            kind="bar"
        )

        plt.title(
            "Placement Status Distribution"
        )

        plt.xlabel(
            "Placement Status"
        )

        plt.ylabel(
            "Number of Students"
        )

        plt.xticks(
            rotation=0
        )

        return self._save_chart(
            "placement_distribution.png"
        )

    # ========================================================
    # PLACEMENT PIE CHART
    # ========================================================

    def plot_placement_pie(self):

        if (
            "PlacementStatus"
            not in self.df.columns
        ):
            return None

        counts = (
            self.df[
                "PlacementStatus"
            ]
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
            autopct="%1.1f%%",
            startangle=90,
        )

        plt.title(
            "Placement Status Percentage"
        )

        return self._save_chart(
            "placement_pie.png"
        )

    # ========================================================
    # CGPA DISTRIBUTION
    # ========================================================

    def plot_cgpa_distribution(self):

        if "CGPA" not in self.df.columns:
            return None

        plt.figure(
            figsize=(8, 5)
        )

        sns.histplot(
            self.df["CGPA"],
            bins=30,
            kde=True,
        )

        plt.title(
            "CGPA Distribution"
        )

        plt.xlabel(
            "CGPA"
        )

        plt.ylabel(
            "Students"
        )

        return self._save_chart(
            "cgpa_distribution.png"
        )

    # ========================================================
    # ATTENDANCE DISTRIBUTION
    # ========================================================

    def plot_attendance_distribution(self):

        if (
            "AttendancePercent"
            not in self.df.columns
        ):
            return None

        plt.figure(
            figsize=(8, 5)
        )

        sns.histplot(
            self.df[
                "AttendancePercent"
            ],
            bins=30,
            kde=True,
        )

        plt.title(
            "Attendance Percentage Distribution"
        )

        plt.xlabel(
            "Attendance Percentage"
        )

        plt.ylabel(
            "Students"
        )

        return self._save_chart(
            "attendance_distribution.png"
        )

    # ========================================================
    # INTERNSHIPS DISTRIBUTION
    # ========================================================

    def plot_internships(self):

        if (
            "Internships"
            not in self.df.columns
        ):
            return None

        counts = (
            self.df[
                "Internships"
            ]
            .value_counts()
            .sort_index()
        )

        plt.figure(
            figsize=(8, 5)
        )

        counts.plot(
            kind="bar"
        )

        plt.title(
            "Internship Distribution"
        )

        plt.xlabel(
            "Number of Internships"
        )

        plt.ylabel(
            "Students"
        )

        plt.xticks(
            rotation=0
        )

        return self._save_chart(
            "internships_distribution.png"
        )

    # ========================================================
    # SGPA COMPARISON
    # ========================================================

    def plot_sgpa_distribution(self):

        sgpa_columns = [

            column

            for column in self.df.columns

            if column.startswith(
                "SGPA_Sem"
            )

        ]

        if not sgpa_columns:
            return None

        averages = (
            self.df[
                sgpa_columns
            ]
            .mean()
        )

        plt.figure(
            figsize=(10, 5)
        )

        averages.plot(
            kind="bar"
        )

        plt.title(
            "Average SGPA by Semester"
        )

        plt.xlabel(
            "Semester"
        )

        plt.ylabel(
            "Average SGPA"
        )

        plt.xticks(
            rotation=45
        )

        return self._save_chart(
            "semester_sgpa.png"
        )

    # ========================================================
    # ACADEMIC FEATURES
    # ========================================================

    def plot_academic_features(self):

        possible = [

            "CGPA",

            "AttendancePercent",

            "AptitudeTestScore",

            "CodingTestScore",

            "MockInterviewScore",

            "SoftSkillsRating",

        ]

        available = [

            column

            for column in possible

            if column in self.df.columns

        ]

        if not available:
            return None

        means = (
            self.df[
                available
            ]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        plt.figure(
            figsize=(10, 6)
        )

        means.plot(
            kind="bar"
        )

        plt.title(
            "Average Academic and Skill Features"
        )

        plt.xlabel(
            "Feature"
        )

        plt.ylabel(
            "Average Value"
        )

        plt.xticks(
            rotation=45,
            ha="right",
        )

        return self._save_chart(
            "academic_skill_features.png"
        )

    # ========================================================
    # CORRELATION HEATMAP
    # ========================================================

    def plot_correlation(self):

        numeric = (
            self.df
            .select_dtypes(
                include=np.number
            )
        )

        if numeric.shape[1] < 2:
            return None

        # Limit extremely large heatmaps
        # while keeping important features.
        if numeric.shape[1] > 25:

            preferred = [

                "PlacementStatus",

                "Salary Package",

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

            ]

            available = [

                column

                for column in preferred

                if column in numeric.columns

            ]

            if len(available) >= 2:

                numeric = numeric[
                    available
                ]

        correlation = (
            numeric.corr()
        )

        plt.figure(
            figsize=(12, 9)
        )

        sns.heatmap(
            correlation,
            cmap="coolwarm",
            center=0,
            annot=False,
            square=False,
        )

        plt.title(
            "Feature Correlation Heatmap"
        )

        return self._save_chart(
            "correlation_heatmap.png"
        )

    # ========================================================
    # PLACEMENT VS CGPA
    # ========================================================

    def plot_cgpa_vs_placement(self):

        if not all(
            column in self.df.columns
            for column in [
                "CGPA",
                "PlacementStatus",
            ]
        ):

            return None

        plt.figure(
            figsize=(8, 5)
        )

        sns.boxplot(
            data=self.df,
            x="PlacementStatus",
            y="CGPA",
        )

        plt.title(
            "CGPA vs Placement Status"
        )

        plt.xlabel(
            "Placement Status"
        )

        plt.ylabel(
            "CGPA"
        )

        return self._save_chart(
            "cgpa_vs_placement.png"
        )

    # ========================================================
    # CODING SCORE VS PLACEMENT
    # ========================================================

    def plot_coding_vs_placement(self):

        if not all(
            column in self.df.columns
            for column in [
                "CodingTestScore",
                "PlacementStatus",
            ]
        ):

            return None

        plt.figure(
            figsize=(8, 5)
        )

        sns.boxplot(
            data=self.df,
            x="PlacementStatus",
            y="CodingTestScore",
        )

        plt.title(
            "Coding Test Score vs Placement"
        )

        plt.xlabel(
            "Placement Status"
        )

        plt.ylabel(
            "Coding Test Score"
        )

        return self._save_chart(
            "coding_vs_placement.png"
        )

    # ========================================================
    # RUN COMPLETE EDA
    # ========================================================

    def run(self):

        charts = []

        chart_methods = [

            self.plot_placement_distribution,

            self.plot_placement_pie,

            self.plot_cgpa_distribution,

            self.plot_attendance_distribution,

            self.plot_internships,

            self.plot_sgpa_distribution,

            self.plot_academic_features,

            self.plot_correlation,

            self.plot_cgpa_vs_placement,

            self.plot_coding_vs_placement,

        ]

        for method in chart_methods:

            try:

                result = method()

                if result:
                    charts.append(result)

            except Exception as exc:

                print(
                    f"EDA chart warning: "
                    f"{method.__name__}: "
                    f"{exc}"
                )

        self.save_csv_reports()

        return {

            "basic_information":
                self.basic_information(),

            "numerical_summary":
                self.numerical_summary(),

            "missing_values":
                self.missing_value_summary(),

            "categorical_summary":
                self.categorical_summary(),

            "placement_distribution":
                self.placement_distribution(),

            "charts":
                charts,

        }


# ============================================================
# FUNCTION API
# ============================================================

def run_eda(
    df: pd.DataFrame,
    output_dir: Path | None = None,
):

    analyzer = EDAAnalyzer(
        df,
        output_dir,
    )

    return analyzer.run()


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

    result = run_eda(
        cleaned
    )

    print(
        "EDA completed."
    )

    print(
        f"Charts generated: "
        f"{len(result['charts'])}"
    )