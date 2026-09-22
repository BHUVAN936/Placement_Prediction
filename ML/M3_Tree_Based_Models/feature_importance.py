from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(
    __file__
).resolve().parents[2]

REPORT_DIR = (
    ROOT
    / "reports"
    / "M3"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def get_feature_importance(
    model,
    feature_names,
):
    """
    Extract feature importance from
    tree-based models.
    """

    if not hasattr(
        model,
        "feature_importances_",
    ):

        raise ValueError(
            "This model does not provide "
            "feature_importances_."
        )

    importance = np.asarray(
        model.feature_importances_
    )

    feature_names = list(
        feature_names
    )

    if len(feature_names) != len(
        importance
    ):

        feature_names = [
            f"Feature_{i}"
            for i in range(
                len(importance)
            )
        ]

    result = pd.DataFrame({

        "Feature":
            feature_names,

        "Importance":
            importance,

    })

    result = (
        result
        .sort_values(
            "Importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return result


def save_feature_importance(
    model,
    feature_names,
    model_name,
    top_n=20,
):
    """
    Save feature importance CSV
    and chart.
    """

    result = get_feature_importance(
        model,
        feature_names,
    )

    csv_name = (
        model_name
        .lower()
        .replace(
            " ",
            "_",
        )
        + "_feature_importance.csv"
    )

    csv_path = (
        REPORT_DIR
        / csv_name
    )

    result.to_csv(
        csv_path,
        index=False,
    )

    top = result.head(
        top_n
    ).sort_values(
        "Importance"
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        top["Feature"],
        top["Importance"],
    )

    plt.title(
        f"{model_name} - Top "
        f"{top_n} Feature Importance"
    )

    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.tight_layout()

    chart_name = (
        model_name
        .lower()
        .replace(
            " ",
            "_",
        )
        + "_feature_importance.png"
    )

    chart_path = (
        REPORT_DIR
        / chart_name
    )

    plt.savefig(
        chart_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()

    return {
        "csv":
            csv_path.name,

        "chart":
            chart_path.name,

        "top_features":
            result.head(
                top_n
            ).to_dict(
                orient="records"
            ),
    }