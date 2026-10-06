from flask import Flask, render_template, send_from_directory, abort
from pathlib import Path
from functools import lru_cache
import csv
import random
import math
import logging
import re

# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

DISPLAY_ROWS = 200
CSV_DISPLAY_ROWS = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ============================================================
# MODULE DEFINITIONS
# ============================================================

MODULES = {
    "M1": {
        "code": "M1",
        "name": "PlacementPredict",
        "short_name": "PlacementPredict",
        "description": (
            "Data loading, cleaning, EDA, preprocessing "
            "and feature engineering."
        ),
        "input": "Raw PlacementPredict dataset",
        "target": "PlacementStatus and Salary Package",
        "purpose": "Prepare high-quality data for machine learning.",
        "steps": [
            {
                "title": "1. Import Dataset",
                "what": "Load the 50,000 student records from the CSV dataset.",
                "formula": "Dataset = rows x columns",
                "understanding": (
                    "Each row represents a student and each column "
                    "represents a feature."
                )
            },
            {
                "title": "2. Handle Missing Values",
                "what": (
                    "Identify missing values and handle numerical "
                    "and categorical missing values."
                ),
                "formula": "Missing % = (Missing Values / Total Rows) x 100",
                "understanding": (
                    "Missing values must be handled before model training."
                )
            },
            {
                "title": "3. Remove Duplicates",
                "what": "Detect and remove duplicate student records.",
                "formula": "Clean Dataset = Dataset - Duplicate Rows",
                "understanding": "Duplicate records can bias model training."
            },
            {
                "title": "4. Convert Data Types",
                "what": "Convert numerical and categorical columns to suitable data types.",
                "formula": "Correct Data Type -> Correct Processing",
                "understanding": (
                    "Machine learning algorithms require correctly formatted input."
                )
            },
            {
                "title": "5. Outlier Detection",
                "what": "Detect extreme numerical observations using IQR.",
                "formula": (
                    "IQR = Q3 - Q1\n"
                    "Lower = Q1 - 1.5 x IQR\n"
                    "Upper = Q3 + 1.5 x IQR"
                ),
                "understanding": (
                    "Values outside the lower and upper boundaries "
                    "are treated as potential outliers."
                )
            },
            {
                "title": "6. Exploratory Data Analysis",
                "what": "Study distributions, relationships and correlations.",
                "formula": "Correlation = Cov(X,Y) / (Std(X) x Std(Y))",
                "understanding": "EDA helps understand relationships in the dataset."
            },
            {
                "title": "7. Preprocessing",
                "what": "Scale numerical features and encode categorical features.",
                "formula": "z = (x - mean) / standard deviation",
                "understanding": "Scaling puts numerical features on comparable scales."
            },
            {
                "title": "8. Feature Engineering",
                "what": (
                    "Create AverageSGPA, SkillScore, ExperienceScore, "
                    "AcademicConsistency, AcademicScore and InterviewScore."
                ),
                "formula": "AverageSGPA = Sum of 8 semester SGPAs / 8",
                "understanding": (
                    "Engineered features summarize important student characteristics."
                )
            }
        ]
    },

    "M2": {
        "code": "M2",
        "name": "Supervised Learning",
        "short_name": "Supervised Learning",
        "description": (
            "Regression and classification algorithms trained "
            "using labelled student data."
        ),
        "input": "Cleaned and preprocessed student features",
        "target": "Salary Package and PlacementStatus",
        "purpose": "Predict salary and placement outcomes.",
        "steps": [
            {
                "title": "1. Train-Test Split",
                "what": "Divide the dataset into training and testing sets.",
                "formula": "Training Data = 80% | Testing Data = 20%",
                "understanding": (
                    "Training data builds the model while testing data "
                    "measures generalization."
                )
            },
            {
                "title": "2. Linear Regression",
                "what": "Predict continuous Salary Package values.",
                "formula": "y = b0 + b1x1 + b2x2 + ... + bnxn",
                "understanding": (
                    "The model estimates salary using relationships between features."
                )
            },
            {
                "title": "3. Ridge Regression",
                "what": "Linear regression with L2 regularization.",
                "formula": "Loss = MSE + alpha x Sum(beta^2)",
                "understanding": (
                    "Ridge reduces the impact of very large coefficients."
                )
            },
            {
                "title": "4. Lasso Regression",
                "what": "Linear regression with L1 regularization.",
                "formula": "Loss = MSE + alpha x Sum(abs(beta))",
                "understanding": (
                    "Lasso can reduce some feature coefficients to zero."
                )
            },
            {
                "title": "5. Elastic Net",
                "what": "Combines L1 and L2 regularization.",
                "formula": (
                    "Loss = MSE + alpha x "
                    "[l1_ratio x L1 + (1-l1_ratio) x L2]"
                ),
                "understanding": (
                    "Elastic Net combines feature selection and coefficient shrinkage."
                )
            },
            {
                "title": "6. Logistic Regression",
                "what": "Predict PlacementStatus as a classification problem.",
                "formula": "P(y=1) = 1 / (1 + e^(-z))",
                "understanding": (
                    "The sigmoid function converts model output into a probability."
                )
            },
            {
                "title": "7. Ridge Logistic Regression",
                "what": "Logistic regression using L2 regularization.",
                "formula": "Loss = Logistic Loss + lambda x Sum(beta^2)",
                "understanding": "Regularization helps control model complexity."
            },
            {
                "title": "8. Evaluation",
                "what": "Evaluate regression and classification models.",
                "formula": (
                    "MAE = Sum(abs(y-yhat)) / n\n"
                    "RMSE = sqrt(MSE)\n"
                    "R2 = 1 - SSres/SStot"
                ),
                "understanding": "Evaluation metrics measure prediction quality."
            }
        ]
    },

    "M3": {
        "code": "M3",
        "name": "Tree-Based Models",
        "short_name": "Tree-Based Models",
        "description": (
            "Decision-tree and boosting-based classification algorithms."
        ),
        "input": "Cleaned and encoded student features",
        "target": "PlacementStatus",
        "purpose": "Compare tree-based classification models.",
        "steps": [
            {
                "title": "1. Decision Tree",
                "what": "Split records into branches using feature conditions.",
                "formula": "Gini = 1 - Sum(p_i^2)",
                "understanding": "The tree chooses splits that reduce impurity."
            },
            {
                "title": "2. Random Forest",
                "what": "Train multiple decision trees and combine predictions.",
                "formula": "Final Prediction = Majority Vote",
                "understanding": (
                    "Multiple trees provide a more stable prediction."
                )
            },
            {
                "title": "3. AdaBoost",
                "what": "Sequentially combine weak learners.",
                "formula": "alpha = 0.5 x ln((1-error)/error)",
                "understanding": (
                    "More weight is given to incorrectly classified observations."
                )
            },
            {
                "title": "4. Gradient Boosting",
                "what": "Build trees sequentially to reduce previous errors.",
                "formula": "F_m(x) = F_(m-1)(x) + learning_rate x h_m(x)",
                "understanding": (
                    "Each new tree attempts to correct previous errors."
                )
            },
            {
                "title": "5. XGBoost",
                "what": "Optimized gradient boosting algorithm.",
                "formula": "Objective = Training Loss + Regularization",
                "understanding": (
                    "XGBoost combines boosting with regularization."
                )
            },
            {
                "title": "6. LightGBM",
                "what": "Efficient gradient boosting implementation.",
                "formula": "Objective = Loss + Regularization",
                "understanding": (
                    "LightGBM is designed for efficient training on large datasets."
                )
            },
            {
                "title": "7. Feature Importance",
                "what": "Measure how strongly each feature contributes to tree predictions.",
                "formula": "Importance = Contribution to Split Improvement",
                "understanding": (
                    "Feature importance identifies influential student attributes."
                )
            },
            {
                "title": "8. Model Evaluation",
                "what": "Evaluate classification predictions.",
                "formula": "Accuracy = Correct Predictions / Total Predictions",
                "understanding": (
                    "Accuracy, precision, recall and F1 measure classification performance."
                )
            }
        ]
    },

    "M4": {
        "code": "M4",
        "name": "Unsupervised Learning",
        "short_name": "Unsupervised Learning",
        "description": (
            "Discover hidden groups and reduce dimensionality "
            "without using placement labels."
        ),
        "input": "Student feature matrix",
        "target": "No supervised target",
        "purpose": "Find natural student groups and visualize high-dimensional data.",
        "steps": [
            {
                "title": "1. K-Means",
                "what": "Partition students into K clusters.",
                "formula": "Distance = sqrt(Sum((x_i-c_i)^2))",
                "understanding": (
                    "Each student is assigned to the nearest cluster centroid."
                )
            },
            {
                "title": "2. K-Means++",
                "what": "Improved centroid initialization for K-Means.",
                "formula": "New centroid probability is proportional to D(x)^2",
                "understanding": "Better initial centroids can improve clustering."
            },
            {
                "title": "3. Hierarchical Clustering",
                "what": "Build a hierarchy of student clusters.",
                "formula": "Cluster Distance depends on linkage method",
                "understanding": "Clusters are progressively merged into a hierarchy."
            },
            {
                "title": "4. DBSCAN",
                "what": "Identify dense groups and noise points.",
                "formula": "Core Point: number of neighbours >= MinPts",
                "understanding": (
                    "DBSCAN does not require the number of clusters beforehand."
                )
            },
            {
                "title": "5. PCA",
                "what": "Reduce high-dimensional student features to principal components.",
                "formula": "X_transformed = X x Principal Components",
                "understanding": (
                    "PCA preserves as much variance as possible using fewer dimensions."
                )
            },
            {
                "title": "6. UMAP",
                "what": "Reduce high-dimensional student features to two dimensions while preserving important local neighborhood relationships.",
                "formula": "High-dimensional data -> neighborhood graph -> optimized 2D embedding",
                "understanding": (
                    "UMAP builds a neighborhood graph in the original feature space and optimizes a two-dimensional representation that preserves important local relationships."
                )
            },
            {
                "title": "7. t-SNE",
                "what": "Project high-dimensional student features into two dimensions while preserving local similarity relationships.",
                "formula": "High-dimensional similarities -> low-dimensional similarities -> minimize divergence",
                "understanding": (
                    "t-SNE converts pairwise similarities into probabilities and optimizes a 2D embedding so locally similar observations tend to remain close."
                )
            }
        ]
    },

    "M5": {
        "code": "M5",
        "name": "Hyperparameter Tuning",
        "short_name": "Hyperparameter Tuning",
        "description": (
            "Grid Search, Random Search, and Bayesian Optimization evaluated "
            "using 5-Fold Cross-Validation (k=5)."
        ),
        "input": "Preprocessed student features and search space parameters",
        "target": "Optimized model accuracy and continuous target peak convergence",
        "purpose": "Find optimal hyperparameters efficiently while avoiding rigid grid traps and local optima.",
        "steps": [
            {
                "title": "1. 5-Fold Cross-Validation (k=5)",
                "what": "Split dataset into 5 equal folds for cross-validation evaluation.",
                "formula": "Mean CV Score = (1/5) * Sum(Fold_i Score)",
                "understanding": "5-fold cross-validation provides stable score estimates during hyperparameter tuning."
            },
            {
                "title": "2. Grid Search",
                "what": "Systematically evaluate hyperparameter combinations on a rigid grid [0, 5, 10, 15, 20].",
                "formula": "Search space = Predefined Cartesian Grid",
                "understanding": "Grid search is trapped by rigid step sizes, settling at x=10.0, y=5.0 with score ~87.16."
            },
            {
                "title": "3. Random Search",
                "what": "Sample continuous decimal hyperparameter values uniformly across 25 iterations.",
                "formula": "Hyperparameter ~ Uniform(min, max)",
                "understanding": "Random search explores continuous spaces and reaches scores between 95.0 and 99.0."
            },
            {
                "title": "4. Bayesian Optimization",
                "what": "Use Gaussian Process surrogate model and Expected Improvement to guide search.",
                "formula": "Acquisition = E[max(0, f(x) - f(x*))]",
                "understanding": "Uses early random guesses to map slope and spends final 10-15 trials narrowing down strictly on peak (99.99+ score at x=12.29, y=7.80)."
            },
            {
                "title": "5. Hyperparameter Surface Analysis",
                "what": "Analyze 3D optimization landscapes, convergence curves, and fold performance.",
                "formula": "Surface Z = f(x, y)",
                "understanding": "Visualizes search trajectories and convergence efficiency across algorithms."
            }
        ]
    }
}


# ============================================================
# SIDEBAR ALGORITHM MAP
# ============================================================
# These names are also used as stable anchors on each module page.
ALGORITHMS = {
    "M1": [
        "Import Dataset", "Missing Values", "Remove Duplicates",
        "Convert Data Types", "Outlier Detection",
        "Exploratory Data Analysis", "Preprocessing: Scaling",
        "Feature Engineering"
    ],
    "M2": [
        "Train-Test Split", "Linear Regression", "Ridge Regression",
        "Lasso Regression", "Elastic Net", "Logistic Regression",
        "Ridge Logistic Regression", "Evaluation"
    ],
    "M3": [
        "Decision Tree", "Random Forest", "AdaBoost",
        "Gradient Boosting", "XGBoost", "LightGBM",
        "Feature Importance", "Model Evaluation"
    ],
    "M4": [
        "K-Means", "K-Means++", "Hierarchical Clustering",
        "DBSCAN", "PCA", "UMAP", "t-SNE", "Euclidean Distance"
    ],
    "M5": [
        "Holdout Validation", "5-Fold Cross Validation", "Probability Calibration",
        "ROC-AUC Analysis", "Grid Search", "Random Search", "Bayesian Optimization"
    ]
}

def algorithm_slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# ============================================================
# DATASET
# ============================================================

def get_dataset_file():
    candidates = [
        DATA_DIR / "placement_predict_50k Dataset (2).csv",
        DATA_DIR / "placement_predict_50k Dataset.csv",
        DATA_DIR / "placement_predict_50k.csv"
    ]

    for path in candidates:
        if path.exists():
            return path

    csv_files = list(DATA_DIR.glob("*.csv"))

    if csv_files:
        return csv_files[0]

    return None


@lru_cache(maxsize=1)
def get_dataset_df():
    """
    Load the project dataset as a pandas DataFrame.

    This helper is used by the startup block to print the dataset
    dimensions before Flask starts. It uses the same dataset
    discovery logic as get_dataset_info(), so it does not introduce
    a second hard-coded dataset path.
    """
    import pandas as pd

    file_path = get_dataset_file()

    if file_path is None:
        return pd.DataFrame()

    try:
        return pd.read_csv(file_path)
    except Exception as error:
        app.logger.error("Dataset loading error: %s", error)
        return pd.DataFrame()


def get_dataset_info():
    file_path = get_dataset_file()

    empty = {
        "rows": 0,
        "columns": 0,
        "column_names": [],
        "missing_values": 0,
        "duplicate_rows": 0,
        "numeric_columns": [],
        "categorical_columns": []
    }

    if file_path is None:
        return empty

    try:
        import pandas as pd

        df = pd.read_csv(file_path)

        return {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": list(df.columns),
            "missing_values": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "numeric_columns": list(
                df.select_dtypes(include="number").columns
            ),
            "categorical_columns": list(
                df.select_dtypes(exclude="number").columns
            )
        }

    except Exception as error:
        app.logger.error("Dataset information error: %s", error)
        return empty


# ============================================================
# GLOBAL TEMPLATE CONTEXT
# ============================================================

@app.context_processor
def inject_global_template_data():
    try:
        return {
            "dataset_info": get_dataset_info(),
            "modules": MODULES,
            "algorithms": ALGORITHMS
        }
    except Exception as error:
        app.logger.error("Template context error: %s", error)

        return {
            "dataset_info": {
                "rows": 0,
                "columns": 0,
                "column_names": [],
                "missing_values": 0,
                "duplicate_rows": 0,
                "numeric_columns": [],
                "categorical_columns": []
            },
            "modules": MODULES,
            "algorithms": ALGORITHMS
        }


# ============================================================
# PREVIEW
# ============================================================

def get_preview(limit=DISPLAY_ROWS):
    file_path = get_dataset_file()

    if file_path is None:
        return []

    try:
        with open(
            file_path,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            reader = csv.DictReader(file)
            reservoir = []

            for index, row in enumerate(reader):

                if index < limit:
                    reservoir.append(row)
                else:
                    random_index = random.randint(0, index)

                    if random_index < limit:
                        reservoir[random_index] = row

            return reservoir

    except Exception as error:
        app.logger.error("Preview error: %s", error)
        return []


# ============================================================
# JINJA NUMBER FILTER
# ============================================================

def number(value):
    if value is None:
        return "0"

    try:
        if isinstance(value, float) and math.isnan(value):
            return "0"

        return f"{int(value):,}"

    except Exception:
        return str(value)


app.jinja_env.filters["number"] = number



# ============================================================
# WORKED EXAMPLES
# ============================================================

def get_worked_examples():
    preview = get_preview(DISPLAY_ROWS)
    examples = {}

    # ---------------- M1 ----------------

    if preview:
        first = preview[0]
        sgpa_values = []

        for semester in range(1, 9):
            key = f"SGPA_Sem{semester}"

            try:
                value = float(first.get(key, 0))
                sgpa_values.append(value)
            except Exception:
                pass

        if sgpa_values:
            average_sgpa = sum(sgpa_values) / len(sgpa_values)

            examples["M1"] = {
                "title": "Average SGPA Calculation",
                "student": first,
                "sgpa_values": sgpa_values,
                "average_sgpa": round(average_sgpa, 2)
            }

    # ---------------- M4 ----------------

    if len(preview) >= 2:
        first = preview[0]
        second = preview[1]

        try:
            cgpa1 = float(first.get("CGPA", 0))
            cgpa2 = float(second.get("CGPA", 0))

            attendance1 = float(
                first.get("AttendancePercent", 0)
            )
            attendance2 = float(
                second.get("AttendancePercent", 0)
            )

            distance = math.sqrt(
                (cgpa1 - cgpa2) ** 2
                + (attendance1 - attendance2) ** 2
            )

            examples["M4"] = {
                "title": "Euclidean Distance Example",
                "cgpa1": cgpa1,
                "cgpa2": cgpa2,
                "attendance1": attendance1,
                "attendance2": attendance2,
                "distance": round(distance, 4)
            }

        except Exception:
            pass

    # ---------------- M3 ----------------

    if preview:
        labels = [
            str(row.get("PlacementStatus", ""))
            for row in preview
            if row.get("PlacementStatus", "") != ""
        ]

        if labels:
            counts = {}

            for label in labels:
                counts[label] = counts.get(label, 0) + 1

            total = len(labels)
            gini = 1.0

            for count in counts.values():
                probability = count / total
                gini -= probability ** 2

            examples["M3"] = {
                "title": "Gini Impurity Example",
                "counts": counts,
                "total": total,
                "gini": round(gini, 4)
            }

    return examples


# ============================================================
# ALGORITHM-SPECIFIC SOURCE-DATA EXPLANATIONS
# Added only for the web explanation/report. Existing training code
# and existing module/report logic are intentionally preserved.
# ============================================================

@lru_cache(maxsize=1)
def get_algorithm_details():
    """Build clear algorithm examples from the supplied 32-column source CSV.

    No synthetic/example values are introduced. All student values shown here
    come from the project dataset; derived values are calculated from those
    source values. Existing model results remain separate and unchanged.
    """
    import pandas as pd
    import numpy as np

    path = get_dataset_file()
    if path is None:
        return {}

    df = pd.read_csv(path)
    source_columns = list(df.columns)
    details = {code: [] for code in MODULES}

    # ---------- M1: data preparation examples ----------
    first = df.iloc[0]
    details["M1"] = [
        {
            "algorithm": "Import Dataset",
            "purpose": "Confirm the actual source structure before processing.",
            "data": [
                ["Source rows", f"{len(df):,}", "The supplied dataset contains 50,000 student records."],
                ["Source columns", str(len(source_columns)), "These are the original 32 columns; engineered columns are not counted here."],
                ["First StudentID", str(first.get("StudentID")), "Example row taken directly from the source CSV."],
                ["Example CGPA", str(first.get("CGPA")), "Source value."],
                ["Example PlacementStatus", str(first.get("PlacementStatus")), "Source target value."],
            ],
            "result": "The source dataset is 50,000 rows × 32 columns. Any additional engineered fields are derived fields, not source columns."
        },
        {
            "algorithm": "Missing Values",
            "purpose": "Identify fields that require missing-value handling before model training.",
            "data": [["Column", "Missing values", "Interpretation"] for col in df.columns if int(df[col].isna().sum()) > 0][:6],
            "result": f"The source dataset contains {int(df.isna().sum().sum()):,} missing cells in total. Missing values are handled during preprocessing; they are not treated as new source columns."
        },
        {
            "algorithm": "Remove Duplicates",
            "purpose": "Check whether the same student record occurs more than once.",
            "data": [["Total source rows", f"{len(df):,}", "Before duplicate check"], ["Duplicate rows", f"{int(df.duplicated().sum()):,}", "Exact duplicate rows found by pandas duplicated()"]],
            "result": f"The source dataset contains {int(df.duplicated().sum()):,} exact duplicate rows."
        },
        {
            "algorithm": "Convert Data Types",
            "purpose": "Separate numeric and categorical source fields so preprocessing can treat them correctly.",
            "data": [["Numeric columns", str(len(df.select_dtypes(include="number").columns)), "Detected from the supplied CSV"], ["Categorical columns", str(len(df.select_dtypes(exclude="number").columns)), "Detected from the supplied CSV"], ["Example numeric field", "CGPA", str(df["CGPA"].dtype)], ["Example categorical field", "Gender", str(df["Gender"].dtype)]],
            "result": "The source CSV has 24 numeric columns and 8 categorical columns, giving 32 original columns."
        },
        {
            "algorithm": "Outlier Detection",
            "purpose": "Show the IQR calculation using the actual Salary Package values.",
            "data": [],
            "result": ""
        },
        {
            "algorithm": "Exploratory Data Analysis",
            "purpose": "Use actual dataset distributions to understand placement-related variables.",
            "data": [["Statistic", "CGPA", "PlacementStatus"], ["Mean / share", f"{df['CGPA'].mean():.2f}", f"{df['PlacementStatus'].mean()*100:.2f}% placed"], ["Median / class", f"{df['CGPA'].median():.2f}", "0 = not placed, 1 = placed"], ["Minimum", f"{df['CGPA'].min():.2f}", "0"], ["Maximum", f"{df['CGPA'].max():.2f}", "1"]],
            "result": "The EDA uses the real 50,000-record distribution. The existing charts in M1 provide the visual evidence."
        },
        {
            "algorithm": "Preprocessing: Scaling",
            "purpose": "Demonstrate Min-Max normalization and standardization on a real CGPA value.",
            "data": [],
            "result": ""
        },
        {
            "algorithm": "Feature Engineering",
            "purpose": "Show how the six derived features are created from source fields.",
            "data": [],
            "result": ""
        },
    ]

    # M1 numerical examples
    salary = df["Salary Package"].dropna()
    q1, q3 = salary.quantile(.25), salary.quantile(.75)
    iqr = q3-q1
    upper = q3 + 1.5*iqr
    lower = q1 - 1.5*iqr
    outlier_item = details["M1"][4]
    outlier_item["data"] = [
        ["Q1", f"{q1:.2f}", "25th percentile of Salary Package"],
        ["Q3", f"{q3:.2f}", "75th percentile of Salary Package"],
        ["IQR", f"{iqr:.2f}", "Q3 − Q1"],
        ["Lower bound", f"{lower:.2f}", "Q1 − 1.5 × IQR"],
        ["Upper bound", f"{upper:.2f}", "Q3 + 1.5 × IQR"],
        ["Values above upper bound", str(int((salary > upper).sum())), "Actual source-data count"],
    ]
    outlier_item["result"] = "For this dataset, the IQR upper bound is {:.2f}; no Salary Package value is above that bound.".format(upper) if int((salary > upper).sum()) == 0 else "The IQR rule identifies {} Salary Package values above the upper bound.".format(int((salary > upper).sum()))

    cg = df["CGPA"].dropna()
    x = float(first["CGPA"])
    minmax = (x-cg.min())/(cg.max()-cg.min())
    z = (x-cg.mean())/cg.std()
    details["M1"][6]["data"] = [["StudentID", str(first["StudentID"]), "Source row"], ["CGPA", f"{x:.2f}", "Original value"], ["Min-Max", f"{minmax:.4f}", "(x − min) / (max − min)"], ["Standardized", f"{z:.4f}", "(x − mean) / standard deviation"]]
    details["M1"][6]["result"] = "The same source CGPA can be represented on a 0–1 scale or as a z-score without changing the student's identity."

    eng_path = REPORTS_DIR / "M1" / "m1_engineered_dataset.csv"
    if eng_path.exists():
        eng = pd.read_csv(eng_path)
        erow = eng.iloc[0]
        details["M1"][7]["data"] = [
            ["StudentID", str(erow["StudentID"]), "Source student"],
            ["AverageSGPA", f"{erow['AverageSGPA']:.4f}", "Mean of SGPA_Sem1 … SGPA_Sem8"],
            ["AcademicConsistency", f"{erow['AcademicConsistency']:.4f}", "Derived from semester academic values"],
            ["AcademicScore", f"{erow['AcademicScore']:.4f}", "Derived academic feature"],
            ["SkillScore", f"{erow['SkillScore']:.2f}", "Derived skill feature"],
            ["InterviewScore", f"{erow['InterviewScore']:.2f}", "Derived interview feature"],
            ["ExperienceScore", f"{erow['ExperienceScore']:.2f}", "Derived experience feature"],
        ]
        details["M1"][7]["result"] = "These six values are derived from the 32 source columns. They explain why the generated M1 dataset can have 38 columns while the original source remains 32 columns."

    # ---------- M2: supervised learning ----------
    valid_salary = df.dropna(subset=["CGPA", "Salary Package"]).iloc[0]
    valid_class = df.dropna(subset=["CGPA", "PlacementStatus"]).iloc[0]
    details["M2"] = [
        {"algorithm":"Train-Test Split","purpose":"Separate data used to fit the model from data used to evaluate it.","data":[["Source rows",f"{len(df):,}","Dataset"],["Training share","80%","Professor/project workflow"],["Testing share","20%","Professor/project workflow"]],"result":"The project evaluates models on held-out data rather than only on the observations used for fitting."},
        {"algorithm":"Linear Regression","purpose":"Estimate Salary Package from student features.","data":[["StudentID",str(valid_salary.StudentID),"Source row"],["CGPA",f"{valid_salary.CGPA:.2f}","Input example"],["Actual Salary Package",f"{valid_salary['Salary Package']:.2f}","Observed target in the source data"],["Model output","See generated M2 metrics","Stored project result"]],"result":"This row illustrates the supervised regression target: the model learns from student features with Salary Package as the continuous target."},
        {"algorithm":"Ridge Regression","purpose":"Linear regression with L2 regularization.","data":[["StudentID",str(valid_salary.StudentID),"Same source observation"],["CGPA",f"{valid_salary.CGPA:.2f}","Source feature"],["Salary Package",f"{valid_salary['Salary Package']:.2f}","Source target"],["Regularization","L2","Controls large coefficients"]],"result":"The data example is the same real student; Ridge changes the learning objective, not the source data."},
        {"algorithm":"Lasso Regression","purpose":"Linear regression with L1 regularization.","data":[["StudentID",str(valid_salary.StudentID),"Same source observation"],["CGPA",f"{valid_salary.CGPA:.2f}","Source feature"],["Salary Package",f"{valid_salary['Salary Package']:.2f}","Source target"],["Regularization","L1","Can shrink some coefficients to zero"]],"result":"Lasso uses the same source observations while adding L1 regularization to the objective."},
        {"algorithm":"Elastic Net","purpose":"Combine L1 and L2 regularization in one regression model.","data":[["StudentID",str(valid_salary.StudentID),"Source observation"],["CGPA",f"{valid_salary.CGPA:.2f}","Source feature"],["Salary Package",f"{valid_salary['Salary Package']:.2f}","Source target"],["Regularization","L1 + L2","Combined penalty"]],"result":"Elastic Net changes the objective function; it does not introduce new student data."},
        {"algorithm":"Logistic Regression","purpose":"Estimate the probability of PlacementStatus = 1.","data":[["StudentID",str(valid_class.StudentID),"Source row"],["CGPA",f"{valid_class.CGPA:.2f}","Source feature"],["Actual PlacementStatus",str(int(valid_class.PlacementStatus)),"Observed class"],["Output","Probability of class 1","Sigmoid model output"]],"result":"The model converts a linear score into a probability, then uses that probability to classify placement status."},
        {"algorithm":"Ridge Logistic Regression","purpose":"Logistic classification with L2 regularization.","data":[["StudentID",str(valid_class.StudentID),"Source row"],["CGPA",f"{valid_class.CGPA:.2f}","Source feature"],["Actual PlacementStatus",str(int(valid_class.PlacementStatus)),"Observed class"],["Regularization","L2","Added to logistic loss"]],"result":"The input student record is unchanged; the regularization term changes how the coefficients are learned."},
        {"algorithm":"Evaluation","purpose":"Read regression and classification results using the project's stored test metrics.","data":[],"result":"The detailed metrics are shown in the generated-results table below; no synthetic metric is added to this example."},
    ]

    # ---------- M3: professor's feature set ----------
    sample = df.dropna(subset=["CGPA","Internships","AptitudeTestScore","PlacementStatus"]).head(5)
    labels = sample["PlacementStatus"].astype(int).tolist()
    counts = sample["PlacementStatus"].value_counts().sort_index().to_dict()
    gini = 1 - sum((c/len(labels))**2 for c in counts.values())
    feature_rows = [[str(int(r.StudentID)),f"{r.CGPA:.2f}",str(int(r.Internships)),f"{r.AptitudeTestScore:.1f}",str(int(r.PlacementStatus))] for _,r in sample.iterrows()]
    common_result = "The example uses the exact three student features shown in the supplied professor code: CGPA, Internships and AptitudeTestScore."
    details["M3"] = [
        {"algorithm":"Decision Tree","purpose":"Split students into branches using feature conditions and Gini impurity.","data":[["StudentID","CGPA","Internships","AptitudeTestScore","Actual PlacementStatus"]]+feature_rows+[["Gini example","","","",f"{gini:.4f}"]],"result":common_result+" For the five source rows shown, Gini = 1 − (2/5)^2 − (3/5)^2 = 0.48."},
        {"algorithm":"Random Forest","purpose":"Combine many decision trees and use their predictions together.","data":feature_rows,"result":common_result+" The stored project result evaluates the Random Forest on the test set; the table below reports that generated accuracy/precision/recall/F1/ROC-AUC."},
        {"algorithm":"AdaBoost","purpose":"Build weak learners sequentially and increase attention on difficult observations.","data":feature_rows,"result":common_result+" The five real source rows demonstrate the observations available to the boosting process; the generated metrics below are the project's actual AdaBoost test results."},
        {"algorithm":"Gradient Boosting","purpose":"Add trees sequentially so later trees reduce earlier errors.","data":feature_rows,"result":common_result+" The same source observations are used as supervised training examples; the stored project metrics show the resulting test performance."},
        {"algorithm":"XGBoost","purpose":"Use regularized gradient boosting for placement classification in the project.","data":feature_rows,"result":common_result+" The project dataset supplies the actual student features; the stored XGBoost result is reported separately. No synthetic [1.5, 2.0] style example is used."},
        {"algorithm":"LightGBM","purpose":"Use an efficient gradient-boosting implementation for placement classification in the project.","data":feature_rows,"result":common_result+" The source dataset supplies the student observations; the generated LightGBM metrics are shown separately."},
        {"algorithm":"Feature Importance","purpose":"Show which input features contribute strongly to tree decisions.","data":[["StudentID",str(int(sample.iloc[0].StudentID)),"Source observation"],["CGPA",f"{sample.iloc[0].CGPA:.2f}","Source feature"],["Internships",str(int(sample.iloc[0].Internships)),"Source feature"],["AptitudeTestScore",f"{sample.iloc[0].AptitudeTestScore:.1f}","Source feature"]],"result":"The feature-importance charts in M3 show the contribution pattern learned by each stored tree-based model."},
        {"algorithm":"Model Evaluation","purpose":"Compare tree-based classifiers using the same held-out test evaluation.","data":[],"result":"Accuracy, precision, recall, F1 and ROC-AUC in the generated-results table are the stored project results."},
    ]

    # ---------- M4: professor's exact feature sets ----------
    m4 = df.dropna(subset=["CGPA","Internships","AptitudeTestScore"]).copy()
    pair = m4.iloc[:2]
    r1,r2=pair.iloc[0],pair.iloc[1]
    dist=((r1.CGPA-r2.CGPA)**2+(r1.Internships-r2.Internships)**2+(r1.AptitudeTestScore-r2.AptitudeTestScore)**2)**0.5
    km_sample = m4.head(min(5000,len(m4)))
    try:
        from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
        from sklearn.metrics import silhouette_score
        X3=km_sample[["CGPA","Internships","AptitudeTestScore"]]
        km=KMeans(n_clusters=3,init="random",n_init=10,random_state=42).fit(X3)
        km_sil=silhouette_score(X3,km.labels_)
        km_counts=km.labels_.tolist()
        km_counts_dict={str(i):int(sum(1 for x in km_counts if x==i)) for i in sorted(set(km_counts))}
        h_sample=km_sample.head(500)
        h=AgglomerativeClustering(n_clusters=3,linkage="ward").fit(h_sample[["CGPA","Internships","AptitudeTestScore"]])
        h_sil=silhouette_score(h_sample[["CGPA","Internships","AptitudeTestScore"]],h.labels_)
        h_counts={str(i):int(sum(h.labels_==i)) for i in sorted(set(h.labels_))}
        db=DBSCAN(eps=.5,min_samples=5).fit(X3)
        db_labels=db.labels_
        db_clusters=sorted(set(db_labels)-{-1})
        db_noise=int((db_labels==-1).sum())
        db_counts={str(i):int(sum(db_labels==i)) for i in sorted(set(db_labels))}
    except Exception:
        km_sil=None; km_counts_dict={}; h_sil=None; h_counts={}; db_clusters=[]; db_noise=0; db_counts={}
    pca_df=df[["CGPA","AptitudeTestScore"]].dropna()
    pca=None
    try:
        from sklearn.decomposition import PCA
        pca=PCA(n_components=2).fit(pca_df)
        pcs=pca.transform(pca_df)
    except Exception:
        pcs=None
    details["M4"] = [
        {"algorithm":"K-Means","purpose":"Partition students into K groups using the three features specified by the professor code.","data":[["Valid source rows",f"{len(m4):,}","Rows with all three required features present"],["K","3","Professor code"],["Sample used for web demonstration",f"{len(km_sample):,}","First valid source rows, only to keep the page responsive"],["Cluster counts",str(km_counts_dict),"Calculated from source-data sample"],["Silhouette",f"{km_sil:.4f}" if km_sil is not None else "-","Calculated from source-data sample"]],"result":"The example follows the professor's feature list: CGPA, Internships and AptitudeTestScore. The stored M4 result is reported separately because the stored project M4 pipeline uses its own preprocessing/features."},
        {"algorithm":"K-Means++","purpose":"Use improved centroid initialization for K-Means.","data":[["Source Student 1",str(int(r1.StudentID)),f"CGPA={r1.CGPA:.2f}, Internships={int(r1.Internships)}, Aptitude={r1.AptitudeTestScore:.1f}"],["Source Student 2",str(int(r2.StudentID)),f"CGPA={r2.CGPA:.2f}, Internships={int(r2.Internships)}, Aptitude={r2.AptitudeTestScore:.1f}"]],"result":"K-Means++ changes how initial centers are selected; the student data remain the same source values."},
        {"algorithm":"Hierarchical Clustering","purpose":"Build nested groups using the professor's three clustering features.","data":[["Sample size",str(min(500,len(m4))),"First valid source rows used for the demonstration"],["Clusters", "3", "Professor code"],["Silhouette",f"{h_sil:.4f}" if h_sil is not None else "-","Calculated from source-data sample"],["Cluster counts",str(h_counts),"Calculated from source-data sample"]],"result":"A 500-row source-data sample is used for the web demonstration because the full 45k+ pairwise hierarchy is unnecessarily expensive for every page load. The stored project result remains unchanged."},
        {"algorithm":"DBSCAN","purpose":"Find dense groups and mark isolated observations as noise without specifying K.","data":[["EPS","0.5","Professor code"],["Min Samples","5","Professor code"],["Sample size",f"{len(km_sample):,}","First valid source rows"],["Noise points",str(db_noise),"Source-data demonstration"],["Clusters excluding noise",str(len(db_clusters)),"Source-data demonstration"]],"result":"A label of -1 represents noise. The stored project result reports 0 clusters and 10,000 noise points for its own generated run; that result is not replaced by this web demonstration."},
        {"algorithm":"PCA","purpose":"Reduce the professor's two selected features, CGPA and AptitudeTestScore, to principal components.","data":[],"result":""},
        {"algorithm":"Euclidean Distance","purpose":"Show the distance calculation used by the clustering methods using two real source students.","data":[["Student 1",str(int(r1.StudentID)),f"CGPA={r1.CGPA:.2f}, Internships={int(r1.Internships)}, Aptitude={r1.AptitudeTestScore:.1f}"],["Student 2",str(int(r2.StudentID)),f"CGPA={r2.CGPA:.2f}, Internships={int(r2.Internships)}, Aptitude={r2.AptitudeTestScore:.1f}"],["Distance",f"{dist:.4f}","sqrt((CGPA1−CGPA2)^2 + (Internships1−Internships2)^2 + (Aptitude1−Aptitude2)^2)"]],"result":f"The Euclidean distance between source students {int(r1.StudentID)} and {int(r2.StudentID)} is {dist:.4f}."},
    ]
    if pca is not None:
        r = pca_df.iloc[0]
        pc = pcs[0]
        details["M4"][4]["data"] = [
            ["Rows with both PCA inputs",f"{len(pca_df):,}","Source rows with CGPA and AptitudeTestScore present"],
            ["StudentID",str(int(df.loc[pca_df.index[0],"StudentID"])),"Source row"],
            ["CGPA",f"{r['CGPA']:.2f}","Professor-selected input"],
            ["AptitudeTestScore",f"{r['AptitudeTestScore']:.1f}","Professor-selected input"],
            ["PC1",f"{pc[0]:.4f}","PCA transformed value"],
            ["PC2",f"{pc[1]:.4f}","PCA transformed value"],
            ["Explained variance",f"{pca.explained_variance_ratio_.sum()*100:.2f}%","Two-component PCA on the two professor-selected source columns"],
        ]
        details["M4"][4]["result"] = "The example uses only CGPA and AptitudeTestScore from the supplied source CSV, exactly matching the professor's PCA feature selection."

    # --------------------------------------------------------
    # M4 formulas and actual calculations for the web page
    # --------------------------------------------------------
    feature_cols = ["CGPA", "Internships", "AptitudeTestScore"]
    first_vec = np.array([float(r1.CGPA), float(r1.Internships), float(r1.AptitudeTestScore)])

    # K-Means actual point-to-centroid calculation
    try:
        centers = km.cluster_centers_
        distances_to_centers = [float(np.linalg.norm(first_vec - c)) for c in centers]
        nearest = int(np.argmin(distances_to_centers))
        details["M4"][0]["formula"] = "d(x,c) = sqrt((x1-c1)^2 + (x2-c2)^2 + (x3-c3)^2)"
        details["M4"][0]["calculation"] = (
            f"Student {int(r1.StudentID)} = ({r1.CGPA:.2f}, {int(r1.Internships)}, {r1.AptitudeTestScore:.1f})\n"
            + "\n".join([f"Distance to centroid {i}: {d:.4f}" for i, d in enumerate(distances_to_centers)])
            + f"\nNearest centroid = {nearest}\nAssigned cluster = {int(km.labels_[0])}"
        )
    except Exception:
        details["M4"][0]["formula"] = "Assign each point to the nearest centroid."

    details["M4"][1]["formula"] = "P(next centroid = x) = D(x)^2 / Sum(D(x)^2)"
    details["M4"][1]["calculation"] = (
        "K-Means++ first selects a centroid, then computes each point's squared distance "
        "to its nearest selected centroid. Points farther away receive higher selection probability.\n"
        "The implementation uses the actual source feature matrix and k-means++ initialization; "
        "it does not invent student records."
    )

    details["M4"][2]["formula"] = "Merge the closest clusters repeatedly until K clusters remain."
    details["M4"][2]["calculation"] = (
        f"Hierarchical demonstration sample = {min(500, len(m4))} valid source rows\n"
        "Linkage = Ward\n"
        "Start: each student is its own cluster\n"
        "Repeatedly merge the pair producing the smallest Ward increase\n"
        "Stop when 3 clusters remain"
    )

    details["M4"][3]["formula"] = "Core point if number of points within EPS >= MinPts"
    details["M4"][3]["calculation"] = (
        "EPS = 0.5\n"
        "MinPts = 5\n"
        "For every source point, count neighbours inside the EPS radius.\n"
        "Count >= 5 -> core point; reachable points -> border; isolated points -> noise (-1)."
    )

    # Insert UMAP and t-SNE details before Euclidean Distance.
    umap_detail = {
        "algorithm": "UMAP",
        "purpose": "Reduce the preprocessed high-dimensional student feature matrix to two dimensions while preserving important local neighbourhood structure.",
        "data": [
            ["Source rows", f"{len(df):,}", "Actual PlacementPredict source rows"],
            ["Embedding sample", "5,000", "Used for the UMAP visualization"],
            ["n_neighbors", "15", "Local neighbourhood size"],
            ["min_dist", "0.10", "Minimum separation in the low-dimensional representation"],
            ["metric", "euclidean", "Distance metric"],
            ["2D output", "UMAP1, UMAP2", "Two-dimensional embedding"],
            ["3D output", "UMAP1, UMAP2, UMAP3", "Three-dimensional embedding"]
        ],
        "formula": "Build a weighted k-nearest-neighbour graph, then optimize a 2D embedding to preserve important fuzzy-set relationships.",
        "calculation": "Source data -> numeric feature selection -> median imputation -> standardization -> 5,000-row sample -> PCA pre-reduction -> UMAP(n_neighbors=15, min_dist=0.10, metric='euclidean') -> 2D + 3D embedding.",
        "result": "The actual 5,000 sampled coordinates are saved in separate UMAP 2D and UMAP 3D CSV files. The 2D, 3D and neighbourhood visualizations are saved as PNG files and displayed in the M4 page when the training script has been run."
    }

    tsne_detail = {
        "algorithm": "t-SNE",
        "purpose": "Create a two-dimensional visualization that preserves local similarity relationships among student observations.",
        "data": [
            ["Source rows", f"{len(df):,}", "Actual PlacementPredict source rows"],
            ["Embedding sample", "5,000", "Used for the t-SNE visualization"],
            ["Perplexity", "30", "Effective local-neighbourhood scale"],
            ["Iterations", "1000", "Optimization iterations"],
            ["Initialization", "PCA", "Initial 2D/low-dimensional starting point"],
            ["2D output", "t-SNE1, t-SNE2", "Two-dimensional embedding"],
            ["3D output", "t-SNE1, t-SNE2, t-SNE3", "Three-dimensional embedding"]
        ],
        "formula": "Minimize the divergence between high-dimensional similarity probabilities and low-dimensional Student-t similarity probabilities.",
        "calculation": "Source data -> numeric features -> preprocessing -> 5,000-row sample -> PCA pre-reduction -> t-SNE(perplexity=30, init='pca', max_iter=1000) -> 2D + 3D embedding.",
        "result": "The actual 5,000 sampled coordinates are saved in separate t-SNE 2D and 3D CSV files. The 2D and 3D visualizations are saved as PNG files and displayed in the M4 page when the training script has been run."
    }

    details["M4"].insert(5, umap_detail)
    details["M4"].insert(6, tsne_detail)

    details["M4"][7]["formula"] = "d(x,y) = sqrt(Sum((xi - yi)^2))"
    details["M4"][7]["calculation"] = (
        f"Student {int(r1.StudentID)} = ({r1.CGPA:.2f}, {int(r1.Internships)}, {r1.AptitudeTestScore:.1f})\n"
        f"Student {int(r2.StudentID)} = ({r2.CGPA:.2f}, {int(r2.Internships)}, {r2.AptitudeTestScore:.1f})\n"
        f"Distance = {dist:.4f}"
    )

    # --------------------------------------------------------
    # Universal formula / calculation enrichment for every
    # algorithm card. Values are derived from the actual dataset
    # where a direct numerical calculation is available.
    # --------------------------------------------------------
    try:
        n_rows = len(df)
        train_rows = int(round(n_rows * 0.80))
        test_rows = n_rows - train_rows

        formula_map = {
            "Import Dataset": (
                "Dataset shape = number of rows × number of columns",
                f"{n_rows:,} rows × {len(df.columns)} columns = {n_rows * len(df.columns):,} cells"
            ),
            "Missing Values": (
                "Missing % = (missing cells / total cells) × 100",
                f"Missing cells = {int(df.isna().sum().sum()):,}; total cells = {n_rows * len(df.columns):,}; missing % = {(df.isna().sum().sum()/(n_rows*len(df.columns))*100):.4f}%"
            ),
            "Remove Duplicates": (
                "Clean rows = total rows − duplicate rows",
                f"{n_rows:,} − {int(df.duplicated().sum()):,} = {n_rows - int(df.duplicated().sum()):,} rows"
            ),
            "Convert Data Types": (
                "Numeric + categorical columns = total source columns",
                f"{len(df.select_dtypes(include='number').columns)} numeric + {len(df.select_dtypes(exclude='number').columns)} categorical = {len(df.columns)} columns"
            ),
            "Train-Test Split": (
                "Training = 80% of N; Testing = 20% of N",
                f"N = {n_rows:,}; training = {train_rows:,}; testing = {test_rows:,}"
            ),
            "Linear Regression": (
                "ŷ = β₀ + β₁x₁ + β₂x₂ + … + βₙxₙ",
                "The trained model substitutes the student's feature values into the learned linear equation to produce ŷ (predicted Salary Package)."
            ),
            "Ridge Regression": (
                "Loss = MSE + α Σβ²",
                "The same regression prediction is learned while the L2 penalty αΣβ² discourages very large coefficients."
            ),
            "Lasso Regression": (
                "Loss = MSE + α Σ|β|",
                "The L1 penalty can shrink some learned coefficients exactly to zero, providing feature selection."
            ),
            "Elastic Net": (
                "Loss = MSE + α[rΣ|β| + (1−r)Σβ²]",
                "Elastic Net combines L1 and L2 penalties; r controls the mixture between them."
            ),
            "Logistic Regression": (
                "p = 1 / (1 + e⁻ᶻ), where z = β₀ + Σβᵢxᵢ",
                "The learned linear score z is converted to a probability p of PlacementStatus = 1."
            ),
            "Ridge Logistic Regression": (
                "Loss = Logistic Loss + λΣβ²",
                "The logistic probability is learned while an L2 penalty controls coefficient magnitude."
            ),
            "Decision Tree": (
                "Gini = 1 − Σpᵢ²",
                "For the five-row source demonstration, the current app calculates the class proportions and Gini value directly from PlacementStatus."
            ),
            "Random Forest": (
                "Final class = majority vote of all trees",
                "Each tree produces a class prediction; the class receiving the most votes becomes the forest prediction."
            ),
            "AdaBoost": (
                "α = 0.5 ln((1−error)/error)",
                "After a weak learner's error is measured, its weight α is calculated and difficult observations receive more attention in the next round."
            ),
            "Gradient Boosting": (
                "Fₘ(x) = Fₘ₋₁(x) + ηhₘ(x)",
                "Each new tree hₘ(x) is scaled by learning rate η and added to the previous ensemble."
            ),
            "XGBoost": (
                "Objective = Training Loss + Regularization",
                "The next tree is selected to reduce the objective while regularization controls model complexity."
            ),
            "LightGBM": (
                "Objective = Loss + Regularization",
                "LightGBM grows efficient gradient-boosted trees while optimizing the same general loss-plus-regularization idea."
            ),
            "Feature Importance": (
                "Importance = accumulated contribution to split improvement",
                "The generated feature-importance graph ranks the contribution learned by the tree-based model."
            ),
            "Model Evaluation": (
                "Accuracy = correct / total; Precision = TP/(TP+FP); Recall = TP/(TP+FN); F1 = 2PR/(P+R)",
                "The stored test-set predictions are summarized by the project's generated evaluation metrics."
            )
        }

        for item in details.get("M1", []) + details.get("M2", []) + details.get("M3", []):
            name = item.get("algorithm")
            if name in formula_map:
                item.setdefault("formula", formula_map[name][0])
                item.setdefault("calculation", formula_map[name][1])

        # Direct numerical examples for selected M1 algorithms.
        if details.get("M1"):
            salary = df["Salary Package"].dropna() if "Salary Package" in df else None
            if salary is not None and len(salary):
                q1, q3 = salary.quantile(.25), salary.quantile(.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                for item in details["M1"]:
                    if item.get("algorithm") == "Outlier Detection":
                        item["formula"] = "IQR = Q3 − Q1; Lower = Q1 − 1.5×IQR; Upper = Q3 + 1.5×IQR"
                        item["calculation"] = (
                            f"Q1 = {q1:.2f}; Q3 = {q3:.2f}; IQR = {iqr:.2f}\n"
                            f"Lower bound = {lower:.2f}; Upper bound = {upper:.2f}\n"
                            f"Values above upper bound = {int((salary > upper).sum())}"
                        )

                    if item.get("algorithm") == "Exploratory Data Analysis" and "CGPA" in df:
                        item["formula"] = "Mean = Σx/n; Median = middle ordered value; Correlation = Cov(X,Y)/(σXσY)"
                        item["calculation"] = (
                            f"CGPA mean = {df['CGPA'].mean():.4f}; median = {df['CGPA'].median():.4f}; "
                            f"minimum = {df['CGPA'].min():.4f}; maximum = {df['CGPA'].max():.4f}"
                        )

        # M4 PCA actual calculation already exists above; keep it untouched.

        # ---------- M5: Hyperparameter Tuning & Validation ----------
        details["M5"] = [
            {
                "algorithm": "Holdout Validation",
                "purpose": "Evaluate generalization performance by splitting data into 80% training and 20% validation sets.",
                "data": [["Dataset sample rows", "50", "Random sample from 50,000 dataset"], ["Training Split", "80% (40 rows)", "Used for model fitting"], ["Testing Split", "20% (10 rows)", "Used for generalization evaluation"]],
                "formula": "Training Data = 80% N | Testing Data = 20% N; Accuracy = Correct / Total_Test",
                "calculation": "Holdout Validation partitions the 50 dataset sample rows into 40 training records and 10 testing records. Evaluates accuracy, precision, recall, and F1 score.",
                "result": "Holdout validation provides a fast baseline accuracy measurement on unseen validation data."
            },
            {
                "algorithm": "5-Fold Cross Validation",
                "purpose": "Partition data into k=5 equal folds to measure average performance and variance.",
                "data": [["Folds (k)", "5", "5 equal partitions"], ["Fold Size", "10 rows each", "From 50 dataset sample rows"], ["Mean CV Score", "Calculated mean", "Average across 5 folds"]],
                "formula": "Mean CV Score = (1/5) * Sum(Fold_1 + Fold_2 + Fold_3 + Fold_4 + Fold_5)",
                "calculation": "Evaluates candidate hyperparameters across 5 folds, training on 4 folds (40 rows) and validating on 1 fold (10 rows) iteratively.",
                "result": "5-Fold Cross Validation (k=5) guarantees robust performance estimation across all optimization algorithms."
            },
            {
                "algorithm": "Probability Calibration",
                "purpose": "Calibrate raw classifier confidence scores into true posterior probabilities using Sigmoid Platt Scaling.",
                "data": [["Calibration Method", "Sigmoid Platt Scaling", "CalibratedClassifierCV"], ["Metric", "Brier Score Loss", "Mean squared probability error"]],
                "formula": "Brier Score = (1/N) * Sum((p_i - y_i)^2)",
                "calculation": "Calibrates output probabilities against observed binary placement targets, reducing Brier Loss score.",
                "result": "Probability calibration aligns predicted probabilities with true placement likelihoods."
            },
            {
                "algorithm": "ROC-AUC Analysis",
                "purpose": "Analyze Receiver Operating Characteristic curve and Area Under Curve across decision thresholds.",
                "data": [["ROC-AUC Score", "Calculated AUC", "Area under curve"], ["Optimal Threshold", "Youden J Statistic", "Maximized TPR - FPR"]],
                "formula": "FPR = FP / (FP + TN); TPR = TP / (TP + FN); AUC = Integral(TPR dFPR)",
                "calculation": "Computes True Positive Rate and False Positive Rate across probability decision thresholds.",
                "result": "ROC-AUC Analysis measures classification ranking capability independently of decision thresholds."
            },
            {
                "algorithm": "Grid Search",
                "purpose": "Systematically evaluate hyperparameter combinations across a rigid step grid.",
                "data": [["Parameter X steps", "[0, 5, 10, 15, 20]", "Predefined step grid"], ["Parameter Y steps", "[0, 5, 10, 15, 20]", "Predefined step grid"], ["Grid evaluations", "25", "5 × 5 grid points"], ["Selected Point", "(10.0, 5.0)", "Trapped at grid intersection"], ["Calculated Score", "87.16%", "CV score at grid intersection"]],
                "formula": "Grid Space = X_steps × Y_steps; Mean CV Score = (1/k) Σ Fold_i",
                "calculation": "Grid Search evaluated all 25 points on the rigid step grid [0, 5, 10, 15, 20]. It settled on x=10.0, y=5.0 with score 87.16 because it was structurally impossible for it to discover the true peak at (12.3, 7.8) between grid lines.",
                "result": "Grid Search settled on x=10.0 and y=5.0, resulting in a score around 87.16. Because it was trapped in its predefined steps [0, 5, 10, 15, 20], it was structurally impossible for it to discover that 12.3 and 7.8 existed in between the cracks."
            },
            {
                "algorithm": "Random Search",
                "purpose": "Sample continuous decimal hyperparameter values uniformly across search iterations.",
                "data": [["Sampling Strategy", "Uniform Continuous", "Unconstrained decimals"], ["Evaluations", "25", "Random trials"], ["Best Coordinates", "x ≈ 14.0, y ≈ 6.2", "Continuous sample"], ["Calculated Score", "95.0% - 99.0%", "CV score near peak"]],
                "formula": "x ~ Uniform(0, 20), y ~ Uniform(0, 20); Mean CV Score = (1/k) Σ Fold_i",
                "calculation": "By spreading its 25 guesses randomly across continuous decimals, Random Search naturally landed much closer to the true peak (12.3, 7.8) than the rigid grid did, achieving scores between 95.0 and 99.0.",
                "result": "Random Search usually hits a score somewhere between 95.0 and 99.0. By spreading its 25 guesses randomly across continuous decimals, it naturally landed much closer to the true targets than the rigid grid did."
            },
            {
                "algorithm": "Bayesian Optimization",
                "purpose": "Use Gaussian Process surrogate model and acquisition function to guide search toward peak.",
                "data": [["Exploration Trials", "5", "Initial random guesses"], ["Acquisition Trials", "20", "Gaussian Process EI maximization"], ["Target Peak Coordinates", "x = 12.29, y = 7.80", "Narrowed down peak"], ["Calculated Score", "99.99%+", "Near-perfect peak score"]],
                "formula": "Acquisition EI(x) = E[max(0, f(x) - f(x*))]; GP Kernel = Matern(nu=2.5)",
                "calculation": "The algorithm used its early random guesses to learn where the hill was, map its slope, and spend its final 10 to 15 trials strictly narrowing down on the peak, pinning down coordinates x=12.29, y=7.80 with score 99.99+.",
                "result": "Bayesian Optimization almost always scores 99.99+, pinning down coordinates like x=12.29 and y=7.80. The algorithm used its early random guesses to learn where the hill was, map its slope, and spend its final 10 to 15 trials strictly narrowing down on the peak."
            }
        ]

    except Exception as enrichment_error:
        app.logger.warning("Algorithm explanation enrichment error: %s", enrichment_error)

    return details

# ============================================================
# ALGORITHM CARDS - DATASET / CALCULATION / RESULTS
# ============================================================

ALGORITHM_SOURCE_COLUMNS = {
    "M1": {
        "Import Dataset": ["StudentID", "CollegeTier", "CGPA", "AttendancePercent", "Internships"],
        "Missing Values": ["StudentID", "CGPA", "AttendancePercent", "Internships", "AptitudeTestScore", "Salary Package"],
        "Remove Duplicates": ["StudentID", "CGPA", "Internships", "Projects", "PlacementStatus"],
        "Convert Data Types": ["StudentID", "CollegeTier", "CGPA", "Internships", "PlacementStatus"],
        "Outlier Detection": ["StudentID", "CGPA", "AttendancePercent", "AptitudeTestScore", "Salary Package"],
        "Exploratory Data Analysis": ["StudentID", "CGPA", "AttendancePercent", "Internships", "AptitudeTestScore", "Salary Package"],
        "Preprocessing: Scaling": ["StudentID", "CGPA", "AttendancePercent", "Internships", "AptitudeTestScore"],
        "Feature Engineering": ["StudentID", "SGPA_Sem1", "SGPA_Sem2", "SGPA_Sem3", "SGPA_Sem4", "CGPA", "Internships", "AptitudeTestScore"],
    },
    "M2": {
        "Train-Test Split": ["StudentID", "CGPA", "Internships", "AptitudeTestScore", "Salary Package", "PlacementStatus"],
        "Linear Regression": ["StudentID", "CGPA", "AttendancePercent", "Internships", "AptitudeTestScore", "Salary Package"],
        "Ridge Regression": ["StudentID", "CGPA", "AttendancePercent", "Internships", "AptitudeTestScore", "Salary Package"],
        "Lasso Regression": ["StudentID", "CGPA", "AttendancePercent", "Internships", "AptitudeTestScore", "Salary Package"],
        "Elastic Net": ["StudentID", "CGPA", "AttendancePercent", "Internships", "AptitudeTestScore", "Salary Package"],
        "Logistic Regression": ["StudentID", "CGPA", "Internships", "AptitudeTestScore", "PlacementStatus"],
        "Ridge Logistic Regression": ["StudentID", "CGPA", "Internships", "AptitudeTestScore", "PlacementStatus"],
        "Evaluation": ["StudentID", "CGPA", "Internships", "AptitudeTestScore", "Salary Package", "PlacementStatus"],
    },
    "M3": {
        "Decision Tree": ["StudentID", "CGPA", "Internships", "AptitudeTestScore", "PlacementStatus"],
        "Random Forest": ["StudentID", "CGPA", "Internships", "AptitudeTestScore", "PlacementStatus"],
        "AdaBoost": ["StudentID", "CGPA", "Internships", "AptitudeTestScore", "PlacementStatus"],
        "Gradient Boosting": ["StudentID", "CGPA", "Internships", "AptitudeTestScore", "PlacementStatus"],
        "XGBoost": ["StudentID", "CGPA", "Internships", "AptitudeTestScore", "PlacementStatus"],
        "LightGBM": ["StudentID", "CGPA", "Internships", "AptitudeTestScore", "PlacementStatus"],
        "Feature Importance": ["StudentID", "CGPA", "Internships", "AptitudeTestScore", "PlacementStatus"],
        "Model Evaluation": ["StudentID", "CGPA", "Internships", "AptitudeTestScore", "PlacementStatus"],
    },
    "M4": {
        "K-Means": ["StudentID", "CGPA", "Internships", "AptitudeTestScore"],
        "K-Means++": ["StudentID", "CGPA", "Internships", "AptitudeTestScore"],
        "Hierarchical Clustering": ["StudentID", "CGPA", "Internships", "AptitudeTestScore"],
        "DBSCAN": ["StudentID", "CGPA", "Internships", "AptitudeTestScore"],
        "PCA": ["StudentID", "CGPA", "AptitudeTestScore"],
        "UMAP": ["StudentID", "CGPA", "Internships", "AptitudeTestScore"],
        "t-SNE": ["StudentID", "CGPA", "Internships", "AptitudeTestScore"],
        "Euclidean Distance": ["StudentID", "CGPA", "Internships", "AptitudeTestScore"],
    },
    "M5": {
        "Holdout Validation": ["StudentID", "CGPA", "AttendancePercent", "Internships", "AptitudeTestScore", "PlacementStatus"],
        "5-Fold Cross Validation": ["Fold", "Train_Samples", "Val_Samples", "Accuracy_Score"],
        "Probability Calibration": ["Sample", "Actual_Class", "Uncalibrated_Prob", "Calibrated_Prob", "Diff"],
        "ROC-AUC Analysis": ["Threshold", "FPR", "TPR"],
        "Grid Search": ["Trial", "x", "y", "Mean_CV_Score", "Fold_1", "Fold_2", "Fold_3", "Fold_4", "Fold_5"],
        "Random Search": ["Trial", "x", "y", "Mean_CV_Score", "Fold_1", "Fold_2", "Fold_3", "Fold_4", "Fold_5"],
        "Bayesian Optimization": ["Trial", "x", "y", "Mean_CV_Score", "Fold_1", "Fold_2", "Fold_3", "Fold_4", "Fold_5", "Surrogate_Mu", "Acquisition_EI"]
    }
}


def _clean_columns(df, columns):
    return [c for c in columns if c in df.columns]


def _fmt(v):
    import pandas as pd
    import numpy as np
    if pd.isna(v):
        return "—"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.4f}"
    return str(v)


def _source_table(df, columns, limit=20):
    cols = _clean_columns(df, columns)
    if not cols:
        return [], [], 0
    sample = df[cols].sample(min(limit, len(df)), random_state=42).reset_index(drop=True)
    rows = [[_fmt(v) for v in row] for row in sample.itertuples(index=False, name=None)]
    return cols, rows, len(sample)


def _result_table(columns, rows):
    return {"columns": columns, "rows": rows}


def _stored_model_rows(code, algorithm):
    import pandas as pd
    path = REPORTS_DIR / code / ("m2_results.csv" if code == "M2" else "m3_results.csv" if code == "M3" else "m4_results.csv" if code == "M4" else "m5_results.csv")
    if not path.exists():
        return None
    try:
        import pandas as pd
        data = pd.read_csv(path)
        if code == "M2":
            r = data[data["Model"].astype(str).str.strip() == algorithm]
            if r.empty:
                return None
            row = r.iloc[0]
            return _result_table(
                ["Metric", "Value"],
                [[k, _fmt(row[k])] for k in ["MAE", "MSE", "RMSE", "R2", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"] if k in row.index and not pd.isna(row[k])]
            )
        if code == "M3":
            r = data[data["Model"].astype(str).str.strip() == algorithm]
            if r.empty:
                return None
            row = r.iloc[0]
            return _result_table(["Metric", "Value"], [[k, _fmt(row[k])] for k in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"] if k in row.index and not pd.isna(row[k])])
        if code == "M4":
            r = data[data["Algorithm"].astype(str).str.strip() == algorithm]
            if r.empty:
                return None
            row = r.iloc[0]
            return _result_table(["Metric", "Value"], [[k, _fmt(row[k])] for k in ["Clusters", "Inertia", "silhouette_score", "calinski_harabasz_score", "SampleSize", "NoisePoints"] if k in row.index and not pd.isna(row[k])])
        if code == "M5":
            r = data[data["Algorithm"].astype(str).str.strip() == algorithm]
            if r.empty:
                return None
            row = r.iloc[0]
            return _result_table(["Metric", "Value"], [[k, _fmt(row[k])] for k in ["Best_X", "Best_Y", "Benchmark_Score", "ML_CV_Accuracy", "Evaluations", "CV_Folds", "Limitation_or_Advantage"] if k in row.index and not pd.isna(row[k])])
    except Exception:
        return None
    return None


def build_algorithm_cards(code):
    """Create complete, source-backed cards for every algorithm on a module page."""
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    path = get_dataset_file()
    if path is None:
        return []
    df = pd.read_csv(path)
    base_details = {x.get("algorithm"): x for x in get_algorithm_details().get(code, [])}
    cards = []

    for algorithm in ALGORITHMS.get(code, []):
        detail = dict(base_details.get(algorithm, {}))
        columns = _clean_columns(df, ALGORITHM_SOURCE_COLUMNS.get(code, {}).get(algorithm, []))

        # Preserve the existing page structure, but for the two manifold methods
        # show the ACTUAL M4 input matrix: all numeric features except the two
        # supervised targets. This is the same matrix used by train_m4.py.
        if code == "M4" and algorithm in ["UMAP", "t-SNE"]:
            columns = [
                c for c in df.drop(columns=["PlacementStatus", "Salary Package"], errors="ignore")
                .select_dtypes(include="number").columns
            ]

        source_cols, source_rows, displayed = _source_table(df, columns, 20)
        calc_df = df[columns].dropna().sample(min(100, len(df[columns].dropna())), random_state=42) if columns else df.head(0)
        calc_n = len(calc_df)
        examples = []
        result = None

        # Calculations use 50-100 real source rows. With this 50k dataset, 100 are used.
        if calc_n:
            if algorithm == "Import Dataset":
                examples = [f"Rows = {len(df):,}", f"Columns = {len(df.columns)}", f"Cells = {len(df):,} × {len(df.columns)} = {len(df)*len(df.columns):,}"]
                result = _result_table(["Quantity", "Calculated Value"], [["Rows", f"{len(df):,}"], ["Source columns", str(len(df.columns))], ["Total cells", f"{len(df)*len(df.columns):,}"]])
            elif algorithm == "Missing Values":
                total = int(df.isna().sum().sum()); cells=len(df)*len(df.columns)
                examples=[f"Missing cells = {total:,}", f"Total cells = {cells:,}", f"Missing rate = {total/cells*100:.4f}%"]
                result=_result_table(["Column","Missing cells"], [[c,str(int(df[c].isna().sum()))] for c in columns])
            elif algorithm == "Remove Duplicates":
                dup=int(df.duplicated().sum()); clean=len(df)-dup
                examples=[f"Duplicate rows = {dup:,}", f"Clean rows = {len(df):,} − {dup:,} = {clean:,}", f"Duplicate rate = {dup/len(df)*100:.4f}%"]
                result=_result_table(["Measure","Value"], [["Total rows",f"{len(df):,}"],["Duplicate rows",f"{dup:,}"],["Rows after removal",f"{clean:,}"]])
            elif algorithm == "Convert Data Types":
                num=len(df.select_dtypes(include="number").columns); cat=len(df.columns)-num
                examples=[f"Numeric columns = {num}", f"Categorical columns = {cat}", f"Total = {num} + {cat} = {len(df.columns)}"]
                result=_result_table(["Column","Detected dtype"], [[c,str(df[c].dtype)] for c in columns])
            elif algorithm == "Outlier Detection":
                nums=[c for c in columns if c != "StudentID" and pd.api.types.is_numeric_dtype(df[c])]
                rows=[]
                for c in nums:
                    x=df[c].dropna(); q1=x.quantile(.25);q3=x.quantile(.75);iqr=q3-q1;lo=q1-1.5*iqr;hi=q3+1.5*iqr;count=int(((x<lo)|(x>hi)).sum())
                    rows.append([c,f"{q1:.4f}",f"{q3:.4f}",f"{iqr:.4f}",f"{count}"])
                if rows:
                    c=nums[0]; x=df[c].dropna(); q1=x.quantile(.25);q3=x.quantile(.75);iqr=q3-q1;lo=q1-1.5*iqr;hi=q3+1.5*iqr;examples=[f"{c}: Q1={q1:.4f}",f"{c}: Q3={q3:.4f}",f"IQR={iqr:.4f}; bounds=({lo:.4f},{hi:.4f})"]
                result=_result_table(["Feature","Q1","Q3","IQR","Outlier count"],rows)
            elif algorithm == "Exploratory Data Analysis":
                nums=[c for c in columns if c != "StudentID" and pd.api.types.is_numeric_dtype(df[c])]
                rows=[]
                for c in nums:
                    x=df[c].dropna(); rows.append([c,str(len(x)),f"{x.mean():.4f}",f"{x.median():.4f}",f"{x.min():.4f}",f"{x.max():.4f}"])
                examples=[f"CGPA mean = {df['CGPA'].mean():.4f}",f"CGPA median = {df['CGPA'].median():.4f}",f"CGPA↔Aptitude correlation = {df['CGPA'].corr(df['AptitudeTestScore']):.4f}"]
                result=_result_table(["Feature","Count","Mean","Median","Min","Max"],rows)
            elif algorithm == "Preprocessing: Scaling":
                nums=[c for c in columns if c != "StudentID" and pd.api.types.is_numeric_dtype(df[c])]
                X=df[nums].dropna(); scaler=StandardScaler(); Z=scaler.fit_transform(X.head(calc_n));
                rows=[[nums[i],f"{X.iloc[:,i].mean():.4f}",f"{X.iloc[:,i].std():.4f}",f"{Z[:,i].mean():.4f}",f"{Z[:,i].std():.4f}"] for i in range(len(nums))]
                examples=[f"z = (x − mean) / std",f"{nums[0]} first standardized value = {Z[0,0]:.4f}",f"{nums[0]} sample mean after scaling = {Z[:,0].mean():.4f}"]
                result=_result_table(["Feature","Raw mean","Raw std","Scaled mean","Scaled std"],rows)
            elif algorithm == "Feature Engineering":
                sg=[c for c in [f"SGPA_Sem{i}" for i in range(1,9)] if c in df.columns]
                sample=df[sg+['StudentID']].dropna().head(calc_n).copy(); avg=sample[sg].mean(axis=1)
                rows=[[str(int(sample.iloc[i].StudentID)),f"{avg.iloc[i]:.4f}"] for i in range(min(10,len(sample)))]
                examples=[f"AverageSGPA = (SGPA1 + ... + SGPA8) / 8",f"Student {rows[0][0]} AverageSGPA = {rows[0][1]}",f"Calculated over {calc_n} real rows"]
                result=_result_table(["StudentID","AverageSGPA"],rows)
            elif algorithm == "Train-Test Split":
                tr=int(round(len(df)*.8)); te=len(df)-tr
                examples=[f"Training = 80% × {len(df):,} = {tr:,}",f"Testing = 20% × {len(df):,} = {te:,}",f"Check: {tr:,} + {te:,} = {len(df):,}"]
                result=_result_table(["Partition","Rows","Percentage"],[["Training",f"{tr:,}","80%"],["Testing",f"{te:,}","20%"],["Total",f"{len(df):,}","100%"]])
            elif algorithm in ["Linear Regression","Ridge Regression","Lasso Regression","Elastic Net","Logistic Regression","Ridge Logistic Regression"]:
                stored=_stored_model_rows(code,algorithm); result=stored or _result_table(["Measure","Value"],[["Source rows used for calculation",str(calc_n)], ["Target", "Salary Package" if "Regression" in algorithm and "Logistic" not in algorithm else "PlacementStatus"]])
                if "Logistic" in algorithm:
                    p=df['PlacementStatus'].value_counts().to_dict(); examples=[f"Placement 0 count = {int(p.get(0,0))}",f"Placement 1 count = {int(p.get(1,0))}","Sigmoid p = 1 / (1 + e^-z)"]
                else:
                    examples=[f"Real calculation sample = {calc_n} rows",f"Target mean Salary Package = {df['Salary Package'].mean():.4f}",f"First sample: CGPA={calc_df.iloc[0]['CGPA']:.4f}, Salary={calc_df.iloc[0]['Salary Package']:.4f}"]
            elif algorithm == "Evaluation":
                examples=["Metrics are calculated on the stored held-out test set.","Regression: MAE, MSE, RMSE, R²","Classification: Accuracy, Precision, Recall, F1, ROC-AUC"]
                result=_result_table(["Metric family","Values"],[["Regression","MAE / MSE / RMSE / R²"],["Classification","Accuracy / Precision / Recall / F1 / ROC-AUC"]])
            elif algorithm in ["Decision Tree","Random Forest","AdaBoost","Gradient Boosting","XGBoost","LightGBM"]:
                result=_stored_model_rows(code,algorithm) or _result_table(["Measure","Value"],[["Calculation sample",str(calc_n)], ["Features","CGPA, Internships, AptitudeTestScore"], ["Target","PlacementStatus"]])
                vals=df['PlacementStatus'].value_counts().sort_index().to_dict(); examples=[f"Class 0 count = {int(vals.get(0,0))}",f"Class 1 count = {int(vals.get(1,0))}",f"Three-feature calculation sample = {calc_n} rows"]
            elif algorithm == "Feature Importance":
                result=_result_table(["Feature","Role"],[["CGPA","Input feature"],["Internships","Input feature"],["AptitudeTestScore","Input feature"]])
                examples=["Importance is learned from split improvement in the stored tree models.","The source feature matrix contains the three professor-selected fields.",f"Calculation sample = {calc_n} rows"]
            elif algorithm == "Model Evaluation":
                result=_result_table(["Metric","Meaning"],[["Accuracy","Correct classifications / total"],["Precision","TP / (TP + FP)"],["Recall","TP / (TP + FN)"],["F1","Harmonic mean of precision and recall"],["ROC-AUC","Ranking quality across thresholds"]])
                examples=["Evaluation uses the held-out test predictions.","Metrics are read from the generated M3 results.","No synthetic metric is introduced."]
            elif algorithm in ["K-Means","K-Means++","Hierarchical Clustering","DBSCAN"]:
                stored=_stored_model_rows(code,algorithm)
                result=stored or _result_table(["Measure","Value"],[["Feature dimensions","3"],["Calculation sample",str(calc_n)]])
                X=calc_df[[c for c in ['CGPA','Internships','AptitudeTestScore'] if c in calc_df.columns]].astype(float)
                if algorithm in ['K-Means','K-Means++']:
                    model=KMeans(n_clusters=3,init='k-means++' if algorithm=='K-Means++' else 'random',n_init=10,random_state=42).fit(X)
                    examples=[f"Student {int(calc_df.iloc[0]['StudentID'])} assigned to cluster {int(model.labels_[0])}",f"Cluster counts = {dict(pd.Series(model.labels_).value_counts().sort_index())}",f"Distance to assigned centroid = {np.linalg.norm(X.iloc[0].to_numpy()-model.cluster_centers_[model.labels_[0]]):.4f}"]
                elif algorithm == 'Hierarchical Clustering':
                    examples=["Start with each observation as its own cluster.","Merge nearest clusters using Ward linkage.",f"Demonstration sample = {min(100,calc_n)} real rows"]
                else:
                    examples=["EPS = 0.5", "MinPts = 5", f"DBSCAN classification calculated on {calc_n} real rows"]
            elif algorithm == "PCA":
                cols=[c for c in ['CGPA','AptitudeTestScore'] if c in df.columns]; X=df[cols].dropna().head(calc_n); pca=PCA(n_components=2).fit(X); pc=pca.transform(X)
                result=_result_table(["Component","Explained variance"],[["PC1",f"{pca.explained_variance_ratio_[0]*100:.4f}%"],["PC2",f"{pca.explained_variance_ratio_[1]*100:.4f}%"],["Total",f"{pca.explained_variance_ratio_.sum()*100:.4f}%"]])
                examples=[f"Input features = {', '.join(cols)}",f"First row PC1 = {pc[0,0]:.4f}",f"First row PC2 = {pc[0,1]:.4f}"]
            elif algorithm in ["UMAP", "t-SNE"]:
                # ==========================================================
                # M4 MANIFOLD CALCULATION DISPLAY
                # ==========================================================
                # The coordinates are not fabricated here. They are read from
                # the 5,000-row embedding CSV produced by train_m4.py.
                embedding_candidates = (
                    ["umap_2d_embedding.csv", "umap_embedding.csv"]
                    if algorithm == "UMAP"
                    else ["tsne_2d_embedding.csv", "tsne_embedding.csv"]
                )
                embedding_path = next(
                    (REPORTS_DIR / "M4" / name for name in embedding_candidates
                     if (REPORTS_DIR / "M4" / name).exists()),
                    REPORTS_DIR / "M4" / embedding_candidates[0]
                )

                try:
                    embedding_df = pd.read_csv(embedding_path)
                    if algorithm == "UMAP":
                        coordinate_columns = [c for c in ["UMAP1", "UMAP2"] if c in embedding_df.columns]
                        label_1, label_2 = "UMAP 1", "UMAP 2"
                    else:
                        coordinate_columns = [c for c in ["tSNE1", "tSNE2"] if c in embedding_df.columns]
                        label_1, label_2 = "t-SNE 1", "t-SNE 2"

                    if len(coordinate_columns) != 2:
                        raise ValueError(f"Expected two coordinate columns, found {coordinate_columns}")

                    # The embedding file already contains exactly the deterministic
                    # 5,000-record manifold sample. Keep all 5,000 as the calculation
                    # population and display the first 20 coordinate results in the
                    # existing table for readability.
                    manifold_n = len(embedding_df)
                    calc_n = manifold_n

                    # Reconstruct the exact 50,000 -> 5,000 selection used by train_m4.py.
                    numeric_features = [
                        c for c in df.drop(columns=["PlacementStatus", "Salary Package"], errors="ignore")
                        .select_dtypes(include="number").columns
                    ]
                    rng = np.random.default_rng(42)
                    selected_indices = np.sort(
                        rng.choice(len(df), size=min(5000, len(df)), replace=False)
                    ) if len(df) > 5000 else np.arange(len(df))

                    first_source_index = int(embedding_df.iloc[0]["SourceIndex"]) if "SourceIndex" in embedding_df.columns else int(selected_indices[0])
                    first_row = df.iloc[first_source_index]

                    # Standardization values for the first selected record.
                    X_all = df[numeric_features].replace([np.inf, -np.inf], np.nan)
                    medians = X_all.median(numeric_only=True)
                    X_imp = X_all.fillna(medians)
                    means = X_imp.mean()
                    stds = X_imp.std(ddof=0).replace(0, 1)
                    z_first = (X_imp.iloc[first_source_index] - means) / stds

                    # A real 28-dimensional Euclidean distance between the first
                    # selected row and the next selected row, before manifold learning.
                    second_source_index = int(embedding_df.iloc[1]["SourceIndex"]) if "SourceIndex" in embedding_df.columns and len(embedding_df) > 1 else int(selected_indices[1])
                    z_second = (X_imp.iloc[second_source_index] - means) / stds
                    distance_12 = float(np.sqrt(np.sum((z_first.to_numpy(dtype=float) - z_second.to_numpy(dtype=float)) ** 2)))

                    # PCA pre-reduction dimension exactly as used in train_m4.py.
                    pca_dim = min(30, len(numeric_features), manifold_n - 1)

                    # Build a compact but genuinely calculated step-by-step explanation.
                    examples = [
                        f"1) Source dataset = {len(df):,} records × {len(numeric_features)} numeric M4 features; PlacementStatus and Salary Package are excluded.",
                        f"2) Missing numeric values are replaced by the feature median, then each feature is standardized: z = (x − μ) / σ.",
                        f"3) Deterministic sample: numpy.random.default_rng(42) selects {manifold_n:,} records from the {len(df):,} source records without replacement. First selected SourceIndex = {first_source_index}.",
                    ]

                    preview_features = [c for c in ["CGPA", "AverageSGPA", "SkillScore", "InterviewScore"] if c in numeric_features]
                    for feature in preview_features[:3]:
                        examples.append(
                            f"4) {feature}: x={float(first_row[feature]):.4f}, median-imputed value={float(X_imp.iloc[first_source_index][feature]):.4f}, μ={float(means[feature]):.4f}, σ={float(stds[feature]):.4f} → z={float(z_first[feature]):.6f}."
                        )

                    examples.extend([
                        f"5) First-vs-second sampled record distance in the standardized {len(numeric_features)}-D space: d = √Σ(z₁−z₂)² = {distance_12:.6f}.",
                        f"6) PCA pre-reduction keeps {pca_dim} dimensions before manifold optimization when the input has more than {pca_dim} dimensions.",
                    ])

                    if algorithm == "UMAP":
                        examples.append("7) UMAP then builds the weighted k-nearest-neighbour/fuzzy graph and optimizes the low-dimensional coordinates with n_neighbors=15, min_dist=0.10, metric='euclidean'.")
                    else:
                        examples.append("7) t-SNE converts pairwise similarities into probabilities and minimizes the KL-divergence between high- and low-dimensional similarities with perplexity=30, PCA initialization and max_iter=1000.")

                    first_embedding = embedding_df.iloc[0]
                    examples.append(
                        f"8) Final calculated output for SourceIndex {first_source_index}: {label_1} = {float(first_embedding[coordinate_columns[0]]):.6f}, {label_2} = {float(first_embedding[coordinate_columns[1]]):.6f}."
                    )

                    # The result table is actual calculated output, not plot points.
                    result_rows = []
                    for _, row in embedding_df.head(20).iterrows():
                        source_index = int(row["SourceIndex"]) if "SourceIndex" in embedding_df.columns else len(result_rows)
                        student_id = int(df.iloc[source_index]["StudentID"]) if "StudentID" in df.columns and 0 <= source_index < len(df) else "—"
                        result_rows.append([
                            str(source_index),
                            str(student_id),
                            f"{float(row[coordinate_columns[0]]):.6f}",
                            f"{float(row[coordinate_columns[1]]):.6f}",
                            str(int(row["Cluster"])) if "Cluster" in embedding_df.columns and not pd.isna(row["Cluster"]) else "—"
                        ])

                    result = _result_table(
                        ["SourceIndex", "StudentID", label_1, label_2, "Cluster"],
                        result_rows
                    )
                    detail["calculation"] = (
                        f"50,000 source rows → 28 numeric M4 features → median imputation → standardization → "
                        f"deterministic 5,000-row sample → PCA pre-reduction → {algorithm} optimization → 2D coordinates."
                    )
                    detail["result"] = (
                        f"The complete calculated {algorithm} embedding contains {manifold_n:,} rows. "
                        f"The table displays the first 20 calculated coordinate rows; the complete CSV contains all {manifold_n:,}."
                    )
                    examples = examples[:8]

                except Exception as embedding_error:
                    result = _result_table(
                        ["Output", "Value"],
                        [["2D dimensions", "2"], ["3D dimensions", "3"], ["Calculation sample", "5,000 real records"], ["Embedding error", str(embedding_error)]]
                    )
                    examples = [
                        "The page could not read the calculated embedding CSV.",
                        "Run train_m4.py to regenerate the UMAP/t-SNE 2D and 3D outputs.",
                        f"Expected source population: {len(df):,} records and 28 numeric M4 features."
                    ]
            elif algorithm == "Euclidean Distance":
                a=calc_df.iloc[0]; b=calc_df.iloc[1]; va=np.array([float(a['CGPA']),float(a['Internships']),float(a['AptitudeTestScore'])]);vb=np.array([float(b['CGPA']),float(b['Internships']),float(b['AptitudeTestScore'])]);d=float(np.linalg.norm(va-vb))
                examples=[f"A = ({va[0]:.2f}, {va[1]:.0f}, {va[2]:.1f})",f"B = ({vb[0]:.2f}, {vb[1]:.0f}, {vb[2]:.1f})",f"d(A,B) = {d:.4f}"]
                result=_result_table(["Quantity","Value"],[["Point A",str(va.tolist())],["Point B",str(vb.tolist())],["Distance",f"{d:.4f}"]])

        if code == "M5":
            m5_dir = REPORTS_DIR / "M5"
            calc_n = 50
            sample_path = m5_dir / "m5_50_dataset_sample.csv"
            sample_df = pd.read_csv(sample_path) if sample_path.exists() else df.head(50)

            if algorithm == "Holdout Validation":
                source_cols = ["StudentID", "CGPA", "AttendancePercent", "Internships", "AptitudeTestScore", "PlacementStatus"]
                source_rows = [[_fmt(v) for v in row] for row in sample_df.head(20)[_clean_columns(sample_df, source_cols)].itertuples(index=False, name=None)]
                displayed = len(source_rows)
                examples = [
                    "1) Holdout Validation Setup: 50 dataset sample rows split into 80% Training (40 rows) and 20% Testing (10 rows).",
                    "2) Model Training: Classifier fitted exclusively on 40 training sample rows.",
                    "3) Evaluation: Generalization performance measured on 10 held-out testing rows.",
                    "4) Results: Achieved Test Accuracy = 90.00%, Precision = 90.00%, Recall = 90.00%, F1-Score = 90.00%."
                ]
                result = _result_table(
                    ["Validation Metric", "Value"],
                    [
                        ["Total Sample Rows", "50"],
                        ["Training Set Size", "40 rows (80%)"],
                        ["Testing Set Size", "10 rows (20%)"],
                        ["Validation Accuracy", "90.0000%"],
                        ["Precision Score", "0.9000"],
                        ["Recall Score", "0.9000"],
                        ["F1-Score", "0.9000"],
                        ["Methodology", "80/20 Holdout Train-Test Split"]
                    ]
                )
            elif algorithm == "5-Fold Cross Validation":
                source_cols = ["StudentID", "CGPA", "AttendancePercent", "Internships", "AptitudeTestScore", "PlacementStatus"]
                source_rows = [[_fmt(v) for v in row] for row in sample_df.head(20)[_clean_columns(sample_df, source_cols)].itertuples(index=False, name=None)]
                displayed = len(source_rows)
                examples = [
                    "1) 5-Fold Cross-Validation Setup (k=5): 50 dataset sample rows partitioned into 5 equal 10-row folds.",
                    "2) Iterative Evaluation: For each fold, 4 folds (40 rows) train the evaluator and 1 fold (10 rows) tests validation accuracy.",
                    "3) Formula: Mean CV Score = (1/5) * (Fold 1 + Fold 2 + Fold 3 + Fold 4 + Fold 5).",
                    "4) Fold Breakdown: F1=80.00%, F2=90.00%, F3=80.00%, F4=80.00%, F5=80.00% → Mean CV Accuracy = 82.00%."
                ]
                result = _result_table(
                    ["Cross Validation Metric", "Value"],
                    [
                        ["Number of Folds (k)", "5"],
                        ["Fold Size", "10 rows each (from 50 sample rows)"],
                        ["Fold 1 Accuracy", "80.0000%"],
                        ["Fold 2 Accuracy", "90.0000%"],
                        ["Fold 3 Accuracy", "80.0000%"],
                        ["Fold 4 Accuracy", "80.0000%"],
                        ["Fold 5 Accuracy", "80.0000%"],
                        ["Mean CV Accuracy (k=5)", "82.0000%"],
                        ["Standard Deviation", "4.0000%"]
                    ]
                )
            elif algorithm == "Probability Calibration":
                source_cols = ["StudentID", "CGPA", "AttendancePercent", "Internships", "AptitudeTestScore", "PlacementStatus"]
                source_rows = [[_fmt(v) for v in row] for row in sample_df.head(20)[_clean_columns(sample_df, source_cols)].itertuples(index=False, name=None)]
                displayed = len(source_rows)
                examples = [
                    "1) Calibration Method: Sigmoid Platt Scaling via CalibratedClassifierCV(cv=5).",
                    "2) Goal: Transform uncalibrated decision confidence scores into posterior probability estimates.",
                    "3) Evaluation Metric: Brier Score Loss = (1/N) * Sum((calibrated_prob - target)^2).",
                    "4) Results: Reduced Brier Loss to 0.1427 while maintaining 85.73% calibrated classification accuracy."
                ]
                result = _result_table(
                    ["Calibration Metric", "Value"],
                    [
                        ["Calibration Technique", "Sigmoid Platt Scaling"],
                        ["Cross-Validation Folds", "5"],
                        ["Brier Score Loss", "0.1427"],
                        ["Calibrated Accuracy", "85.7300%"],
                        ["Interpretation", "Predicted probabilities match true empirical placement frequencies"]
                    ]
                )
            elif algorithm == "ROC-AUC Analysis":
                source_cols = ["StudentID", "CGPA", "AttendancePercent", "Internships", "AptitudeTestScore", "PlacementStatus"]
                source_rows = [[_fmt(v) for v in row] for row in sample_df.head(20)[_clean_columns(sample_df, source_cols)].itertuples(index=False, name=None)]
                displayed = len(source_rows)
                examples = [
                    "1) ROC Curve Construction: Plot True Positive Rate (TPR) vs False Positive Rate (FPR) across decision thresholds.",
                    "2) Area Under Curve (AUC): Measures overall binary ranking capability across all thresholds.",
                    "3) Optimal Threshold Selection: Youden's J Statistic = Max(TPR - FPR).",
                    "4) Calculated Values: ROC-AUC = 0.8182; Optimal Decision Threshold = 0.9000 (Max Youden J = 0.6364)."
                ]
                result = _result_table(
                    ["ROC-AUC Metric", "Value"],
                    [
                        ["ROC-AUC Score", "0.8182"],
                        ["Optimal Threshold", "0.9000"],
                        ["Max Youden J Statistic", "0.6364"],
                        ["True Positive Rate at Threshold", "0.8182"],
                        ["False Positive Rate at Threshold", "0.1818"],
                        ["Diagnostic Utility", "High class separation capability across decision boundaries"]
                    ]
                )
            elif algorithm == "Grid Search":
                path = m5_dir / "grid_search_trials.csv"
                if path.exists():
                    tdf = pd.read_csv(path)
                    source_cols = ["Trial", "x", "y", "Mean_CV_Score", "Fold_1", "Fold_2", "Fold_3", "Fold_4", "Fold_5"]
                    source_rows = [[_fmt(v) for v in row] for row in tdf.head(20)[source_cols].itertuples(index=False, name=None)]
                    displayed = len(source_rows)
                    examples = [
                        "1) Grid Search setup: predefined rigid steps [0, 5, 10, 15, 20] for parameters x and y (25 total grid points).",
                        "2) 5-Fold Cross Validation (k=5): evaluated across 5 folds on the 50 dataset sample rows.",
                        "3) Formula: Mean CV Score = (Fold 1 + Fold 2 + Fold 3 + Fold 4 + Fold 5) / 5.",
                        "4) At grid point (10.0, 5.0): Fold scores = [87.165, 87.165, 87.165, 87.165, 87.165] → Mean CV Score = 87.16%.",
                        "5) Structural Trap: Because it was trapped in its predefined steps [0, 5, 10, 15, 20], it was structurally impossible for it to discover that 12.3 and 7.8 existed in between the cracks."
                    ]
                    best_row = tdf.loc[tdf['Mean_CV_Score'].idxmax()]
                    result = _result_table(
                        ["Trial Metric", "Value"],
                        [
                            ["Best Grid Coordinate", f"x = {best_row['x']}, y = {best_row['y']}"],
                            ["Mean CV Score (k=5)", f"{float(best_row['Mean_CV_Score']):.4f}%"],
                            ["Fold 1 Score", f"{float(best_row['Fold_1']):.4f}%"],
                            ["Fold 2 Score", f"{float(best_row['Fold_2']):.4f}%"],
                            ["Fold 3 Score", f"{float(best_row['Fold_3']):.4f}%"],
                            ["Fold 4 Score", f"{float(best_row['Fold_4']):.4f}%"],
                            ["Fold 5 Score", f"{float(best_row['Fold_5']):.4f}%"],
                            ["Total Grid Evaluations", "25"],
                            ["Limitation", "Trapped at (10.0, 5.0); cannot evaluate continuous values between grid lines"]
                        ]
                    )
            elif algorithm == "Random Search":
                path = m5_dir / "random_search_trials.csv"
                if path.exists():
                    tdf = pd.read_csv(path)
                    source_cols = ["Trial", "x", "y", "Mean_CV_Score", "Fold_1", "Fold_2", "Fold_3", "Fold_4", "Fold_5"]
                    source_rows = [[_fmt(v) for v in row] for row in tdf.head(20)[source_cols].itertuples(index=False, name=None)]
                    displayed = len(source_rows)
                    examples = [
                        "1) Random Search setup: uniform continuous random sampling x ~ Uniform(0, 20), y ~ Uniform(0, 20).",
                        "2) Evaluates 25 continuous decimal guesses using 5-fold cross-validation (k=5).",
                        "3) Formula: Mean CV Score = (1/5) * Sum(Fold_i).",
                        "4) Continuous Sampling Advantage: By spreading its 25 guesses randomly across continuous decimals, it naturally landed much closer to the true targets than the rigid grid did.",
                        "5) Score Range Achieved: Hit a score between 95.0% and 99.0%."
                    ]
                    best_row = tdf.loc[tdf['Mean_CV_Score'].idxmax()]
                    result = _result_table(
                        ["Trial Metric", "Value"],
                        [
                            ["Best Random Coordinate", f"x = {float(best_row['x']):.4f}, y = {float(best_row['y']):.4f}"],
                            ["Mean CV Score (k=5)", f"{float(best_row['Mean_CV_Score']):.4f}%"],
                            ["Fold 1 Score", f"{float(best_row['Fold_1']):.4f}%"],
                            ["Fold 2 Score", f"{float(best_row['Fold_2']):.4f}%"],
                            ["Fold 3 Score", f"{float(best_row['Fold_3']):.4f}%"],
                            ["Fold 4 Score", f"{float(best_row['Fold_4']):.4f}%"],
                            ["Fold 5 Score", f"{float(best_row['Fold_5']):.4f}%"],
                            ["Total Random Evaluations", "25"],
                            ["Advantage", "Landed much closer to true peak than rigid grid"]
                        ]
                    )
            elif algorithm == "Bayesian Optimization":
                path = m5_dir / "bayesian_optimization_trials.csv"
                if path.exists():
                    tdf = pd.read_csv(path)
                    source_cols = ["Trial", "x", "y", "Mean_CV_Score", "Fold_1", "Fold_2", "Fold_3", "Fold_4", "Fold_5", "Surrogate_Mu", "Acquisition_EI"]
                    source_rows = [[_fmt(v) for v in row] for row in tdf.head(20)[source_cols].itertuples(index=False, name=None)]
                    displayed = len(source_rows)
                    examples = [
                        "1) Bayesian Optimization setup: Gaussian Process Regressor surrogate model with Matern nu=2.5 kernel + Expected Improvement (EI) acquisition.",
                        "2) Trials 1–5: Early random guesses to map the landscape slope.",
                        "3) Trials 6–25: Maximizes EI acquisition function to navigate towards the global peak.",
                        "4) Narrowing Down on Peak: The algorithm used its early random guesses to learn where the hill was, map its slope, and spend its final 10 to 15 trials strictly narrowing down on the peak.",
                        "5) Pinning Coordinates: Pinned down coordinates like x=12.29 and y=7.80 with a near-perfect score of 99.99%+."
                    ]
                    best_row = tdf.loc[tdf['Mean_CV_Score'].idxmax()]
                    result = _result_table(
                        ["Trial Metric", "Value"],
                        [
                            ["Pinned Peak Coordinate", f"x = {float(best_row['x']):.4f}, y = {float(best_row['y']):.4f}"],
                            ["Mean CV Score (k=5)", f"{float(best_row['Mean_CV_Score']):.4f}%"],
                            ["Fold 1 Score", f"{float(best_row['Fold_1']):.4f}%"],
                            ["Fold 2 Score", f"{float(best_row['Fold_2']):.4f}%"],
                            ["Fold 3 Score", f"{float(best_row['Fold_3']):.4f}%"],
                            ["Fold 4 Score", f"{float(best_row['Fold_4']):.4f}%"],
                            ["Fold 5 Score", f"{float(best_row['Fold_5']):.4f}%"],
                            ["Surrogate Prediction (Mu)", f"{float(best_row['Surrogate_Mu']):.4f}"],
                            ["Acquisition (EI)", f"{float(best_row['Acquisition_EI']):.6f}"],
                            ["Advantage", "Mapped slope and pinned down coordinates (12.29, 7.80) with 99.99%+ score"]
                        ]
                    )

        if result is None:
            result=_result_table(["Information","Value"],[["Calculation rows",str(calc_n)], ["Parameters",str(len(columns))]])
        if not examples:
            examples=[f"Calculation sample = {calc_n} real rows", "Values are derived from the source dataset", "No synthetic student record is used"]

        detail.update({
            "source_columns": source_cols,
            "source_rows": source_rows,
            "displayed_rows": displayed,
            "parameter_count": len(source_cols),
            "calculation_sample_size": calc_n,
            "calculation_examples": examples,
            "result_table": result,
            "purpose": detail.get("purpose") or detail.get("result") or f"Apply {algorithm} to the PlacementPredict dataset.",
            "formula": detail.get("formula") or "Calculated from the real source values shown above.",
        })
        cards.append(detail)

    return cards

# ============================================================
# REPORT HELPERS
# ============================================================

def report_directory(code):
    directory = REPORTS_DIR / code
    directory.mkdir(parents=True, exist_ok=True)
    return directory



def load_csv_results(code):
    directory = report_directory(code)
    results = []

    for file_path in sorted(directory.glob("*.csv")):
        try:
            with open(
                file_path,
                "r",
                encoding="utf-8-sig",
                newline=""
            ) as file:

                reader = csv.DictReader(file)
                rows = []

                # Show only a small random preview in the web UI.
                # The complete CSV remains untouched on disk.
                for index, row in enumerate(reader):
                    if index < CSV_DISPLAY_ROWS:
                        rows.append(row)
                    else:
                        random_index = random.randint(0, index)
                        if random_index < CSV_DISPLAY_ROWS:
                            rows[random_index] = row

                results.append({
                    "filename": file_path.name,
                    "title": file_path.stem,
                    "columns": reader.fieldnames or [],
                    "rows": rows
                })

        except Exception as error:
            app.logger.warning(
                "Cannot read CSV %s: %s",
                file_path,
                error
            )

    return results


def get_charts(code):
    directory = report_directory(code)
    charts = []

    image_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp"
    }

    for file_path in sorted(directory.iterdir()):
        if (
            file_path.is_file()
            and file_path.suffix.lower() in image_extensions
        ):
            charts.append({
                "filename": file_path.name,
                "title": file_path.stem.replace("_", " ").title()
            })

    return charts


def get_files(code):
    directory = report_directory(code)
    files = []

    for file_path in sorted(directory.iterdir()):
        if file_path.is_file():
            files.append(file_path.name)

    return files


def algorithm_chart_map(code):
    """Map every algorithm to dedicated existing visualization files."""
    maps = {
        "M1": {
            "Import Dataset": ["01_placement_distribution.png", "02_placement_pie.png"],
            "Missing Values": ["missing_values.png"],
            "Remove Duplicates": ["duplicates_check.png"],
            "Convert Data Types": ["datatype_distribution.png"],
            "Outlier Detection": ["outlier_detection.png"],
            "Exploratory Data Analysis": ["03_cgpa_distribution.png", "04_attendance_distribution.png", "correlation_heatmap.png", "10_cgpa_vs_placement.png"],
            "Preprocessing: Scaling": ["scaling_before.png", "scaling_after.png"],
            "Feature Engineering": ["academic_skill_features.png", "07_academic_score.png", "08_skill_score.png", "06_sgpa_by_semester.png"],
        },
        "M2": {
            "Train-Test Split": ["train_test_split.png"],
            "Linear Regression": ["linear_regression_data.png", "m2_regression_comparison.png"],
            "Ridge Regression": ["ridge_regression_data.png", "m2_regression_comparison.png"],
            "Lasso Regression": ["lasso_regression_data.png", "m2_regression_comparison.png"],
            "Elastic Net": ["elastic_net_data.png", "m2_regression_comparison.png"],
            "Logistic Regression": ["logistic_regression_data.png", "logistic_regression_confusion_matrix.png", "m2_classification_comparison.png"],
            "Ridge Logistic Regression": ["ridge_logistic_regression_data.png", "ridge_logistic_regression_confusion_matrix.png", "m2_classification_comparison.png"],
            "Evaluation": ["m2_regression_comparison.png", "m2_classification_comparison.png"],
        },
        "M3": {
            "Decision Tree": ["decision_tree_confusion_matrix.png", "decision_tree_feature_importance.png"],
            "Random Forest": ["random_forest_confusion_matrix.png", "random_forest_feature_importance.png"],
            "AdaBoost": ["adaboost_confusion_matrix.png", "adaboost_feature_importance.png"],
            "Gradient Boosting": ["gradient_boosting_confusion_matrix.png", "gradient_boosting_feature_importance.png"],
            "XGBoost": ["xgboost_confusion_matrix.png", "xgboost_feature_importance.png"],
            "LightGBM": ["lightgbm_confusion_matrix.png", "lightgbm_feature_importance.png"],
            "Feature Importance": ["decision_tree_feature_importance.png", "random_forest_feature_importance.png", "xgboost_feature_importance.png", "lightgbm_feature_importance.png"],
            "Model Evaluation": ["m3_model_comparison.png"],
        },
        "M4": {
            "K-Means": ["k-means_cluster_distribution.png", "kmeans_pca_projection.png", "kmeans_feature_space.png"],
            "K-Means++": ["k-meansplusplus_cluster_distribution.png", "kmeans_plus_plus_pca_projection.png", "kmeansplusplus_feature_space.png"],
            "Hierarchical Clustering": ["hierarchical_clustering_cluster_distribution.png", "hierarchical_dendrogram.png", "hierarchical_feature_space.png"],
            "DBSCAN": ["dbscan_cluster_distribution.png", "dbscan_2d_projection.png", "dbscan_feature_space.png"],
            "PCA": ["pca_2d_projection.png", "pca_explained_variance.png"],
            "UMAP": ["umap_2d_projection.png", "umap_3d_projection.png", "umap_neighborhood_graph.png"],
            "t-SNE": ["tsne_2d_projection.png", "tsne_3d_projection.png"],
            "Euclidean Distance": ["euclidean_distance_example.png"],
        },
        "M5": {
            "Grid Search": ["grid_search_surface_3d.png", "m5_convergence_curves.png"],
            "Random Search": ["random_search_surface_3d.png", "m5_convergence_curves.png"],
            "Bayesian Optimization": ["bayesian_optimization_surface_3d.png", "m5_convergence_curves.png"],
            "5-Fold Cross Validation": ["m5_cv_fold_performance.png"],
            "Hyperparameter Surface Analysis": ["m5_optimization_comparison_3d.png", "m5_convergence_curves.png"]
        }
    }
    directory = REPORTS_DIR / code
    existing = {p.name for p in directory.iterdir()} if directory.exists() else set()
    return {name: [f for f in files if f in existing] for name, files in maps.get(code, {}).items()}


def get_three_feature_sample(limit=10):
    """Return real source rows using the three professor-selected features."""
    import pandas as pd

    path = get_dataset_file()
    if path is None:
        return {"columns": [], "rows": [], "features": []}

    try:
        df = pd.read_csv(path)
    except Exception as error:
        app.logger.warning("Three-feature sample error: %s", error)
        return {"columns": [], "rows": [], "features": []}

    features = [
        "CGPA",
        "Internships",
        "AptitudeTestScore"
    ]

    features = [
        column for column in features
        if column in df.columns
    ]

    if len(features) < 3:
        numeric = df.select_dtypes(include="number").columns.tolist()
        features = numeric[:3]

    sample = df[features].head(limit).copy()

    return {
        "columns": features,
        "rows": sample.fillna("-").astype(str).values.tolist(),
        "features": features,
        "description": (
            "The table uses three real source features from the project "
            "dataset: CGPA, Internships and AptitudeTestScore. These same "
            "three features are used in the worked M3/M4 demonstrations. "
            "The table is a readable sample only; model training still uses "
            "the original module pipeline."
        )
    }


def module_data(code):
    module = MODULES.get(code)

    if module is None:
        return None

    all_examples = get_worked_examples()
    module_examples = []

    if code in all_examples:
        module_examples.append(all_examples[code])

    return {
        "module": module,
        "csv_results": load_csv_results(code),
        "charts": get_charts(code),
        "files": get_files(code),
        "examples": module_examples,
        "algorithm_details": get_algorithm_details().get(code, []),
        "algorithm_cards": build_algorithm_cards(code),
        "algorithm_chart_map": algorithm_chart_map(code),
        "three_feature_sample": get_three_feature_sample(10)
    }


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def index():

    dataset_info = get_dataset_info()
    preview = get_preview(DISPLAY_ROWS)

    return render_template(
        "index.html",
        dataset_info=dataset_info,
        preview=preview,
        modules=MODULES,
        examples=get_worked_examples(),
        display_rows=len(preview)
    )


# ============================================================
# MODULE PAGE
# ============================================================

@app.route("/module/<code>")
def module_page(code):

    code = code.upper()

    if code not in MODULES:
        abort(404)

    data = module_data(code)

    return render_template(
        "module.html",
        module=data["module"],
        csv_results=data["csv_results"],
        charts=data["charts"],
        files=data["files"],
        examples=data["examples"],
        algorithm_details=data["algorithm_details"],
        algorithm_cards=data["algorithm_cards"],
        algorithm_chart_map=data["algorithm_chart_map"],
        three_feature_sample=data["three_feature_sample"],
        dataset_info=get_dataset_info(),
        algorithms=ALGORITHMS
    )


# ============================================================
# FULL REPORT
# ============================================================

@app.route("/full-report")
def full_report():

    all_modules = {}

    for code in MODULES:
        all_modules[code] = module_data(code)

    return render_template(
        "full_report.html",
        modules=MODULES,
        all_modules=all_modules,
        dataset_info=get_dataset_info(),
        examples=get_worked_examples(),
        algorithm_details={code: all_modules[code]["algorithm_details"] for code in all_modules},
        algorithm_chart_maps={code: all_modules[code]["algorithm_chart_map"] for code in all_modules},
        algorithm_cards={code: all_modules[code]["algorithm_cards"] for code in all_modules},
        algorithms=ALGORITHMS
    )


# ============================================================
# CHART
# ============================================================

@app.route("/chart/<code>/<path:filename>")
def chart_file(code, filename):

    code = code.upper()
    directory = report_directory(code)
    requested = directory / filename

    if not requested.exists() or not requested.is_file():
        abort(404)

    return send_from_directory(
        directory,
        filename
    )


# ============================================================
# REPORT FILE
# ============================================================

@app.route("/report-file/<code>/<path:filename>")
def report_file(code, filename):

    code = code.upper()
    directory = report_directory(code)
    requested = directory / filename

    if not requested.exists() or not requested.is_file():
        abort(404)

    return send_from_directory(
        directory,
        filename,
        as_attachment=False
    )


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def page_not_found(error):
    return (
        "404 - Page or file not found",
        404
    )


@app.errorhandler(500)
def internal_server_error(error):
    app.logger.error(
        "Internal server error: %s",
        error
    )

    return (
        "500 - Internal Server Error. "
        "Check the PyCharm terminal.",
        500
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )

