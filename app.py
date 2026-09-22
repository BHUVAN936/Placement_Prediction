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

DISPLAY_ROWS = 100
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
        "DBSCAN", "PCA", "Euclidean Distance"
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
    return details

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
        "algorithm_details": get_algorithm_details().get(code, [])
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

    print()
    print("PlacementPredict")
    print("http://127.0.0.1:5000")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )
