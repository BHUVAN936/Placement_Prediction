import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. Locate the file
possible_paths = [
    "data/placement_predict_50K Dataset (2).csv",
    "placement_predict_50K Dataset (2).csv",
    "placement_predict_50k Dataset.csv",
    "data/placement_predict_50k Dataset.csv",
]

file_path = None
for path in possible_paths:
    if os.path.exists(path):
        file_path = path
        break

if not file_path:
    raise FileNotFoundError("Could not find the dataset file!")

print(f"Loading file from: {file_path}")

# 2. Load dataset
try:
    df = pd.read_excel(file_path)
except Exception:
    try:
        df = pd.read_csv(file_path, encoding="latin1")
    except Exception:
        df = pd.read_csv(file_path)

df.columns = df.columns.astype(str).str.strip()

# 3. Exclude IDs and binary targets from features to scale
exclude_cols = ["StudentID", "PlacementStatus", "IsAnomaly"]

# Select feature columns to scale
scale_cols = [
    c
    for c in df.select_dtypes(include=["int64", "float64"]).columns
    if c not in exclude_cols
]

print("\nColumns selected for scaling:")
print(scale_cols)

# Convert integer feature columns to float64 to avoid LossySetitemError
df[scale_cols] = df[scale_cols].astype("float64")

# 4. Apply Min-Max Scaling
scaler = MinMaxScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

print("\nScaled Data Head (Sample Columns):")
print(df[["CGPA", "AttendancePercent", "Salary Package"]].head())

# 5. Save scaled data
output_filename = "placement_predict_50k_scaled.csv"
df.to_csv(output_filename, index=False)

print(f"\nScaled dataset successfully saved to: {output_filename}")