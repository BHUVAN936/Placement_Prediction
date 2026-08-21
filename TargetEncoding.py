import pandas as pd
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder

# Load dataset
df = pd.read_csv("data/placement_predict_50k Dataset (2).csv")

# Remove spaces from column names
df.columns = df.columns.str.strip()

# Target column
target_col = "PlacementStatus"

# Categorical columns for Target Encoding
categorical_cols = [
    "Gender",
    "City",
    "CollegeTier",
    "Stream",
    "Specialisation",
    "Hostel",
    "HistoryOfBacklogs",
    "CGPA_Tier"
]

# Separate input and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Target Encoder
encoder = TargetEncoder(
    cols=categorical_cols,
    handle_missing="value",
    handle_unknown="value"
)

# Fit ONLY on training data
X_train[categorical_cols] = encoder.fit_transform(
    X_train[categorical_cols],
    y_train
)

# Transform test data using the encoder learned from training data
X_test[categorical_cols] = encoder.transform(
    X_test[categorical_cols]
)

print("After Target Encoding:")
print(X_train.head())

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)