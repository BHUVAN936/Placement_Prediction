import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/placement_predict_50k Dataset (2).csv")

# Remove unwanted spaces from column names
df.columns = df.columns.str.strip()

# Categorical / nominal columns
nominal_cols = [
    "Gender",
    "City",
    "Stream",
    "Specialisation",
    "Hostel",
    "HistoryOfBacklogs"
]

# Target column
target_col = "PlacementStatus"

# Split dataset into training and testing data
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df[target_col]
)

# Create One-Hot Encoder
ohe = OneHotEncoder(
    drop="first",
    sparse_output=False,
    handle_unknown="ignore"
)

# Fit encoder on training data
train_ohe = ohe.fit_transform(train_df[nominal_cols])

# Transform test data using the same encoder
test_ohe = ohe.transform(test_df[nominal_cols])

# Get encoded column names
ohe_cols = ohe.get_feature_names_out(nominal_cols)

# Convert encoded data to DataFrames
# astype(int) changes 0.0/1.0 to 0/1
train_ohe_df = pd.DataFrame(
    train_ohe,
    columns=ohe_cols,
    index=train_df.index
).astype(int)

test_ohe_df = pd.DataFrame(
    test_ohe,
    columns=ohe_cols,
    index=test_df.index
).astype(int)

# Remove original categorical columns
train_df = train_df.drop(columns=nominal_cols)
test_df = test_df.drop(columns=nominal_cols)

# Add encoded columns
train_df = pd.concat(
    [train_df, train_ohe_df],
    axis=1
)

test_df = pd.concat(
    [test_df, test_ohe_df],
    axis=1
)

# Display results
print("\nOriginal Dataset Shape:")
print(df.shape)

print("\nTraining Dataset Shape:")
print(train_df.shape)

print("\nTesting Dataset Shape:")
print(test_df.shape)

print("\nEncoded Training Data:")
print(train_df.head())

print("\nEncoded Testing Data:")
print(test_df.head())

# Display only the One-Hot Encoded columns
print("\nOne-Hot Encoded Training Columns:")
print(train_ohe_df.head())