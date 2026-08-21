import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split


df = pd.read_csv("data/placement_predict_50k Dataset (2).csv")

df.columns = df.columns.str.strip()


ordinal_cols = ["CollegeTier"]

# Target column
target_col = "PlacementStatus"


train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df[target_col]
)




ordinal_encoder = OrdinalEncoder(
    categories=[[1, 2, 3]],
    handle_unknown="use_encoded_value",
    unknown_value=-1
)



train_ordinal = ordinal_encoder.fit_transform(
    train_df[ordinal_cols]
)



test_ordinal = ordinal_encoder.transform(
    test_df[ordinal_cols]
)


# --------------------------------------------------
# 7. Convert Encoded Data to DataFrames
# --------------------------------------------------

train_ordinal_df = pd.DataFrame(
    train_ordinal,
    columns=ordinal_cols,
    index=train_df.index
).astype(int)

test_ordinal_df = pd.DataFrame(
    test_ordinal,
    columns=ordinal_cols,
    index=test_df.index
).astype(int)


# --------------------------------------------------
# 8. Remove Original CollegeTier Column
# --------------------------------------------------

train_df = train_df.drop(
    columns=ordinal_cols
)

test_df = test_df.drop(
    columns=ordinal_cols
)


# --------------------------------------------------
# 9. Add Encoded CollegeTier Column
# --------------------------------------------------

train_df = pd.concat(
    [train_df, train_ordinal_df],
    axis=1
)

test_df = pd.concat(
    [test_df, test_ordinal_df],
    axis=1
)


# --------------------------------------------------
# 10. Display Dataset Shapes
# --------------------------------------------------

print("\nOriginal Dataset Shape:")
print(df.shape)

print("\nTraining Dataset Shape:")
print(train_df.shape)

print("\nTesting Dataset Shape:")
print(test_df.shape)


# --------------------------------------------------
# 11. Display Encoded Training Data
# --------------------------------------------------

print("\nOrdinal Encoded Training Data:")
print(train_df.head())


# --------------------------------------------------
# 12. Display Encoded Testing Data
# --------------------------------------------------

print("\nOrdinal Encoded Testing Data:")
print(test_df.head())


# --------------------------------------------------
# 13. Display Only CollegeTier Encoding
# --------------------------------------------------

print("\nOrdinal Encoded CollegeTier:")
print(train_ordinal_df.head(10))


# --------------------------------------------------
# 14. Display Encoding Mapping
# --------------------------------------------------

print("\nOrdinal Encoding Mapping:")
print("CollegeTier 1 -> 0")
print("CollegeTier 2 -> 1")
print("CollegeTier 3 -> 2")