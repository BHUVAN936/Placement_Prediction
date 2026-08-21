import pandas as pd
import numpy as np

print("\nDataset : ")

df = pd.DataFrame({
    "Age": [25, np.nan, 30, np.nan, 50, 35, 12, 11],
    "Salary": [50000, 60000, 54000, 55000, 225000, 250500, 3400, 2500],
    "City": ["Vjy", "Hyd", "Bgl", "Visk", np.nan, "Visk", "Visk", np.nan],
    "Date": ["2026-08-01", np.nan, "2026-08-03", np.nan, "2026-08-05", "2026-08-06", np.nan, "2026-08-08"],
    "Temp": [30, np.nan, 32, np.nan, 35, 34, np.nan, 31]

})

print(df)



print("\nMode : ")
df['Age'] = df['Age'].fillna(df['Age'].mode()[0])
print(df)

print("\nMean : ")
df['Age'] = df['Age'].fillna(df['Age'].mean())
print(df)

print("\nMedian : ")
df['Age'] = df['Age'].fillna(df['Age'].median())
print(df)


print("\nCity - Mode:")
df["City"] = df["City"].fillna(df["City"].mode()[0])
print(df)

print("\nCity - Mode:")
df["City"] = df["City"].fillna(df["City"].mode()[0])
print(df)

print("\nDate - Backward Fill : ")
df["Date"] = df["Date"].bfill()
print(df)

print("\nDate - Forward Fill:")
df["Date"] = df["Date"].ffill()
print(df)

print("\nTemp - Forward Fill:")
df["Temp"] = df["Temp"].ffill()
print(df)

