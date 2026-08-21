# --------------------------------------------------
# SIMPLE LINEAR REGRESSION - PLACEMENT PREDICTION
# --------------------------------------------------

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# --------------------------------------------------
# 2. Load the CSV file
# --------------------------------------------------

df = pd.read_csv("data/placement_predict_50k Dataset (2).csv")

print("Columns in dataset:")
print(df.columns.tolist())


# --------------------------------------------------
# 3. Randomly select 100 records
# --------------------------------------------------

sample_df = df.sample(n=100, random_state=42)

print("\nSelected records:")
print(sample_df[['CGPA', 'Salary Package']].head())


# --------------------------------------------------
# 4. Select X and y
# --------------------------------------------------

# Independent variable
X = sample_df[['CGPA']]

# Dependent variable
y = sample_df['Salary Package']


# --------------------------------------------------
# 5. Split the data
# 80 records -> Training
# 20 records -> Testing
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------------------------
# 6. Create the Linear Regression model
# --------------------------------------------------

model = LinearRegression()


# --------------------------------------------------
# 7. Train the model
# --------------------------------------------------

model.fit(X_train, y_train)


# --------------------------------------------------
# 8. Get slope and intercept
# --------------------------------------------------

slope = model.coef_[0]
intercept = model.intercept_

print("\n--- Fitted Model ---")
print("Slope (b1):", round(slope, 2))
print("Intercept (b0):", round(intercept, 2))

print(
    f"Regression Equation: Salary Package = "
    f"{slope:.2f} × CGPA + {intercept:.2f}"
)


# --------------------------------------------------
# 9. Predict salary for test data
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 10. Evaluation metrics
# --------------------------------------------------

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Evaluation Metrics ---")
print("Mean Squared Error (MSE):", round(mse, 3))
print("Root Mean Squared Error (RMSE):", round(rmse, 3))
print("R² Score:", round(r2, 3))


# --------------------------------------------------
# 11. Display actual vs predicted values
# --------------------------------------------------

results = pd.DataFrame({
    'CGPA': X_test['CGPA'].values,
    'Actual Salary': y_test.values,
    'Predicted Salary': y_pred
})

print("\n--- Actual vs Predicted ---")
print(results.to_string(index=False))


# --------------------------------------------------
# 12. Plot regression line
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    X_train,
    y_train,
    label="Training Data"
)

plt.scatter(
    X_test,
    y_test,
    label="Testing Data"
)

# Regression line
x_line = np.linspace(X['CGPA'].min(), X['CGPA'].max(), 100)

x_line_df = pd.DataFrame({'CGPA': x_line})

y_line = model.predict(x_line_df)

plt.plot(
    x_line,
    y_line,
    label="Regression Line"
)

plt.xlabel("CGPA")
plt.ylabel("Salary Package (LPA)")
plt.title("Simple Linear Regression: CGPA vs Salary Package")
plt.legend()
plt.grid(True)

plt.show()