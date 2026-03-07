# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Load Dataset
df = pd.read_csv("../dataset/student_data.csv")
print("Dataset Shape:", df.shape)
print(df.head())
# Dataset Information
print("\nDataset Info")
print(df.info())
print("\nMissing Values")
print(df.isnull().sum())
# Data Cleaning
# Remove duplicates
df = df.drop_duplicates()
# Remove missing values
df = df.dropna()
# Encoding categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])
# Feature and Target
# Predicting final grade
X = df.drop("G3", axis=1)
y = df["G3"]
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)
# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Train Models
# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
# Decision Tree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
# Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
# Evaluation
results = []
def evaluate_model(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    results.append([name, mse, rmse, mae, r2])
    print("\nModel:", name)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2 Score:", r2)
evaluate_model(y_test, y_pred_lr, "Linear Regression")
evaluate_model(y_test, y_pred_dt, "Decision Tree")
evaluate_model(y_test, y_pred_rf, "Random Forest")
# Model Comparison Table
results_df = pd.DataFrame(
    results, columns=["Model", "MSE", "RMSE", "MAE", "R2 Score"])
print("\nModel Comparison Table")
print(results_df)
# Visualization
plt.figure(figsize=(6,4))
plt.bar(results_df["Model"], results_df["R2 Score"])
plt.title("Model Comparison (R2 Score)")
plt.xlabel("Model")
plt.ylabel("R2 Score")
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()