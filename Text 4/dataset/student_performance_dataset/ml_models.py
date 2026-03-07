# Import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# 1 Load Dataset
# -----------------------------
df = pd.read_csv("dataset/student_data.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# -----------------------------
# 2 Basic Info
# -----------------------------
print("\nDataset Info")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# -----------------------------
# 3 Handle Missing Values
# -----------------------------
df = df.dropna()

# -----------------------------
# 4 Encoding Categorical Columns
# -----------------------------
label_encoder = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = label_encoder.fit_transform(df[col])

print("\nAfter Encoding")
print(df.head())

# -----------------------------
# 5 Define Features and Target
# -----------------------------
X = df.drop("G3", axis=1)   # Final grade prediction
y = df["G3"]

# -----------------------------
# 6 Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Size:", X_train.shape)
print("Testing Size:", X_test.shape)

# -----------------------------
# 7 Linear Regression
# -----------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

# -----------------------------
# 8 Decision Tree
# -----------------------------
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

# -----------------------------
# 9 Random Forest
# -----------------------------
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

# -----------------------------
# 10 Evaluation Function
# -----------------------------
def evaluate_model(y_true, y_pred, model_name):

    print("\n", model_name)
    print("R2 Score:", r2_score(y_true, y_pred))
    print("MSE:", mean_squared_error(y_true, y_pred))
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error))