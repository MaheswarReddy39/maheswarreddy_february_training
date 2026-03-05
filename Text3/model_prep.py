import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Load scaled dataset (choose one scaling method)
df = pd.read_csv("scaled_standard.csv")
print("Dataset Shape:", df.shape)
# Train-Test Split
X = df.drop(columns=['SalePrice'])
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)
# Handle Skewed Features
# Check skewness
skewed_features = X_train.select_dtypes(include=['int64', 'float64']).skew().sort_values(ascending=False)
print("Top Skewed Features:")
print(skewed_features.head())
# Apply log transformation to highly skewed features (>1)
high_skew = skewed_features[skewed_features > 1].index
for col in high_skew:
    X_train[col] = np.log1p(X_train[col])
    X_test[col] = np.log1p(X_test[col])
print("Skewness Transformation Done")
# Save split datasets
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
print("Part 5 Completed Successfully")