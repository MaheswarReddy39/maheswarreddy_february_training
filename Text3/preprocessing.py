import pandas as pd
import numpy as np
# Load dataset
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "train.csv")
df = pd.read_csv(file_path)
print("Original Shape:", df.shape)
# Handling Missing Values
# Numerical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
# Categorical columns
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
print("Missing values handled")
# 2. Fix Data Types (if needed)
# Example: Ensure MSSubClass is integer
df['MSSubClass'] = df['MSSubClass'].astype(int)
print("Data types fixed")
# 3. Remove Outliers (SalePrice)
Q1 = df['SalePrice'].quantile(0.25)
Q3 = df['SalePrice'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df = df[(df['SalePrice'] >= lower) & (df['SalePrice'] <= upper)]
print("Outliers removed")
# 4. Remove Duplicates
df = df.drop_duplicates()
print("Duplicates removed")
# 5. Drop Irrelevant Columns
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])
print("Irrelevant columns dropped")
# Save Clean Dataset
df.to_csv("cleaned_train.csv", index=False)
print("Preprocessing Completed Successfully")
print("Final Shape:", df.shape)