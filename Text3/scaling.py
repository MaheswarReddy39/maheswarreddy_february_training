import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, Normalizer
# Load encoded dataset
df = pd.read_csv("encoded_train.csv")
print("Original Shape:", df.shape)
# Select only numerical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
# Remove target column from scaling
num_cols = num_cols.drop('SalePrice')
# Min-Max Scaling
minmax = MinMaxScaler()
df_minmax = df.copy()
df_minmax[num_cols] = minmax.fit_transform(df[num_cols])
print("Min-Max Scaling Done")
# Max Absolute Scaling
maxabs = MaxAbsScaler()
df_maxabs = df.copy()
df_maxabs[num_cols] = maxabs.fit_transform(df[num_cols])
print("Max Absolute Scaling Done")
# Z-Score (Standardization)
standard = StandardScaler()
df_standard = df.copy()
df_standard[num_cols] = standard.fit_transform(df[num_cols])
print("Z-Score Scaling Done")
# Vector Normalization
normalizer = Normalizer()
df_normalized = df.copy()
df_normalized[num_cols] = normalizer.fit_transform(df[num_cols])
print("Vector Normalization Done")
# Save all versions
df_minmax.to_csv("scaled_minmax.csv", index=False)
df_maxabs.to_csv("scaled_maxabs.csv", index=False)
df_standard.to_csv("scaled_standard.csv", index=False)
df_normalized.to_csv("scaled_normalized.csv", index=False)
print("Feature Scaling Completed Successfully")