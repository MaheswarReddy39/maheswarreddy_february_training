import sys
print("Using python from : ", sys.executable)
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
# Load cleaned dataset
df = pd.read_csv("cleaned_train.csv")
print("Original Shape:", df.shape)
# LABEL ENCODING
le = LabelEncoder()
df['Neighborhood_Label'] = le.fit_transform(df['Neighborhood'])
print("Label Encoding Done")
# ONE HOT ENCODING
df = pd.get_dummies(df, columns=['MSZoning'], drop_first=True)
print("One Hot Encoding Done")
# ORDINAL ENCODING
# Example ordered column: ExterQual
# Typical order: Po < Fa < TA < Gd < Ex
quality_order = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
oe = OrdinalEncoder(categories=[quality_order])
df['ExterQual_Ordinal'] = oe.fit_transform(df[['ExterQual']])
print("Ordinal Encoding Done")
# FREQUENCY ENCODING
freq = df['Neighborhood'].value_counts()
df['Neighborhood_Freq'] = df['Neighborhood'].map(freq)
print("Frequency Encoding Done")
# TARGET ENCODING
target_mean = df.groupby('Neighborhood')['SalePrice'].mean()
df['Neighborhood_Target'] = df['Neighborhood'].map(target_mean)
print("Target Encoding Done")
# SAVE FILE
df.to_csv("encoded_train.csv", index=False)
print("Encoding Completed Successfully")
print("Final Shape:", df.shape)