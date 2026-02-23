import pandas as pd
df = pd.read_excel("studentsdata.xlsx")
# print(df)
# Basic Functions
""" print(df.head())  #First 5 rows
print(df.tail())  #Last 5 rows
print(df.info())  #Datatypes & info
print(df.describe()) #statistics
print(df.shape)     #rows and columns
print(df.columns)   #column names """
# Selecting Data
""" print(df["JAVA"])   #single column
print(df[["PYTHON", "JAVA", "C", "FSD"]])   #multiple condition
print(df.loc[0])    #label based
print(df.iloc[0])     #index based """ 
# Filtering Data
""" print([df["C"]>80])   #single condition
print((df["C"]>90) & (df["TOTAL ATTENDANCE"]>90)) #multiple conditions """
