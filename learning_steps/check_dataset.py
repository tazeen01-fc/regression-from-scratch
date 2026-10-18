import os
print(os.listdir())

import pandas as pd

df = pd.read_csv("../data/data.csv")

print("Rows, columns:", df.shape)
print("\nColumn types:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nFirst 5 rows:")
print(df.head())