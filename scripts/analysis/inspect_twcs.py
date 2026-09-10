import pandas as pd

file = "twcs.csv"

df = pd.read_csv(file)

print("Dataset Size:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())