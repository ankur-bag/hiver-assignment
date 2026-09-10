import pandas as pd

df = pd.read_csv("twcs.csv")

brands = df[df["inbound"] == False]["author_id"].value_counts()

print("Top support accounts:")
print(brands.head(50))