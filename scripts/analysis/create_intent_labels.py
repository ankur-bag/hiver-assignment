import pandas as pd
import json


INPUT_FILE = "amazon_clustered_conversations.csv"
OUTPUT_FILE = "amazon_intent_dataset.csv"
MAPPING_FILE = "docs/intent_mapping.json"


df = pd.read_csv(INPUT_FILE)


with open(MAPPING_FILE, "r") as f:
    mapping = json.load(f)


df["intent"] = df["cluster_id"].astype(str).map(mapping)


df = df.dropna(
    subset=["intent"]
)


print("Intent distribution:\n")

print(
    df["intent"].value_counts()
)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nSaved:")
print(OUTPUT_FILE)