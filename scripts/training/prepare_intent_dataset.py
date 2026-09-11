import pandas as pd


INPUT_FILE = "amazon_intent_dataset.csv"
OUTPUT_FILE = "amazon_final_intent_dataset.csv"


df = pd.read_csv(INPUT_FILE)


remove_intents = [
    "THANKS_OR_CLOSING",
    "GENERAL_SUPPORT_NOISE",
    "NOISE_URL_ATTACHMENT"
]


df = df[
    ~df["intent"].isin(remove_intents)
]


df = df.dropna(
    subset=["customer_text", "intent"]
)


print("Final dataset size:")
print(len(df))


print("\nIntent distribution:")
print(
    df["intent"].value_counts()
)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nSaved:")
print(OUTPUT_FILE)