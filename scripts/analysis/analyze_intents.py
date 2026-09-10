import pandas as pd
from collections import Counter
import re

df = pd.read_csv("amazon_clean_conversations.csv")

texts = df["customer_text"].astype(str).str.lower()

words = []

for text in texts:
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    words.extend(text.split())


common = Counter(words)

print("Total conversations:", len(df))

print("\nMost common words:")
for word, count in common.most_common(50):
    print(word, ":", count)