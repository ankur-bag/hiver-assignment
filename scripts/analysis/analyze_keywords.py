import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


df = pd.read_csv("amazon_clean_conversations.csv")


texts = df["customer_text"].astype(str).str.lower()


words = []


for text in texts:

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z ]", " ", text)

    for word in text.split():

        if word not in ENGLISH_STOP_WORDS and len(word) > 2:
            words.append(word)


counter = Counter(words)


print("Useful keywords:\n")

for word, count in counter.most_common(100):
    print(word, ":", count)