import pandas as pd
import re

input_file = "amazon_conversations.csv"
output_file = "amazon_clean_conversations.csv"

df = pd.read_csv(input_file)

print("Original:", len(df))


def clean_text(x):
    if pd.isna(x):
        return ""

    x = str(x)

    x = x.replace("\\n", " ")
    x = re.sub(r"\s+", " ", x)

    return x.strip()


df["customer_text"] = df["customer_text"].apply(clean_text)
df["amazon_reply"] = df["amazon_reply"].apply(clean_text)


# Remove empty rows
df = df[
    (df["customer_text"] != "") &
    (df["amazon_reply"] != "")
]


# Remove very short customer messages
df = df[
    df["customer_text"].str.len() > 15
]


# Remove obvious greetings/thanks
remove_words = [
    "thank you",
    "thanks",
    "thanks amazon",
    "hello",
    "hi"
]


def is_closing(text):
    t = text.lower()

    for word in remove_words:
        if t.strip() == word:
            return True

    return False


df = df[
    ~df["customer_text"].apply(is_closing)
]


# Remove duplicates

df = df.drop_duplicates(
    subset=[
        "customer_text",
        "amazon_reply"
    ]
)


print("After cleaning:", len(df))


df.to_csv(
    output_file,
    index=False
)

print("Saved:", output_file)