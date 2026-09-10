import pandas as pd

df = pd.read_csv("amazon_support_tweets.csv")

tweets = df.set_index("tweet_id")

pairs = []

for _, row in df[df["inbound"] == False].iterrows():

    parent_id = row["in_response_to_tweet_id"]

    if pd.notna(parent_id) and parent_id in tweets.index:

        parent = tweets.loc[parent_id]

        if parent["inbound"] == True:

            pairs.append({
                "customer_text": parent["text"],
                "amazon_reply": row["text"]
            })


pairs_df = pd.DataFrame(pairs)

print("Conversation pairs:")
print(pairs_df.shape)

print("\nExamples:")
print(pairs_df.head(10))

pairs_df.to_csv(
    "amazon_conversations.csv",
    index=False
)

print("\nSaved!")