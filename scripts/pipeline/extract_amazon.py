import pandas as pd

df = pd.read_csv("twcs.csv")

amazon = df[
    (df["author_id"] == "AmazonHelp") |
    (df["in_response_to_tweet_id"].isin(
        df[df["author_id"] == "AmazonHelp"]["tweet_id"]
    ))
]

print("Amazon related tweets:")
print(amazon.shape)

amazon.to_csv(
    "amazon_support_tweets.csv",
    index=False
)

print("Saved!")