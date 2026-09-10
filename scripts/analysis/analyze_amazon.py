import pandas as pd

df = pd.read_csv("amazon_support_tweets.csv")

print("Total tweets:")
print(len(df))

print("\nCustomer tweets:")
print((df["inbound"] == True).sum())

print("\nAmazon replies:")
print((df["inbound"] == False).sum())

print("\nMissing response links:")
print(df["in_response_to_tweet_id"].isna().sum())

print("\nConversation examples:")
print(df[df["inbound"] == True]["text"].head(10).to_string())