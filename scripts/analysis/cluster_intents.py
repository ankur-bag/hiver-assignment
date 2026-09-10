import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import os


INPUT_FILE = "amazon_clean_conversations.csv"
OUTPUT_FILE = "amazon_clustered_conversations.csv"


df = pd.read_csv(INPUT_FILE)

df = df.dropna(
    subset=["customer_text"]
)

texts = df["customer_text"].astype(str).tolist()


print("Total conversations:", len(texts))


print("\nLoading embedding model...")

model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)


print("\nGenerating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)


print("\nEmbedding shape:")
print(embeddings.shape)


clusters = 12


print("\nRunning clustering...")


kmeans = KMeans(
    n_clusters=clusters,
    random_state=42,
    n_init=10
)


labels = kmeans.fit_predict(
    embeddings
)


df["cluster_id"] = labels


df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nSaved:")
print(OUTPUT_FILE)


print("\nCluster examples")


for cluster in sorted(df["cluster_id"].unique()):

    print("\n==============================")
    print("CLUSTER:", cluster)
    print("==============================")

    samples = df[
        df["cluster_id"] == cluster
    ]["customer_text"].head(8)

    for sample in samples:
        print("-", sample[:200])