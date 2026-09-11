import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


INPUT_FILE = "amazon_final_intent_dataset.csv"


df = pd.read_csv(INPUT_FILE)


X = df["customer_text"].astype(str)
y = df["intent"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y
)


print("Loading embedding model...")


embedder = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)


print("Generating train embeddings...")


X_train_emb = embedder.encode(
    X_train.tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)


print("Generating test embeddings...")


X_test_emb = embedder.encode(
    X_test.tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)


print("Training classifier...")


model = LogisticRegression(
    max_iter=1000
)


model.fit(
    X_train_emb,
    y_train
)


pred = model.predict(
    X_test_emb
)


print("\nAccuracy:")
print(
    accuracy_score(
        y_test,
        pred
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        pred
    )
)


joblib.dump(
    model,
    "models/embedding_intent_classifier.pkl"
)


print("\nSaved model")