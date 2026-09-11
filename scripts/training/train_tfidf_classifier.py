import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report
)


INPUT_FILE = "amazon_final_intent_dataset.csv"


df = pd.read_csv(INPUT_FILE)


X = df["customer_text"]
y = df["intent"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1,2)
)


X_train_vec = vectorizer.fit_transform(
    X_train
)

X_test_vec = vectorizer.transform(
    X_test
)


model = LogisticRegression(
    max_iter=1000
)


model.fit(
    X_train_vec,
    y_train
)


pred = model.predict(
    X_test_vec
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
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)


joblib.dump(
    model,
    "models/tfidf_intent_classifier.pkl"
)


print("\nModel saved")