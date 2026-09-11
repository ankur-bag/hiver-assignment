"""
Gemini Embedding 2 Intent Classifier Training & Benchmark.
Generates embeddings using gemini-embedding-2 (384d & 768d, single-vector query vs dual-vector classification),
evaluates intent classification metrics, compares dimensions, and exports production NumPy weights to backend/ml/models/gemini_classifier_weights.npz.
"""

import os
import time
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / "backend" / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in backend/.env")

DATASET_PATH = ROOT_DIR / "amazon_final_intent_dataset.csv"
CACHE_DIR = ROOT_DIR / ".cache" / "gemini_embeddings"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_gemini_client():
    return genai.Client(api_key=API_KEY)


def embed_batch_safe(
    client: genai.Client,
    texts: list[str],
    prefix: str,
    dimension: int,
    cache_path: Path,
    batch_size: int = 10
) -> np.ndarray:
    if cache_path.exists():
        logger.info(f"Loading cached embeddings from {cache_path}")
        data = np.load(cache_path)
        if len(data["embeddings"]) == len(texts):
            return data["embeddings"]

    total = len(texts)
    embeddings = []
    formatted = [f"{prefix}{t}" for t in texts]

    for i in range(0, total, batch_size):
        batch = formatted[i : i + batch_size]
        contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in batch]

        for attempt in range(1, 15):
            try:
                res = client.models.embed_content(
                    model="gemini-embedding-2",
                    contents=contents,
                    config={"output_dimensionality": dimension},
                )
                batch_vecs = [emb.values for emb in res.embeddings]
                embeddings.extend(batch_vecs)
                time.sleep(1.5)
                break
            except Exception as e:
                err_msg = str(e)
                logger.warning(f"Batch {i}/{total} failed (attempt {attempt}): {err_msg}")
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    sleep_sec = 35.0
                    logger.info(f"429 rate limit encountered. Backing off for {sleep_sec}s...")
                    time.sleep(sleep_sec)
                else:
                    time.sleep(2 ** min(attempt, 5))
                if attempt == 14:
                    raise e

        logger.info(f"Embedded {len(embeddings)}/{total} items (dim={dimension})")

    arr = np.array(embeddings, dtype=np.float32)
    # L2 normalization
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    np.savez_compressed(cache_path, embeddings=arr)
    logger.info(f"Saved {cache_path}")
    return arr


def run():
    logger.info("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    logger.info(f"Dataset has {len(df)} total rows across 9 intents.")

    # Select representative stratified sample of 180 train (20 per intent) and 90 test (10 per intent) for fast clean training & validation
    intents = df["intent"].unique()
    train_dfs = []
    test_dfs = []
    for intent in intents:
        subset = df[df["intent"] == intent]
        train_sub, test_sub = train_test_split(subset, train_size=20, test_size=10, random_state=42)
        train_dfs.append(train_sub)
        test_dfs.append(test_sub)

    train_df = pd.concat(train_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)
    test_df = pd.concat(test_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)

    X_train = train_df["customer_text"].tolist()
    y_train = train_df["intent"].tolist()
    X_test = test_df["customer_text"].tolist()
    y_test = test_df["intent"].tolist()

    logger.info(f"Training set: {len(X_train)} samples across 9 intents. Test set: {len(X_test)} samples.")

    client = get_gemini_client()
    query_prefix = "task: search result | query: "
    class_prefix = "task: classification | query: "

    # 1. 384d Single-Vector Query Embedding
    logger.info("\n=== Benchmark 1: 384d Single-Vector Query Embedding ===")
    train_384_query = embed_batch_safe(
        client, X_train, query_prefix, 384, CACHE_DIR / "train_384_query.npz"
    )
    test_384_query = embed_batch_safe(
        client, X_test, query_prefix, 384, CACHE_DIR / "test_384_query.npz"
    )

    clf_384 = LogisticRegression(max_iter=1000, C=5.0)
    clf_384.fit(train_384_query, y_train)
    pred_384 = clf_384.predict(test_384_query)
    acc_384 = accuracy_score(y_test, pred_384)
    macro_f1_384 = f1_score(y_test, pred_384, average="macro")
    weighted_f1_384 = f1_score(y_test, pred_384, average="weighted")
    logger.info(f"384d Query Model - Accuracy: {acc_384*100:.2f}%, Macro F1: {macro_f1_384:.4f}, Weighted F1: {weighted_f1_384:.4f}")

    # 2. 768d Single-Vector Query Embedding
    logger.info("\n=== Benchmark 2: 768d Single-Vector Query Embedding ===")
    train_768_query = embed_batch_safe(
        client, X_train, query_prefix, 768, CACHE_DIR / "train_768_query.npz"
    )
    test_768_query = embed_batch_safe(
        client, X_test, query_prefix, 768, CACHE_DIR / "test_768_query.npz"
    )

    clf_768 = LogisticRegression(max_iter=1000, C=5.0)
    clf_768.fit(train_768_query, y_train)
    pred_768 = clf_768.predict(test_768_query)
    acc_768 = accuracy_score(y_test, pred_768)
    macro_f1_768 = f1_score(y_test, pred_768, average="macro")
    weighted_f1_768 = f1_score(y_test, pred_768, average="weighted")
    logger.info(f"768d Query Model - Accuracy: {acc_768*100:.2f}%, Macro F1: {macro_f1_768:.4f}, Weighted F1: {weighted_f1_768:.4f}")

    # 3. 384d Classification-Specific Embedding (Dual-Vector comparison)
    logger.info("\n=== Benchmark 3: 384d Classification-Specific Task Embedding ===")
    train_384_class = embed_batch_safe(
        client, X_train, class_prefix, 384, CACHE_DIR / "train_384_class.npz"
    )
    test_384_class = embed_batch_safe(
        client, X_test, class_prefix, 384, CACHE_DIR / "test_384_class.npz"
    )

    clf_384_class = LogisticRegression(max_iter=1000, C=5.0)
    clf_384_class.fit(train_384_class, y_train)
    pred_384_class = clf_384_class.predict(test_384_class)
    acc_384_class = accuracy_score(y_test, pred_384_class)
    macro_f1_384_class = f1_score(y_test, pred_384_class, average="macro")
    logger.info(f"384d Classification Model - Accuracy: {acc_384_class*100:.2f}%, Macro F1: {macro_f1_384_class:.4f}")

    # Save benchmark metrics
    benchmark_data = {
        "gemini_384_query_single_vector": {
            "dimension": 384,
            "task": "search result query",
            "accuracy": acc_384,
            "macro_f1": macro_f1_384,
            "weighted_f1": weighted_f1_384,
        },
        "gemini_768_query_single_vector": {
            "dimension": 768,
            "task": "search result query",
            "accuracy": acc_768,
            "macro_f1": macro_f1_768,
            "weighted_f1": weighted_f1_768,
        },
        "gemini_384_classification_dual_vector": {
            "dimension": 384,
            "task": "classification prefix",
            "accuracy": acc_384_class,
            "macro_f1": macro_f1_384_class,
        }
    }
    with open(ROOT_DIR / "scripts" / "evaluation" / "gemini_dimension_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    # Export production weights for 384d Single-Vector Query model
    selected_clf = clf_384
    weights_path = ROOT_DIR / "backend" / "ml" / "models" / "gemini_classifier_weights.npz"
    np.savez_compressed(
        weights_path,
        classes=selected_clf.classes_,
        coef=selected_clf.coef_,
        intercept=selected_clf.intercept_,
    )
    logger.info(f"Exported production NumPy weights to {weights_path}")

    # Export label map
    label_map = {str(i): cls for i, cls in enumerate(selected_clf.classes_)}
    with open(ROOT_DIR / "backend" / "ml" / "models" / "gemini_label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    # Export metadata
    metadata = {
        "model_name": "gemini_embedding_intent_classifier",
        "embedding_model": "gemini-embedding-2",
        "dimension": 384,
        "task_prefix": query_prefix,
        "accuracy": float(acc_384),
        "macro_f1": float(macro_f1_384),
        "weighted_f1": float(weighted_f1_384),
        "classes": list(selected_clf.classes_),
    }
    with open(ROOT_DIR / "backend" / "ml" / "models" / "gemini_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("\n=== Training & Export Complete! ===")


if __name__ == "__main__":
    run()
