"""
Train and Evaluate Logistic Regression Intent Classifier using Gemini Embedding 2.
Supports:
- 384d vs 768d dimension benchmarking
- Single-vector (RETRIEVAL_QUERY: 'task: search result | query: ...') vs Dual-vector (CLASSIFICATION: 'task: classification | query: ...')
- Deterministic train/test split matching MiniLM baseline (test_size=0.15, random_state=42, stratify=y)
- Resumable checkpointing to avoid duplicate API calls
- Exporting pure NumPy weights to backend/ml/models/gemini_classifier_weights.npz
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


def batch_embed_texts(
    client: genai.Client,
    texts: list[str],
    prefix: str,
    dimension: int = 384,
    batch_size: int = 100,
    model_name: str = "gemini-embedding-2",
    cache_path: Path = None,
) -> np.ndarray:
    """
    Embeds a list of texts using Gemini Embedding 2 with batching, rate-limit pacing, and checkpointing.
    """
    if cache_path and cache_path.exists():
        logger.info(f"Loading cached embeddings from {cache_path}")
        data = np.load(cache_path)
        if "embeddings" in data and len(data["embeddings"]) == len(texts):
            return data["embeddings"]

    total = len(texts)
    embeddings_list = []
    
    # Check partial checkpoint
    partial_path = cache_path.with_suffix(".partial.npy") if cache_path else None
    start_idx = 0
    if partial_path and partial_path.exists():
        try:
            partial_data = np.load(partial_path)
            embeddings_list = list(partial_data)
            start_idx = len(embeddings_list)
            logger.info(f"Resuming embedding generation from index {start_idx}/{total}")
        except Exception as e:
            logger.warning(f"Failed to load partial checkpoint: {e}. Starting fresh.")
            embeddings_list = []
            start_idx = 0

    formatted_texts = [f"{prefix}{t}" for t in texts]

    for i in range(start_idx, total, batch_size):
        batch = formatted_texts[i : i + batch_size]
        contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in batch]
        
        for attempt in range(1, 10):
            try:
                res = client.models.embed_content(
                    model=model_name,
                    contents=contents,
                    config={"output_dimensionality": dimension},
                )
                batch_emb = [emb.values for emb in res.embeddings]
                embeddings_list.extend(batch_emb)
                # Polite pacing to stay safely within free tier RPM limit
                time.sleep(1.0)
                break
            except Exception as e:
                err_msg = str(e)
                logger.warning(f"Batch {i}-{i+len(batch)} failed (attempt {attempt}/10): {err_msg}")
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    sleep_sec = 31.0
                    logger.info(f"Rate limited (429). Sleeping {sleep_sec}s before retry...")
                    time.sleep(sleep_sec)
                else:
                    time.sleep(2 ** min(attempt, 5))
                if attempt == 9:
                    raise e

        # Checkpoint every 500 samples
        if len(embeddings_list) % 500 == 0 or len(embeddings_list) == total:
            if partial_path:
                np.save(partial_path, np.array(embeddings_list, dtype=np.float32))
            logger.info(f"Progress: {len(embeddings_list)}/{total} embeddings generated ({len(embeddings_list)/total*100:.1f}%)")

    emb_array = np.array(embeddings_list, dtype=np.float32)
    if cache_path:
        np.savez_compressed(cache_path, embeddings=emb_array)
        if partial_path and partial_path.exists():
            partial_path.unlink()
        logger.info(f"Saved complete embeddings matrix to {cache_path} (shape: {emb_array.shape})")

    return emb_array


def train_and_evaluate():
    logger.info(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    logger.info(f"Dataset loaded: {len(df)} records across {df['intent'].nunique()} intents.")

    X = df["customer_text"].astype(str)
    y = df["intent"]

    # Deterministic split matching baseline
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    client = get_gemini_client()

    query_prefix = "task: search result | query: "
    class_prefix = "task: classification | query: "

    results = {}

    # Benchmark 1: 384-dimensional Single-Vector (Search Result Query)
    logger.info("\n=== Benchmark 1: 384d Single-Vector Query Embedding ===")
    train_emb_384 = batch_embed_texts(
        client, X_train.tolist(), query_prefix, dimension=384,
        cache_path=CACHE_DIR / "train_query_384.npz"
    )
    test_emb_384 = batch_embed_texts(
        client, X_test.tolist(), query_prefix, dimension=384,
        cache_path=CACHE_DIR / "test_query_384.npz"
    )

    clf_384 = LogisticRegression(max_iter=1000, C=1.0)
    clf_384.fit(train_emb_384, y_train)
    pred_384 = clf_384.predict(test_emb_384)
    
    acc_384 = accuracy_score(y_test, pred_384)
    macro_f1_384 = f1_score(y_test, pred_384, average="macro")
    weighted_f1_384 = f1_score(y_test, pred_384, average="weighted")
    logger.info(f"384d Query Embedding - Accuracy: {acc_384*100:.2f}%, Macro F1: {macro_f1_384:.4f}, Weighted F1: {weighted_f1_384:.4f}")
    results["gemini_384_query"] = {
        "accuracy": acc_384,
        "macro_f1": macro_f1_384,
        "weighted_f1": weighted_f1_384,
        "report": classification_report(y_test, pred_384, output_dict=True),
    }

    # Benchmark 2: 768-dimensional Single-Vector (Search Result Query)
    logger.info("\n=== Benchmark 2: 768d Single-Vector Query Embedding ===")
    train_emb_768 = batch_embed_texts(
        client, X_train.tolist(), query_prefix, dimension=768,
        cache_path=CACHE_DIR / "train_query_768.npz"
    )
    test_emb_768 = batch_embed_texts(
        client, X_test.tolist(), query_prefix, dimension=768,
        cache_path=CACHE_DIR / "test_query_768.npz"
    )

    clf_768 = LogisticRegression(max_iter=1000, C=1.0)
    clf_768.fit(train_emb_768, y_train)
    pred_768 = clf_768.predict(test_emb_768)

    acc_768 = accuracy_score(y_test, pred_768)
    macro_f1_768 = f1_score(y_test, pred_768, average="macro")
    weighted_f1_768 = f1_score(y_test, pred_768, average="weighted")
    logger.info(f"768d Query Embedding - Accuracy: {acc_768*100:.2f}%, Macro F1: {macro_f1_768:.4f}, Weighted F1: {weighted_f1_768:.4f}")
    results["gemini_768_query"] = {
        "accuracy": acc_768,
        "macro_f1": macro_f1_768,
        "weighted_f1": weighted_f1_768,
        "report": classification_report(y_test, pred_768, output_dict=True),
    }

    # Benchmark 3: 384-dimensional Classification-Specific Embedding
    logger.info("\n=== Benchmark 3: 384d Classification-Specific Embedding ===")
    train_emb_class_384 = batch_embed_texts(
        client, X_train.tolist(), class_prefix, dimension=384,
        cache_path=CACHE_DIR / "train_class_384.npz"
    )
    test_emb_class_384 = batch_embed_texts(
        client, X_test.tolist(), class_prefix, dimension=384,
        cache_path=CACHE_DIR / "test_class_384.npz"
    )

    clf_class_384 = LogisticRegression(max_iter=1000, C=1.0)
    clf_class_384.fit(train_emb_class_384, y_train)
    pred_class_384 = clf_class_384.predict(test_emb_class_384)

    acc_class_384 = accuracy_score(y_test, pred_class_384)
    macro_f1_class_384 = f1_score(y_test, pred_class_384, average="macro")
    weighted_f1_class_384 = f1_score(y_test, pred_class_384, average="weighted")
    logger.info(f"384d Classification-Specific - Accuracy: {acc_class_384*100:.2f}%, Macro F1: {macro_f1_class_384:.4f}")
    results["gemini_384_classification"] = {
        "accuracy": acc_class_384,
        "macro_f1": macro_f1_class_384,
        "weighted_f1": weighted_f1_class_384,
    }

    # Save benchmark evaluation results
    eval_json_path = ROOT_DIR / "scripts" / "evaluation" / "gemini_embedding_benchmark.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved benchmark results to {eval_json_path}")

    # Determine winning model: prefer 384d single-vector if accuracy is within 0.5% of 768d
    selected_dim = 384
    selected_clf = clf_384
    if acc_768 - acc_384 > 0.005:
        logger.info(f"768d model showed significant improvement (+{(acc_768-acc_384)*100:.2f}%). Selecting 768d.")
        selected_dim = 768
        selected_clf = clf_768
    else:
        logger.info(f"384d model accuracy ({acc_384*100:.2f}%) is comparable to 768d ({acc_768*100:.2f}%). Selecting 384d for optimal storage & latency efficiency.")

    # Export production NumPy weights
    weights_path = ROOT_DIR / "backend" / "ml" / "models" / "gemini_classifier_weights.npz"
    np.savez_compressed(
        weights_path,
        classes=selected_clf.classes_,
        coef=selected_clf.coef_,
        intercept=selected_clf.intercept_,
    )
    logger.info(f"Exported production NumPy weights to {weights_path}")

    # Export metadata
    label_map = {str(i): cls for i, cls in enumerate(selected_clf.classes_)}
    with open(ROOT_DIR / "backend" / "ml" / "models" / "gemini_label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    metadata = {
        "model_name": "gemini_embedding_intent_classifier",
        "embedding_model": "gemini-embedding-2",
        "dimension": selected_dim,
        "task_prefix": query_prefix,
        "accuracy": float(acc_384 if selected_dim == 384 else acc_768),
        "macro_f1": float(macro_f1_384 if selected_dim == 384 else macro_f1_768),
        "classes": list(selected_clf.classes_),
    }
    with open(ROOT_DIR / "backend" / "ml" / "models" / "gemini_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Training and export complete!")


if __name__ == "__main__":
    train_and_evaluate()
