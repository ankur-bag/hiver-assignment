"""
Benchmark Gemini Embedding 2:
1. 384d vs 768d dimensional representation
2. Single-Vector (Search Result Query: 'task: search result | query: ...') vs Dual-Vector (Classification Task: 'task: classification | query: ...')
3. Multilingual intent evaluation across EN, HI, HI-LATN, ES, FR, DE
4. Export comparison results to scripts/evaluation/gemini_dimension_benchmark.json
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


def batch_embed(
    client: genai.Client,
    texts: list[str],
    prefix: str,
    dimension: int,
    cache_name: str,
    batch_size: int = 100
) -> np.ndarray:
    cache_file = CACHE_DIR / f"{cache_name}_{dimension}.npz"
    if cache_file.exists():
        logger.info(f"Loading cached embeddings from {cache_file}")
        data = np.load(cache_file)
        if len(data["embeddings"]) == len(texts):
            return data["embeddings"]

    total = len(texts)
    embeddings = []
    formatted = [f"{prefix}{t}" for t in texts]

    partial_file = CACHE_DIR / f"{cache_name}_{dimension}.partial.npy"
    start_idx = 0
    if partial_file.exists():
        try:
            partial_data = np.load(partial_file)
            embeddings = list(partial_data)
            start_idx = len(embeddings)
            logger.info(f"Resuming {cache_name} from {start_idx}/{total}")
        except Exception:
            embeddings = []
            start_idx = 0

    for i in range(start_idx, total, batch_size):
        batch = formatted[i : i + batch_size]
        contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in batch]

        for attempt in range(1, 10):
            try:
                res = client.models.embed_content(
                    model="gemini-embedding-2",
                    contents=contents,
                    config={"output_dimensionality": dimension},
                )
                batch_emb = [emb.values for emb in res.embeddings]
                embeddings.extend(batch_emb)
                time.sleep(3.0)
                break
            except Exception as e:
                err = str(e)
                logger.warning(f"Batch {i}/{total} failed (attempt {attempt}): {err}")
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    time.sleep(32.0)
                else:
                    time.sleep(2 ** min(attempt, 5))
                if attempt == 9:
                    raise e

        if len(embeddings) % 500 == 0 or len(embeddings) == total:
            np.save(partial_file, np.array(embeddings, dtype=np.float32))
            logger.info(f"[{cache_name}_{dimension}d] Processed {len(embeddings)}/{total} ({len(embeddings)/total*100:.1f}%)")

    emb_arr = np.array(embeddings, dtype=np.float32)
    np.savez_compressed(cache_file, embeddings=emb_arr)
    if partial_file.exists():
        partial_file.unlink()
    return emb_arr


def run_benchmark():
    df = pd.read_csv(DATASET_PATH)
    logger.info(f"Loaded {len(df)} samples.")

    X = df["customer_text"].astype(str)
    y = df["intent"]

    # Use stratified split (85% train, 15% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # Evaluate on a stratified representative benchmark subset of 500 train and 200 test
    train_subset, _, y_train_sub, _ = train_test_split(
        X_train, y_train, train_size=500, random_state=42, stratify=y_train
    )
    test_subset, _, y_test_sub, _ = train_test_split(
        X_test, y_test, train_size=200, random_state=42, stratify=y_test
    )

    client = get_gemini_client()
    query_prefix = "task: search result | query: "
    class_prefix = "task: classification | query: "

    benchmark_results = {}

    # 1. 384d Single-Vector Query Embedding
    logger.info("=== Embedding 384d Query Vectors ===")
    t0 = time.time()
    train_384_query = batch_embed(client, train_subset.tolist(), query_prefix, 384, "sub_train_query")
    test_384_query = batch_embed(client, test_subset.tolist(), query_prefix, 384, "sub_test_query")
    emb_time_384 = time.time() - t0

    clf_384 = LogisticRegression(max_iter=1000, C=1.0)
    clf_384.fit(train_384_query, y_train_sub)
    pred_384 = clf_384.predict(test_384_query)
    acc_384 = accuracy_score(y_test_sub, pred_384)
    macro_f1_384 = f1_score(y_test_sub, pred_384, average="macro")
    weighted_f1_384 = f1_score(y_test_sub, pred_384, average="weighted")

    logger.info(f"384d Query: Acc={acc_384*100:.2f}%, Macro F1={macro_f1_384:.4f}, Weighted F1={weighted_f1_384:.4f}")
    benchmark_results["gemini_384_query"] = {
        "dimension": 384,
        "task": "search result query (single-vector)",
        "accuracy": acc_384,
        "macro_f1": macro_f1_384,
        "weighted_f1": weighted_f1_384,
    }

    # 2. 768d Single-Vector Query Embedding
    logger.info("=== Embedding 768d Query Vectors ===")
    t0 = time.time()
    train_768_query = batch_embed(client, train_subset.tolist(), query_prefix, 768, "sub_train_query")
    test_768_query = batch_embed(client, test_subset.tolist(), query_prefix, 768, "sub_test_query")
    emb_time_768 = time.time() - t0

    clf_768 = LogisticRegression(max_iter=1000, C=1.0)
    clf_768.fit(train_768_query, y_train_sub)
    pred_768 = clf_768.predict(test_768_query)
    acc_768 = accuracy_score(y_test_sub, pred_768)
    macro_f1_768 = f1_score(y_test_sub, pred_768, average="macro")
    weighted_f1_768 = f1_score(y_test_sub, pred_768, average="weighted")

    logger.info(f"768d Query: Acc={acc_768*100:.2f}%, Macro F1={macro_f1_768:.4f}, Weighted F1={weighted_f1_768:.4f}")
    benchmark_results["gemini_768_query"] = {
        "dimension": 768,
        "task": "search result query (single-vector)",
        "accuracy": acc_768,
        "macro_f1": macro_f1_768,
        "weighted_f1": weighted_f1_768,
    }

    # 3. 384d Classification-Specific Embedding (Dual-Vector comparison)
    logger.info("=== Embedding 384d Classification Vectors ===")
    train_384_class = batch_embed(client, train_subset.tolist(), class_prefix, 384, "sub_train_class")
    test_384_class = batch_embed(client, test_subset.tolist(), class_prefix, 384, "sub_test_class")

    clf_384_class = LogisticRegression(max_iter=1000, C=1.0)
    clf_384_class.fit(train_384_class, y_train_sub)
    pred_384_class = clf_384_class.predict(test_384_class)
    acc_384_class = accuracy_score(y_test_sub, pred_384_class)
    macro_f1_384_class = f1_score(y_test_sub, pred_384_class, average="macro")
    weighted_f1_384_class = f1_score(y_test_sub, pred_384_class, average="weighted")

    logger.info(f"384d Class-Specific: Acc={acc_384_class*100:.2f}%, Macro F1={macro_f1_384_class:.4f}")
    benchmark_results["gemini_384_classification"] = {
        "dimension": 384,
        "task": "classification prefix (dual-vector)",
        "accuracy": acc_384_class,
        "macro_f1": macro_f1_384_class,
        "weighted_f1": weighted_f1_384_class,
    }

    # Save benchmark summary
    out_path = ROOT_DIR / "scripts" / "evaluation" / "gemini_dimension_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    logger.info(f"Saved benchmark summary to {out_path}")

    # Export production classifier weights (from 384d single-vector model)
    weights_path = ROOT_DIR / "backend" / "ml" / "models" / "gemini_classifier_weights.npz"
    np.savez_compressed(
        weights_path,
        classes=clf_384.classes_,
        coef=clf_384.coef_,
        intercept=clf_384.intercept_,
    )
    logger.info(f"Exported production NumPy weights to {weights_path}")

    # Export label map and metadata
    label_map = {str(i): cls for i, cls in enumerate(clf_384.classes_)}
    with open(ROOT_DIR / "backend" / "ml" / "models" / "gemini_label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    metadata = {
        "model_name": "gemini_embedding_intent_classifier",
        "embedding_model": "gemini-embedding-2",
        "dimension": 384,
        "task_prefix": query_prefix,
        "accuracy": float(acc_384),
        "macro_f1": float(macro_f1_384),
        "weighted_f1": float(weighted_f1_384),
        "classes": list(clf_384.classes_),
    }
    with open(ROOT_DIR / "backend" / "ml" / "models" / "gemini_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Benchmark and classifier export complete!")


if __name__ == "__main__":
    run_benchmark()
