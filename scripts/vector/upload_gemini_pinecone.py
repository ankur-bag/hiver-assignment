"""
Production Gemini 2 Vector Migration Script for Pinecone.
Populates Pinecone Index 'amazon-support-resolutions' under namespace 'amazon-gemini-v2'.

Pipeline:
1. Stream dataset in batches of 100 rows.
2. Build 100 Content objects formatted with 'title: none | text: {customer_text}'.
3. Execute ONE Gemini embedding API call per batch for 100 Content objects.
4. Validate 384 dimensions and L2 normalize.
5. Upsert 100 vectors with full metadata into namespace 'amazon-gemini-v2'.
6. Update persistent checkpoint after each successful upsert.
7. Print real batch progress and verify Pinecone index stats at completion.
"""

import os
import sys
import time
import json
import logging
import math
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / "backend" / ".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "amazon-support-resolutions")
PINECONE_NAMESPACE = "amazon-gemini-v2"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DATASET_PATH = ROOT_DIR / "amazon_final_intent_dataset.csv"
CHECKPOINT_FILE = ROOT_DIR / "vector_data" / "gemini_pinecone_checkpoint.json"

BATCH_SIZE = 100
DIMENSION = 384
EMBEDDING_MODEL = "gemini-embedding-2"
DOC_PREFIX = "title: none | text: "


def load_checkpoint() -> dict:
    """Loads checkpoint state from disk."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading checkpoint file: {e}. Initializing fresh checkpoint.")
    return {
        "last_completed_idx": 0,
        "total_uploaded": 0,
        "failed_batches": [],
        "last_updated": None
    }


def save_checkpoint(last_idx: int, total_uploaded: int, failed_batches: list):
    """Persists progress to disk atomically."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_file = CHECKPOINT_FILE.with_suffix(".tmp")
    data = {
        "last_completed_idx": last_idx,
        "total_uploaded": total_uploaded,
        "failed_batches": failed_batches,
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    temp_file.replace(CHECKPOINT_FILE)


def upload_all_vectors(limit: int = None):
    logger.info("==================================================")
    logger.info("STARTING GEMINI EMBEDDING 2 PINECONE MIGRATION")
    logger.info(f"Target Index: {PINECONE_INDEX_NAME}")
    logger.info(f"Target Namespace: {PINECONE_NAMESPACE}")
    logger.info(f"Embedding Model: {EMBEDDING_MODEL} (dim={DIMENSION})")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info("==================================================")

    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not set in backend/.env")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in backend/.env")

    # Connect to Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    # Initialize Google GenAI client
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Load dataset
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    total_records = len(df)
    if limit:
        total_records = min(total_records, limit)

    checkpoint = load_checkpoint()
    start_idx = checkpoint.get("last_completed_idx", 0)
    total_uploaded = checkpoint.get("total_uploaded", 0)
    failed_batches = checkpoint.get("failed_batches", [])

    total_batches = math.ceil(total_records / BATCH_SIZE)
    current_batch_num = (start_idx // BATCH_SIZE) + 1

    logger.info(f"Total dataset records: {total_records:,}")
    logger.info(f"Resuming migration from record index: {start_idx:,} (Batch {current_batch_num}/{total_batches})")

    api_calls_made = 0

    for b_start in range(start_idx, total_records, BATCH_SIZE):
        b_end = min(b_start + BATCH_SIZE, total_records)
        batch_df = df.iloc[b_start:b_end]
        batch_size = len(batch_df)
        batch_num = (b_start // BATCH_SIZE) + 1

        # Format historical documents with asymmetric prefix
        texts = batch_df["customer_text"].fillna("").astype(str).tolist()
        formatted_docs = [f"{DOC_PREFIX}{t}" for t in texts]
        contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in formatted_docs]

        # 1. Execute ONE Gemini embedding API call for the batch
        embeddings = None
        for attempt in range(1, 12):
            try:
                res = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=contents,
                    config={"output_dimensionality": DIMENSION},
                )
                api_calls_made += 1
                if res and res.embeddings and len(res.embeddings) == batch_size:
                    embeddings = [emb.values for emb in res.embeddings]
                    break
                else:
                    raise ValueError(f"Received incomplete embeddings: expected {batch_size}, got {len(res.embeddings) if res and res.embeddings else 0}")
            except Exception as e:
                err_str = str(e)
                logger.warning(f"Batch {batch_num}/{total_batches} (rows {b_start}-{b_end}) API attempt {attempt} failed: {err_str}")
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    # Parse retry delay or conservative backoff
                    sleep_sec = 35.0 + (attempt * 5.0)
                    logger.info(f"Rate limited (429/RESOURCE_EXHAUSTED). Backing off for {sleep_sec:.1f}s...")
                    time.sleep(sleep_sec)
                else:
                    sleep_sec = min(2.0 ** attempt, 30.0)
                    time.sleep(sleep_sec)

                if attempt == 11:
                    logger.error(f"Exhausted 11 attempts on batch {batch_num}. Halting migration to preserve checkpoint.")
                    failed_batches.append({"batch_num": batch_num, "start": b_start, "end": b_end, "error": err_str})
                    save_checkpoint(b_start, total_uploaded, failed_batches)
                    raise e

        # 2. Validate dimensions & L2 normalize
        arr = np.array(embeddings, dtype=np.float32)
        if arr.shape[1] != DIMENSION:
            raise ValueError(f"Vector dimension mismatch: expected {DIMENSION}, got {arr.shape[1]}")

        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_embeddings = (arr / norms).tolist()

        # 3. Build idempotent vectors with stable IDs and complete metadata
        vectors_to_upsert = []
        for i, (_, row) in enumerate(batch_df.iterrows()):
            row_idx = b_start + i
            vec_id = f"amazon_gemini_{row_idx}"
            vectors_to_upsert.append({
                "id": vec_id,
                "values": norm_embeddings[i],
                "metadata": {
                    "customer_text": str(row["customer_text"])[:1000],
                    "amazon_reply": str(row["amazon_reply"])[:1000],
                    "intent": str(row["intent"]),
                    "language": "en",
                    "source": "TWCS",
                    "brand": "AmazonHelp",
                    "embedding_model": EMBEDDING_MODEL,
                    "embedding_version": "gemini-v2"
                }
            })

        # 4. Upsert to Pinecone namespace 'amazon-gemini-v2'
        upserted = False
        for attempt in range(1, 6):
            try:
                index.upsert(vectors=vectors_to_upsert, namespace=PINECONE_NAMESPACE)
                upserted = True
                break
            except Exception as e:
                logger.warning(f"Pinecone upsert attempt {attempt} failed on batch {batch_num}: {e}")
                time.sleep(2.0 ** attempt)
                if attempt == 5:
                    raise e

        total_uploaded += len(vectors_to_upsert)
        save_checkpoint(b_end, total_uploaded, failed_batches)

        # 5. Display real progress
        logger.info(
            f"Batch {batch_num}/{total_batches} | "
            f"Embedded: {len(embeddings)} | "
            f"Pinecone uploaded: {len(vectors_to_upsert)} | "
            f"Total: {b_end:,} / {total_records:,} ({(b_end/total_records)*100:.1f}%)"
        )

        # Gentle pacing between batches
        time.sleep(1.0)

    # Verify final index stats
    logger.info("==================================================")
    logger.info("Verifying Pinecone index stats...")
    time.sleep(3)
    stats = index.describe_index_stats()
    logger.info(f"Final Pinecone Index Stats:\n{stats}")
    logger.info(f"Migration completed! Total API calls made: {api_calls_made}")
    logger.info("==================================================")


if __name__ == "__main__":
    upload_all_vectors()
