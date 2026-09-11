"""
Batch Embedding Generation Script for Historical Customer Conversations.
Encodes customer queries from amazon_final_intent_dataset.csv into
384-dimensional normalized vectors using paraphrase-multilingual-MiniLM-L12-v2,
and saves the result to vector_data/vectors.parquet.
"""

import os
import time
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DATASET = PROJECT_ROOT / "vector_data" / "clean_vectors.csv"
OUTPUT_DIR = PROJECT_ROOT / "vector_data"
OUTPUT_PARQUET = OUTPUT_DIR / "vectors.parquet"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 128


def generate_embeddings():
    print(f"=== Starting Embedding Generation ===")
    print(f"Dataset: {INPUT_DATASET}")
    print(f"Model: {MODEL_NAME}")
    print(f"Expected dimension: {EMBEDDING_DIM}")

    if not INPUT_DATASET.exists():
        raise FileNotFoundError(f"Input dataset not found at {INPUT_DATASET}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check if output already exists
    if OUTPUT_PARQUET.exists():
        print(f"Found existing vector store artifact at {OUTPUT_PARQUET}.")
        existing_table = pq.read_table(OUTPUT_PARQUET)
        print(f"Existing records: {len(existing_table)}, schema: {existing_table.schema.names}")
        if len(existing_table) == 52124:
            print("Dataset already fully embedded and verified! Skipping recomputation.")
            return

    # Load dataset
    print(f"Loading {INPUT_DATASET}...")
    df = pd.read_csv(INPUT_DATASET)
    print(f"Total rows loaded: {len(df):,}")

    required_cols = ["customer_text", "amazon_reply", "intent"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in dataset.")

    # Clean text representation
    customer_texts = df["customer_text"].astype(str).tolist()
    amazon_replies = df["amazon_reply"].astype(str).tolist()
    intents = df["intent"].astype(str).tolist()

    # Load SentenceTransformer
    print(f"Loading embedding model '{MODEL_NAME}'...")
    t0 = time.time()
    embedder = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded in {time.time() - t0:.2f}s")

    # Generate embeddings in batches
    print(f"Generating embeddings for {len(customer_texts):,} queries (batch_size={BATCH_SIZE})...")
    t_start = time.time()
    embeddings = embedder.encode(
        customer_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2 normalization
        convert_to_numpy=True
    )
    total_time = time.time() - t_start
    print(f"Embeddings generated in {total_time:.2f}s ({total_time / len(customer_texts) * 1000:.2f} ms/query).")

    # Validate dimensions
    assert embeddings.shape == (len(df), EMBEDDING_DIM), (
        f"Shape mismatch: expected ({len(df)}, {EMBEDDING_DIM}), got {embeddings.shape}"
    )

    # Format dataframe for parquet
    print("Assembling structured vector records...")
    ids = df["id"].tolist() if "id" in df.columns else [f"case_{i:06d}" for i in range(len(df))]
    languages = df["language"].tolist() if "language" in df.columns else ["en"] * len(df)
    
    # Store embedding as list of floats for parquet
    embedding_list = [emb.tolist() for emb in embeddings]

    table_data = {
        "id": ids,
        "embedding": embedding_list,
        "customer_text": customer_texts,
        "amazon_reply": amazon_replies,
        "intent": intents,
        "language": languages
    }

    # Use PyArrow to write high-compression parquet
    print(f"Writing to {OUTPUT_PARQUET}...")
    arrow_table = pa.Table.from_pydict(table_data)
    pq.write_table(arrow_table, OUTPUT_PARQUET, compression="snappy")

    file_size_mb = OUTPUT_PARQUET.stat().st_size / (1024 * 1024)
    print(f"Successfully saved {len(df):,} vector records to {OUTPUT_PARQUET} ({file_size_mb:.2f} MB).")
    print("=== Embedding Generation Complete ===")


if __name__ == "__main__":
    generate_embeddings()
