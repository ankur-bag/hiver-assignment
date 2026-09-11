"""
High-Throughput Pinecone Batch Vector Uploader.
Reads vector_data/vectors.parquet, formats records with metadata,
and uploads them concurrently using ThreadPoolExecutor to Pinecone serverless
index 'amazon-support-resolutions' in the 'amazon' namespace with retries,
rate-limit resilience, and checkpoint tracking.
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from dotenv import load_dotenv
import pyarrow.parquet as pq
from pinecone import Pinecone

# Load environment
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / "backend" / ".env"
load_dotenv(ENV_FILE)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "amazon-support-resolutions")
NAMESPACE = "amazon"
PARQUET_PATH = PROJECT_ROOT / "vector_data" / "vectors.parquet"
PROGRESS_FILE = PROJECT_ROOT / "vector_data" / "upload_progress.txt"

BATCH_SIZE = 200
MAX_WORKERS = 8
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.5

# Thread synchronization for progress tracking
progress_lock = Lock()
total_uploaded_count = 0


def get_completed_checkpoint() -> int:
    """Reads the last successfully uploaded batch index."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r") as f:
                return int(f.read().strip())
        except Exception:
            return 0
    return 0


def update_progress(batch_count: int, current_max: int):
    """Safely updates uploaded counter and checkpoint file."""
    global total_uploaded_count
    with progress_lock:
        total_uploaded_count += batch_count
        if PROGRESS_FILE.exists():
            try:
                with open(PROGRESS_FILE, "r") as f:
                    prev = int(f.read().strip())
            except Exception:
                prev = 0
            if current_max > prev:
                with open(PROGRESS_FILE, "w") as f:
                    f.write(str(current_max))
        else:
            with open(PROGRESS_FILE, "w") as f:
                f.write(str(current_max))


def upload_batch_worker(index, batch_vectors, batch_start, batch_end):
    """Worker task for uploading a single batch with exponential backoff."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            index.upsert(
                vectors=batch_vectors,
                namespace=NAMESPACE
            )
            update_progress(len(batch_vectors), batch_end)
            return len(batch_vectors)
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"\n[Error] Failed batch {batch_start}-{batch_end} after {MAX_RETRIES} attempts: {e}")
                raise e
            sleep_time = INITIAL_BACKOFF * (2 ** (attempt - 1))
            time.sleep(sleep_time)


def upload_vectors(limit: int = None):
    global total_uploaded_count
    print("=== High-Throughput Pinecone Vector Upload ===")
    print(f"Target Index: {INDEX_NAME}")
    print(f"Namespace: {NAMESPACE}")
    print(f"Parquet source: {PARQUET_PATH}")
    print(f"Batch size: {BATCH_SIZE} | Concurrent workers: {MAX_WORKERS}")

    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"Parquet artifact not found at {PARQUET_PATH}. Run scripts/vector/create_embeddings.py first."
        )

    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY missing in backend/.env")

    # Connect to Pinecone
    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    # Read parquet
    print(f"Loading vectors from {PARQUET_PATH}...")
    table = pq.read_table(PARQUET_PATH)
    total_records = len(table)
    if limit:
        total_records = min(total_records, limit)
    print(f"Total records to upload: {total_records:,}")

    ids = table["id"].to_pylist()[:total_records]
    embeddings = table["embedding"].to_pylist()[:total_records]
    customer_texts = table["customer_text"].to_pylist()[:total_records]
    amazon_replies = table["amazon_reply"].to_pylist()[:total_records]
    intents = table["intent"].to_pylist()[:total_records]
    languages = table["language"].to_pylist()[:total_records]

    # Checkpoint check
    start_idx = get_completed_checkpoint()
    if start_idx >= total_records:
        print(f"All {total_records:,} records already uploaded according to {PROGRESS_FILE}!")
        stats = index.describe_index_stats()
        print(f"Index stats: {stats}")
        return

    if start_idx > 0:
        print(f"Resuming upload from record index {start_idx:,}...")

    total_uploaded_count = start_idx
    batches = []
    for b_start in range(start_idx, total_records, BATCH_SIZE):
        b_end = min(b_start + BATCH_SIZE, total_records)
        b_vectors = []
        for i in range(b_start, b_end):
            b_vectors.append({
                "id": ids[i],
                "values": embeddings[i],
                "metadata": {
                    "customer_text": customer_texts[i],
                    "amazon_reply": amazon_replies[i],
                    "intent": intents[i],
                    "language": languages[i]
                }
            })
        batches.append((b_vectors, b_start, b_end))

    total_batches = len(batches)
    print(f"Dispatching {total_batches:,} batches across {MAX_WORKERS} worker threads...")

    t0 = time.time()
    completed_batches = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(upload_batch_worker, index, b_vecs, b_s, b_e)
            for b_vecs, b_s, b_e in batches
        ]

        for future in as_completed(futures):
            future.result()  # Will raise if any batch failed
            completed_batches += 1
            elapsed = time.time() - t0
            current_total = total_uploaded_count
            pct = (current_total / total_records) * 100
            rate = (current_total - start_idx) / elapsed if elapsed > 0 else 0
            sys.stdout.write(f"\rProgress: {current_total:,}/{total_records:,} ({pct:.1f}%) | {rate:.0f} vecs/sec | Batches: {completed_batches}/{total_batches}")
            sys.stdout.flush()

    total_elapsed = time.time() - t0
    avg_rate = (total_records - start_idx) / total_elapsed if total_elapsed > 0 else 0
    print(f"\n\nUpload completed in {total_elapsed:.2f}s (Average rate: {avg_rate:.0f} vecs/sec)!")
    print("Verifying index statistics...")
    time.sleep(3)
    stats = index.describe_index_stats()
    print(f"Current Index Stats:\n{stats}")
    print("=== Pinecone Upload Complete ===")


if __name__ == "__main__":
    upload_vectors()
