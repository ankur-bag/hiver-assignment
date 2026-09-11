"""
Vector Dataset Deduplication Script.
Deduplicates (customer_text, amazon_reply) pairs, validates records,
adds deterministic unique IDs and detected language tags,
and writes to vector_data/clean_vectors.csv without modifying the original dataset.
"""

import sys
from pathlib import Path
import pandas as pd
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.vector.analyze_vector_dataset import detect_language_heuristic
INPUT_FILE = PROJECT_ROOT / "amazon_final_intent_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "vector_data"
OUTPUT_FILE = OUTPUT_DIR / "clean_vectors.csv"


def deduplicate_dataset():
    print(f"=== Vector Dataset Deduplication ===")
    print(f"Source: {INPUT_FILE}")
    print(f"Target: {OUTPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found at {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    initial_count = len(df)
    print(f"Initial raw record count: {initial_count:,}")

    # Strip extraneous whitespaces
    df["customer_text"] = df["customer_text"].astype(str).str.strip()
    df["amazon_reply"] = df["amazon_reply"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()

    # Drop null or blank rows
    valid_mask = (df["customer_text"] != "") & (df["amazon_reply"] != "") & (df["intent"] != "")
    df = df[valid_mask].copy()

    # Deduplicate strictly on (customer_text + amazon_reply)
    df = df.drop_duplicates(subset=["customer_text", "amazon_reply"], keep="first").copy()
    final_count = len(df)
    dropped = initial_count - final_count
    print(f"Deduplicated records: {final_count:,} (Removed {dropped:,} exact duplicate pairs)")

    # Assign deterministic IDs using hash or index prefix
    # e.g., case_000001 or hash-based
    df["id"] = [f"case_{i:06d}" for i in range(len(df))]

    # Tag language heuristic
    print("Tagging detected languages...")
    df["language"] = df["customer_text"].apply(lambda t: detect_language_heuristic(t).split()[0])

    # Select and order required columns: id, customer_text, amazon_reply, intent, language
    output_cols = ["id", "customer_text", "amazon_reply", "intent", "language"]
    clean_df = df[output_cols]

    clean_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"Saved clean dataset to {OUTPUT_FILE} ({len(clean_df):,} rows)")
    print("=== Deduplication Complete ===")


if __name__ == "__main__":
    deduplicate_dataset()
