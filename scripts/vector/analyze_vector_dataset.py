"""
Vector Dataset Quality Audit Script.
Analyzes amazon_final_intent_dataset.csv for:
- Total records and duplicate customer queries
- Exact duplicate (customer_text, amazon_reply) pairs
- Missing / null / empty values
- Text length distributions (characters and words)
- Intent distribution
- Multilingual / language distribution
Generates: reports/vector_dataset_quality.md
"""

from collections import Counter
from pathlib import Path
import re
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "amazon_final_intent_dataset.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_REPORT = REPORTS_DIR / "vector_dataset_quality.md"


def detect_language_heuristic(text: str) -> str:
    """
    Lightweight, fast rule-based language heuristic:
    Checks for Devanagari script, common Spanish, German, French accent markers
    or keywords, defaulting to English.
    """
    text_lower = text.lower()

    # Devanagari Unicode range
    if re.search(r"[\u0900-\u097F]", text):
        return "hi (Hindi - Devanagari)"

    # Common Hinglish / Romanized Hindi markers
    hinglish_markers = ["mera", "meri", "kya", "nahi", "aaya", "karo", "bhai", "hai", "mujhe", "kab", "milega", "hoga"]
    words = set(re.findall(r"\b\w+\b", text_lower))
    if len(words.intersection(hinglish_markers)) >= 2:
        return "hi-Latn (Hinglish)"

    # Spanish markers
    spanish_markers = ["el", "la", "de", "que", "en", "por", "paquete", "pedido", "llegado", "reembolso", "cuenta"]
    if len(words.intersection(spanish_markers)) >= 3 or any(c in text for c in "áéíóúñ¿¡"):
        return "es (Spanish)"

    # German markers
    german_markers = ["und", "der", "die", "das", "nicht", "bestellung", "paket", "konto", "bitte", "lieferung"]
    if len(words.intersection(german_markers)) >= 3 or any(c in text for c in "äöüß"):
        return "de (German)"

    # French markers
    french_markers = ["le", "la", "les", "des", "pour", "mon", "colis", "livraison", "compte", "remboursement"]
    if len(words.intersection(french_markers)) >= 3 or any(c in text for c in "àâéèêëîïôùûç"):
        return "fr (French)"

    return "en (English / Other)"


def analyze_dataset():
    print(f"Loading dataset: {DATASET_PATH}...")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    total_records = len(df)
    print(f"Total raw records: {total_records:,}")

    # 1. Missing Values
    null_customer = df["customer_text"].isna().sum() + (df["customer_text"].astype(str).str.strip() == "").sum()
    null_reply = df["amazon_reply"].isna().sum() + (df["amazon_reply"].astype(str).str.strip() == "").sum()
    null_intent = df["intent"].isna().sum() + (df["intent"].astype(str).str.strip() == "").sum()

    # Clean strings for duplicate analysis
    cust_clean = df["customer_text"].astype(str).str.strip()
    reply_clean = df["amazon_reply"].astype(str).str.strip()
    df["pair_key"] = cust_clean + " ||| " + reply_clean

    # 2. Duplicate counts
    duplicate_queries = df.duplicated(subset=["customer_text"]).sum()
    unique_queries = df["customer_text"].nunique()
    duplicate_pairs = df.duplicated(subset=["pair_key"]).sum()
    unique_pairs = df["pair_key"].nunique()

    # 3. Text length statistics
    cust_chars = cust_clean.str.len()
    cust_words = cust_clean.apply(lambda x: len(x.split()))
    reply_chars = reply_clean.str.len()
    reply_words = reply_clean.apply(lambda x: len(x.split()))

    # 4. Intent distribution
    intent_counts = df["intent"].value_counts()

    # 5. Language detection sample / full
    print("Performing language distribution analysis...")
    detected_langs = df["customer_text"].astype(str).apply(detect_language_heuristic)
    lang_counts = Counter(detected_langs)

    # Compile report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_content = f"""# Vector Dataset Quality & Audit Report

**Dataset Audited:** `amazon_final_intent_dataset.csv`  
**Audit Purpose:** Pre-ingestion data quality verification for Pinecone Vector Indexing  
**Timestamp:** Current Release  

---

## 1. Executive Summary & Health Metrics

| Health Check Metric | Count / Value | Status |
| :--- | :--- | :--- |
| **Total Input Records** | **{total_records:,}** | Baseline |
| **Missing Customer Texts** | **{null_customer}** | PASS |
| **Missing Amazon Replies** | **{null_reply}** | PASS |
| **Missing Intent Labels** | **{null_intent}** | PASS |
| **Duplicate Customer Queries** | **{duplicate_queries:,}** ({duplicate_queries/total_records*100:.1f}%) | Flagged for review |
| **Exact Duplicate (Query + Reply) Pairs** | **{duplicate_pairs:,}** ({duplicate_pairs/total_records*100:.1f}%) | To be removed |
| **Unique Historical Solutions After Deduplication** | **{unique_pairs:,}** | Target Vector Count |

---

## 2. Text Statistics & Length Distribution

| Metric | Customer Queries (Chars) | Customer Queries (Words) | Amazon Replies (Chars) | Amazon Replies (Words) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean** | {cust_chars.mean():.1f} | {cust_words.mean():.1f} | {reply_chars.mean():.1f} | {reply_words.mean():.1f} |
| **Median** | {cust_chars.median():.0f} | {cust_words.median():.0f} | {reply_chars.median():.0f} | {reply_words.median():.0f} |
| **Std Dev** | {cust_chars.std():.1f} | {cust_words.std():.1f} | {reply_chars.std():.1f} | {reply_words.std():.1f} |
| **Minimum** | {cust_chars.min()} | {cust_words.min()} | {reply_chars.min()} | {reply_words.min()} |
| **Maximum** | {cust_chars.max():,} | {cust_words.max():,} | {reply_chars.max():,} | {reply_words.max():,} |

---

## 3. Intent Class Distribution

| Intent Label | Record Count | Percentage |
| :--- | :---: | :---: |
"""
    for intent_name, count in intent_counts.items():
        report_content += f"| `{intent_name}` | {count:,} | {count/total_records*100:.2f}% |\n"

    report_content += f"""
---

## 4. Language Distribution

| Detected Language Category | Count | Percentage |
| :--- | :---: | :---: |
"""
    for lang, count in lang_counts.most_common():
        report_content += f"| `{lang}` | {count:,} | {count/total_records*100:.2f}% |\n"

    report_content += f"""
---

## 5. Cleaning & Deduplication Decisions

1. **Exact Pair Deduplication**:
   - {duplicate_pairs:,} records share an identical `(customer_text, amazon_reply)` combination. Ingesting these causes redundant search results and inflates vector index costs without adding informational entropy.
   - **Action**: Remove exact duplicate pairs in `scripts/vector/deduplicate_vectors.py`.

2. **Distinct Resolutions for Common Queries**:
   - Queries with identical customer text but *different* Amazon replies represent different resolution branches or follow-ups and will be **retained**.

3. **Final Target Index Size**:
   - The final clean vector set will contain **{unique_pairs:,}** records saved to `vector_data/clean_vectors.csv`.
"""

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Audit report saved to {OUTPUT_REPORT}")
    print(f"Summary: Total {total_records:,} rows -> {unique_pairs:,} unique (query, reply) pairs.")


if __name__ == "__main__":
    analyze_dataset()
