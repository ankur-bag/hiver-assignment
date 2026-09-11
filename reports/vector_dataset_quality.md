# Vector Dataset Quality & Audit Report

**Dataset Audited:** `amazon_final_intent_dataset.csv`  
**Audit Purpose:** Pre-ingestion data quality verification for Pinecone Vector Indexing  
**Timestamp:** Current Release  

---

## 1. Executive Summary & Health Metrics

| Health Check Metric | Count / Value | Status |
| :--- | :--- | :--- |
| **Total Input Records** | **52,124** | Baseline |
| **Missing Customer Texts** | **0** | PASS |
| **Missing Amazon Replies** | **0** | PASS |
| **Missing Intent Labels** | **0** | PASS |
| **Duplicate Customer Queries** | **4,173** (8.0%) | Flagged for review |
| **Exact Duplicate (Query + Reply) Pairs** | **0** (0.0%) | To be removed |
| **Unique Historical Solutions After Deduplication** | **52,124** | Target Vector Count |

---

## 2. Text Statistics & Length Distribution

| Metric | Customer Queries (Chars) | Customer Queries (Words) | Amazon Replies (Chars) | Amazon Replies (Words) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean** | 123.6 | 21.3 | 125.9 | 20.8 |
| **Median** | 122 | 21 | 123 | 20 |
| **Std Dev** | 59.7 | 11.3 | 45.1 | 8.1 |
| **Minimum** | 16 | 2 | 9 | 2 |
| **Maximum** | 322 | 65 | 304 | 57 |

---

## 3. Intent Class Distribution

| Intent Label | Record Count | Percentage |
| :--- | :---: | :---: |
| `ACCOUNT_ACCESS` | 7,825 | 15.01% |
| `DELIVERY_DELAY` | 7,814 | 14.99% |
| `ESCALATION` | 7,304 | 14.01% |
| `PACKAGE_NOT_RECEIVED` | 6,204 | 11.90% |
| `PRODUCT_ISSUE` | 4,918 | 9.44% |
| `CUSTOMER_SERVICE_CONTACT` | 4,854 | 9.31% |
| `REFUND_PENDING` | 4,501 | 8.64% |
| `ORDER_STATUS` | 4,383 | 8.41% |
| `ACCOUNT_SUPPORT` | 4,321 | 8.29% |

---

## 4. Language Distribution

| Detected Language Category | Count | Percentage |
| :--- | :---: | :---: |
| `en (English / Other)` | 46,517 | 89.24% |
| `es (Spanish)` | 4,378 | 8.40% |
| `de (German)` | 694 | 1.33% |
| `fr (French)` | 515 | 0.99% |
| `hi (Hindi - Devanagari)` | 15 | 0.03% |
| `hi-Latn (Hinglish)` | 5 | 0.01% |

---

## 5. Cleaning & Deduplication Decisions

1. **Exact Pair Deduplication**:
   - 0 records share an identical `(customer_text, amazon_reply)` combination. Ingesting these causes redundant search results and inflates vector index costs without adding informational entropy.
   - **Action**: Remove exact duplicate pairs in `scripts/vector/deduplicate_vectors.py`.

2. **Distinct Resolutions for Common Queries**:
   - Queries with identical customer text but *different* Amazon replies represent different resolution branches or follow-ups and will be **retained**.

3. **Final Target Index Size**:
   - The final clean vector set will contain **52,124** records saved to `vector_data/clean_vectors.csv`.
