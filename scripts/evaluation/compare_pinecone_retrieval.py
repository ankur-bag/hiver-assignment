"""
Pinecone Retrieval Validation: Baseline vs ONNX FP32 vs ONNX INT8.
Runs real queries against the Amazon Pinecone index and compares top-k retrieval results,
top-1 agreement, top-3 overlap, top-5 overlap, and similarity scores.
Saves results to JSON.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone
import onnxruntime as ort
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / "backend" / ".env")

ONNX_DIR = PROJECT_ROOT / "backend" / "ml" / "models" / "onnx"
FP32_PATH = ONNX_DIR / "model.onnx"
INT8_PATH = ONNX_DIR / "model_quantized.onnx"
TOKENIZER_PATH = ONNX_DIR / "tokenizer.json"

TEST_QUERIES = [
    "My package has not arrived yet",
    "Where is my refund?",
    "I cannot access my account",
    "My order is delayed",
    "Someone accessed my account",
    "Mera parcel abhi tak nahi aaya",
    "Necesito ayuda con mi pedido",
]


class FastOnnxEncoder:
    def __init__(self, model_path: Path, tokenizer_path: Path):
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=128)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_names = [inp.name for inp in self.session.get_inputs()]

    def encode(self, text: str) -> np.ndarray:
        encoding = self.tokenizer.encode(text)
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = np.array([encoding.type_ids], dtype=np.int64)

        outputs = self.session.run(None, inputs)
        last_hidden_state = outputs[0]

        input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
        sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        mean_pooled = sum_embeddings / sum_mask

        norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
        normalized = mean_pooled / np.clip(norms, a_min=1e-12, a_max=None)
        return normalized[0]


def query_pinecone(index, vector, top_k=5, namespace="amazon"):
    res = index.query(vector=vector.tolist(), top_k=top_k, namespace=namespace, include_metadata=True)
    return [(m["id"], round(float(m["score"]), 4)) for m in res.get("matches", [])]


def run_pinecone_validation():
    print("=== Running Pinecone Retrieval Validation ===")
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "amazon-support-resolutions")

    if not api_key:
        print("PINECONE_API_KEY missing. Skipping live Pinecone test.")
        return

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    print("Loading Baseline SentenceTransformer...")
    baseline_embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    print("Loading ONNX FP32...")
    fp32_encoder = FastOnnxEncoder(FP32_PATH, TOKENIZER_PATH)

    print("Loading ONNX INT8...")
    int8_encoder = FastOnnxEncoder(INT8_PATH, TOKENIZER_PATH)

    results = []
    top1_fp32_matches = 0
    top1_int8_matches = 0
    top3_fp32_overlaps = []
    top3_int8_overlaps = []
    top5_fp32_overlaps = []
    top5_int8_overlaps = []

    print("\n--- Retrieval Overlap Comparison ---")
    for q in TEST_QUERIES:
        v_base = baseline_embedder.encode(q, normalize_embeddings=True, convert_to_numpy=True)
        v_fp32 = fp32_encoder.encode(q)
        v_int8 = int8_encoder.encode(q)

        base_res = query_pinecone(index, v_base, top_k=5)
        fp32_res = query_pinecone(index, v_fp32, top_k=5)
        int8_res = query_pinecone(index, v_int8, top_k=5)

        base_ids = [r[0] for r in base_res]
        fp32_ids = [r[0] for r in fp32_res]
        int8_ids = [r[0] for r in int8_res]

        # Top 1 agreement
        fp32_top1 = int(base_ids[0] == fp32_ids[0]) if base_ids and fp32_ids else 0
        int8_top1 = int(base_ids[0] == int8_ids[0]) if base_ids and int8_ids else 0
        top1_fp32_matches += fp32_top1
        top1_int8_matches += int8_top1

        # Top 3 & 5 overlap
        fp32_top3_ov = len(set(base_ids[:3]).intersection(set(fp32_ids[:3]))) / 3.0
        int8_top3_ov = len(set(base_ids[:3]).intersection(set(int8_ids[:3]))) / 3.0
        top3_fp32_overlaps.append(fp32_top3_ov)
        top3_int8_overlaps.append(int8_top3_ov)

        fp32_top5_ov = len(set(base_ids[:5]).intersection(set(fp32_ids[:5]))) / 5.0
        int8_top5_ov = len(set(base_ids[:5]).intersection(set(int8_ids[:5]))) / 5.0
        top5_fp32_overlaps.append(fp32_top5_ov)
        top5_int8_overlaps.append(int8_top5_ov)

        print(f"Query: '{q}'")
        print(f"  Base Top-1 ID: {base_ids[0]} (score: {base_res[0][1]})")
        print(f"  FP32 Top-1 ID: {fp32_ids[0]} (score: {fp32_res[0][1]}) | Top-3 overlap: {fp32_top3_ov*100:.0f}% | Top-5: {fp32_top5_ov*100:.0f}%")
        print(f"  INT8 Top-1 ID: {int8_ids[0]} (score: {int8_res[0][1]}) | Top-3 overlap: {int8_top3_ov*100:.0f}% | Top-5: {int8_top5_ov*100:.0f}%")

        results.append({
            "query": q,
            "baseline_top_id": base_ids[0] if base_ids else None,
            "fp32_top_id": fp32_ids[0] if fp32_ids else None,
            "int8_top_id": int8_ids[0] if int8_ids else None,
            "fp32_top3_overlap": fp32_top3_ov,
            "int8_top3_overlap": int8_top3_ov,
            "fp32_top5_overlap": fp32_top5_ov,
            "int8_top5_overlap": int8_top5_ov,
        })

    summary = {
        "num_queries": len(TEST_QUERIES),
        "fp32": {
            "top1_agreement_pct": round(top1_fp32_matches / len(TEST_QUERIES) * 100, 2),
            "mean_top3_overlap_pct": round(float(np.mean(top3_fp32_overlaps)) * 100, 2),
            "mean_top5_overlap_pct": round(float(np.mean(top5_fp32_overlaps)) * 100, 2),
        },
        "int8": {
            "top1_agreement_pct": round(top1_int8_matches / len(TEST_QUERIES) * 100, 2),
            "mean_top3_overlap_pct": round(float(np.mean(top3_int8_overlaps)) * 100, 2),
            "mean_top5_overlap_pct": round(float(np.mean(top5_int8_overlaps)) * 100, 2),
        },
        "details": results
    }

    print("\n--- Summary Overlap ---")
    print(f"FP32: Top-1 Agreement={summary['fp32']['top1_agreement_pct']}%, Top-3 Overlap={summary['fp32']['mean_top3_overlap_pct']}%, Top-5 Overlap={summary['fp32']['mean_top5_overlap_pct']}%")
    print(f"INT8: Top-1 Agreement={summary['int8']['top1_agreement_pct']}%, Top-3 Overlap={summary['int8']['mean_top3_overlap_pct']}%, Top-5 Overlap={summary['int8']['mean_top5_overlap_pct']}%")

    out_file = PROJECT_ROOT / "scripts" / "evaluation" / "pinecone_retrieval_comparison.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    run_pinecone_validation()
