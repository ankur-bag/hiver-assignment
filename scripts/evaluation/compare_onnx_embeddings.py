"""
Embedding Equivalence Evaluation Script.
Compares Original SentenceTransformer vs ONNX FP32 vs ONNX INT8 embeddings
across multilingual test queries (English, Hindi, Hinglish, Spanish, French, German).
Saves results to JSON.
"""

import json
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ONNX_DIR = PROJECT_ROOT / "backend" / "ml" / "models" / "onnx"
FP32_PATH = ONNX_DIR / "model.onnx"
INT8_PATH = ONNX_DIR / "model_quantized.onnx"
TOKENIZER_PATH = ONNX_DIR / "tokenizer.json"


TEST_QUERIES = [
    # English
    "Where is my package? It was supposed to arrive yesterday.",
    "I need a refund for my cancelled order #108-9812",
    "Someone compromised my account with unauthorized orders",
    "The item delivered is defective and damaged",
    "How can I return an item I received last week?",
    "I want to speak with a human agent or manager immediately",
    # Hindi / Hinglish
    "Mera package delay ho gaya hai abhi tak nahi aaya",
    "Mera order deliver nahi hua",
    "Mujhe refund kab milega mere order ka?",
    "Mera account hack ho gaya hai",
    # Spanish
    "Mi paquete no ha llegado todavía, está retrasado",
    "Necesito un reembolso de mi dinero por favor",
    "No puedo entrar a mi cuenta de Amazon",
    # French
    "Mon colis n'est pas encore arrivé",
    "Je souhaite obtenir un remboursement pour ma commande",
    # German
    "Wo ist meine Bestellung? Das Paket ist verspätet",
    "Mein Konto ist gesperrt, ich kann mich nicht einloggen",
]


class OnnxEncoder:
    def __init__(self, model_path: Path, tokenizer_path: Path):
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        # Enable truncation & padding to max length in batch
        self.tokenizer.enable_truncation(max_length=128)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_names = [inp.name for inp in self.session.get_inputs()]

    def encode(self, texts: list[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)

        outputs = self.session.run(None, inputs)
        last_hidden_state = outputs[0]  # (batch_size, seq_len, 384)

        # Attention mask-aware mean pooling
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
        sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        mean_pooled = sum_embeddings / sum_mask

        # L2 normalization
        norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
        normalized = mean_pooled / np.clip(norms, a_min=1e-12, a_max=None)
        return normalized


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes row-wise cosine similarity between two normalized matrices."""
    return np.sum(a * b, axis=1)


def evaluate_equivalence():
    print(f"=== Evaluating Embedding Equivalence Across {len(TEST_QUERIES)} Multilingual Queries ===")

    print("Loading Baseline SentenceTransformer...")
    baseline_embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    baseline_vectors = baseline_embedder.encode(TEST_QUERIES, normalize_embeddings=True, convert_to_numpy=True)

    print("Loading ONNX FP32 Encoder...")
    fp32_encoder = OnnxEncoder(FP32_PATH, TOKENIZER_PATH)
    fp32_vectors = fp32_encoder.encode(TEST_QUERIES)

    print("Loading ONNX INT8 Encoder...")
    int8_encoder = OnnxEncoder(INT8_PATH, TOKENIZER_PATH)
    int8_vectors = int8_encoder.encode(TEST_QUERIES)

    # Compute cosine similarities
    fp32_sims = cosine_similarity_matrix(baseline_vectors, fp32_vectors)
    int8_sims = cosine_similarity_matrix(baseline_vectors, int8_vectors)

    results = {
        "num_queries": len(TEST_QUERIES),
        "embedding_dim": int(baseline_vectors.shape[1]),
        "fp32": {
            "mean_cosine_similarity": float(np.mean(fp32_sims)),
            "min_cosine_similarity": float(np.min(fp32_sims)),
            "max_cosine_similarity": float(np.max(fp32_sims)),
            "mean_norm": float(np.mean(np.linalg.norm(fp32_vectors, axis=1))),
        },
        "int8": {
            "mean_cosine_similarity": float(np.mean(int8_sims)),
            "min_cosine_similarity": float(np.min(int8_sims)),
            "max_cosine_similarity": float(np.max(int8_sims)),
            "mean_norm": float(np.mean(np.linalg.norm(int8_vectors, axis=1))),
        },
        "per_query_results": []
    }

    print("\n--- Per Query Cosine Similarity ---")
    for i, q in enumerate(TEST_QUERIES):
        print(f"[{i+1:02d}] FP32 Sim: {fp32_sims[i]:.6f} | INT8 Sim: {int8_sims[i]:.6f} | Query: {q[:50]}...")
        results["per_query_results"].append({
            "query": q,
            "fp32_cosine_sim": float(fp32_sims[i]),
            "int8_cosine_sim": float(int8_sims[i])
        })

    print(f"\nSummary:")
    print(f"FP32 Mean Cosine Similarity to PyTorch Baseline: {results['fp32']['mean_cosine_similarity']:.6f} (Min: {results['fp32']['min_cosine_similarity']:.6f})")
    print(f"INT8 Mean Cosine Similarity to PyTorch Baseline: {results['int8']['mean_cosine_similarity']:.6f} (Min: {results['int8']['min_cosine_similarity']:.6f})")

    out_file = PROJECT_ROOT / "scripts" / "evaluation" / "onnx_embedding_equivalence.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    evaluate_equivalence()
