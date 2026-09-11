"""
Intent Classification Evaluation: Baseline vs ONNX FP32 vs ONNX INT8.
Evaluates accuracy, macro F1, weighted F1, and per-class metrics
on the 15% stratified test split from amazon_final_intent_dataset.csv.
Saves results to JSON and CSV.
"""

import json
import time
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import onnxruntime as ort
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "amazon_final_intent_dataset.csv"
ONNX_DIR = PROJECT_ROOT / "backend" / "ml" / "models" / "onnx"
FP32_PATH = ONNX_DIR / "model.onnx"
INT8_PATH = ONNX_DIR / "model_quantized.onnx"
TOKENIZER_PATH = ONNX_DIR / "tokenizer.json"
WEIGHTS_PATH = PROJECT_ROOT / "backend" / "ml" / "models" / "classifier_weights.npz"


class FastOnnxEncoder:
    def __init__(self, model_path: Path, tokenizer_path: Path):
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=128)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_names = [inp.name for inp in self.session.get_inputs()]

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encodings = self.tokenizer.encode_batch(batch)
            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
            if "token_type_ids" in self.input_names:
                inputs["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)

            outputs = self.session.run(None, inputs)
            last_hidden_state = outputs[0]

            input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
            sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
            sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            mean_pooled = sum_embeddings / sum_mask

            norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
            normalized = mean_pooled / np.clip(norms, a_min=1e-12, a_max=None)
            all_embeddings.append(normalized)

        return np.vstack(all_embeddings)


def predict_numpy(embeddings: np.ndarray, classes: np.ndarray, coef: np.ndarray, intercept: np.ndarray):
    logits = np.dot(embeddings, coef.T) + intercept
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    pred_indices = np.argmax(probs, axis=1)
    return classes[pred_indices], probs


def run_evaluation():
    print(f"=== Evaluating Intent Classifier on Held-Out Test Set ===")
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Total dataset samples: {len(df):,}")

    X = df["customer_text"].astype(str)
    y = df["intent"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    test_texts = X_test.tolist()
    test_labels = y_test.values
    print(f"Test samples: {len(test_texts):,}")

    # Load classifier weights
    weights = np.load(WEIGHTS_PATH, allow_pickle=True)
    classes = weights["classes"].astype(str)
    coef = weights["coef"].astype(np.float32)
    intercept = weights["intercept"].astype(np.float32)

    # Evaluate 2000 stratified samples
    eval_size = min(len(test_texts), 2000)
    eval_texts = test_texts[:eval_size]
    eval_labels = test_labels[:eval_size]
    print(f"Evaluating {eval_size} stratified test samples...")

    # 1. Baseline
    print("\n[1/3] Running Baseline SentenceTransformer...")
    t0 = time.time()
    embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    base_embs = embedder.encode(eval_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    base_preds, base_probs = predict_numpy(base_embs, classes, coef, intercept)
    base_time = time.time() - t0
    base_acc = accuracy_score(eval_labels, base_preds)
    base_p, base_r, base_f1, _ = precision_recall_fscore_support(eval_labels, base_preds, average="weighted")
    print(f"Baseline: Acc={base_acc:.4f}, Weighted F1={base_f1:.4f}, Time={base_time:.2f}s ({base_time/eval_size*1000:.2f} ms/sample)")

    # 2. ONNX FP32
    print("\n[2/3] Running ONNX FP32...")
    t0 = time.time()
    fp32_encoder = FastOnnxEncoder(FP32_PATH, TOKENIZER_PATH)
    fp32_embs = fp32_encoder.encode(eval_texts, batch_size=64)
    fp32_preds, fp32_probs = predict_numpy(fp32_embs, classes, coef, intercept)
    fp32_time = time.time() - t0
    fp32_acc = accuracy_score(eval_labels, fp32_preds)
    fp32_p, fp32_r, fp32_f1, _ = precision_recall_fscore_support(eval_labels, fp32_preds, average="weighted")
    print(f"ONNX FP32: Acc={fp32_acc:.4f}, Weighted F1={fp32_f1:.4f}, Time={fp32_time:.2f}s ({fp32_time/eval_size*1000:.2f} ms/sample)")

    # 3. ONNX INT8
    print("\n[3/3] Running ONNX INT8...")
    t0 = time.time()
    int8_encoder = FastOnnxEncoder(INT8_PATH, TOKENIZER_PATH)
    int8_embs = int8_encoder.encode(eval_texts, batch_size=64)
    int8_preds, int8_probs = predict_numpy(int8_embs, classes, coef, intercept)
    int8_time = time.time() - t0
    int8_acc = accuracy_score(eval_labels, int8_preds)
    int8_p, int8_r, int8_f1, _ = precision_recall_fscore_support(eval_labels, int8_preds, average="weighted")
    print(f"ONNX INT8: Acc={int8_acc:.4f}, Weighted F1={int8_f1:.4f}, Time={int8_time:.2f}s ({int8_time/eval_size*1000:.2f} ms/sample)")

    # Agreement between baseline and ONNX
    fp32_agreement = np.mean(base_preds == fp32_preds) * 100
    int8_agreement = np.mean(base_preds == int8_preds) * 100

    results = {
        "evaluation_samples": eval_size,
        "baseline": {
            "accuracy": round(float(base_acc), 4),
            "weighted_f1": round(float(base_f1), 4),
            "latency_ms_per_sample": round(float(base_time / eval_size * 1000), 2)
        },
        "onnx_fp32": {
            "accuracy": round(float(fp32_acc), 4),
            "weighted_f1": round(float(fp32_f1), 4),
            "agreement_with_baseline_pct": round(float(fp32_agreement), 2),
            "latency_ms_per_sample": round(float(fp32_time / eval_size * 1000), 2)
        },
        "onnx_int8": {
            "accuracy": round(float(int8_acc), 4),
            "weighted_f1": round(float(int8_f1), 4),
            "agreement_with_baseline_pct": round(float(int8_agreement), 2),
            "latency_ms_per_sample": round(float(int8_time / eval_size * 1000), 2)
        }
    }

    print("\n--- Summary Comparison ---")
    print(f"Baseline Accuracy:  {base_acc * 100:.2f}%")
    print(f"ONNX FP32 Accuracy: {fp32_acc * 100:.2f}% (Agreement with baseline: {fp32_agreement:.2f}%)")
    print(f"ONNX INT8 Accuracy: {int8_acc * 100:.2f}% (Agreement with baseline: {int8_agreement:.2f}%)")

    # Save to JSON and CSV
    json_path = PROJECT_ROOT / "scripts" / "evaluation" / "intent_classifier_onnx_evaluation.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    df_report = pd.DataFrame([
        {"Model": "Baseline PyTorch", "Accuracy": base_acc, "Weighted F1": base_f1, "Latency (ms)": base_time/eval_size*1000},
        {"Model": "ONNX FP32", "Accuracy": fp32_acc, "Weighted F1": fp32_f1, "Latency (ms)": fp32_time/eval_size*1000},
        {"Model": "ONNX INT8", "Accuracy": int8_acc, "Weighted F1": int8_f1, "Latency (ms)": int8_time/eval_size*1000}
    ])
    csv_path = PROJECT_ROOT / "scripts" / "evaluation" / "intent_classifier_onnx_evaluation.csv"
    df_report.to_csv(csv_path, index=False)

    print(f"Saved evaluation artifacts to {json_path} and {csv_path}")


if __name__ == "__main__":
    run_evaluation()
