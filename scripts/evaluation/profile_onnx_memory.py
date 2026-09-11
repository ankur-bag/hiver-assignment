"""
Memory Profiling Script for Production ONNX Inference Pipeline.
Measures process RSS memory across imports, ONNX loading, single prediction,
and 10 sequential inference requests.
"""

import os
import sys
import json
from pathlib import Path

# Add backend to sys.path
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "backend"))


def get_rss_mb() -> float:
    """Returns current process Resident Set Size (RSS) in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        import ctypes
        if sys.platform == "win32":
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), ctypes.sizeof(counters)):
                return counters.WorkingSetSize / (1024 * 1024)
        return 0.0


def profile():
    results = {}

    m0 = get_rss_mb()
    results["0_initial_rss_mb"] = round(m0, 2)
    print(f"[Step 0] Initial RSS: {m0:.2f} MB")

    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer
    m1 = get_rss_mb()
    results["1_after_onnx_imports_mb"] = round(m1, 2)
    print(f"[Step 1] After numpy, onnxruntime, tokenizers imports: {m1:.2f} MB (+{m1 - m0:.2f} MB)")

    from ml.embeddings.encoder import get_embedding_encoder
    encoder = get_embedding_encoder()
    m2 = get_rss_mb()
    results["2_after_onnx_model_load_mb"] = round(m2, 2)
    print(f"[Step 2] After ONNX model & tokenizer load: {m2:.2f} MB (+{m2 - m1:.2f} MB)")

    emb = encoder.encode("Where is my package? It has not arrived yet.")
    m3 = get_rss_mb()
    results["3_after_first_embedding_mb"] = round(m3, 2)
    print(f"[Step 3] After first ONNX embedding generation: {m3:.2f} MB (+{m3 - m2:.2f} MB)")

    from ml.inference import get_intent_classifier
    classifier = get_intent_classifier()
    m4 = get_rss_mb()
    results["4_after_classifier_load_mb"] = round(m4, 2)
    print(f"[Step 4] After NumPy classifier load: {m4:.2f} MB (+{m4 - m3:.2f} MB)")

    pred = classifier.predict("Where is my package? It has not arrived yet.", include_metadata=True)
    m5 = get_rss_mb()
    results["5_after_first_prediction_mb"] = round(m5, 2)
    print(f"[Step 5] After first prediction: {m5:.2f} MB (+{m5 - m4:.2f} MB)")
    print(f"Sample prediction output: {pred['intent']} (confidence: {pred['confidence']})")

    # 10 Sequential requests
    queries = [
        "My package has not arrived yet",
        "Where is my refund?",
        "I cannot access my account",
        "My order is delayed",
        "Someone accessed my account",
        "Mera parcel abhi tak nahi aaya",
        "Necesito ayuda con mi pedido",
        "Product is defective and broken",
        "I want to speak with a human agent",
        "Cancel my subscription immediately"
    ]
    for q in queries:
        classifier.predict(q)

    m6 = get_rss_mb()
    results["6_after_10_sequential_predictions_mb"] = round(m6, 2)
    print(f"[Step 6] After 10 sequential predictions: {m6:.2f} MB (+{m6 - m5:.2f} MB)")

    total_growth = m6 - m0
    results["total_growth_mb"] = round(total_growth, 2)
    results["peak_rss_mb"] = round(m6, 2)
    print(f"\n=== Memory Optimization Result ===")
    print(f"Total Memory Growth: {total_growth:.2f} MB | Peak RSS: {m6:.2f} MB")
    print(f"Headroom under 512 MiB: {512 - m6:.2f} MB")

    output_json = BASE_DIR / "scripts" / "evaluation" / "optimized_memory_profile.json"
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved optimized memory profile to {output_json}")


if __name__ == "__main__":
    profile()
