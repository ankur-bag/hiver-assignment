"""
Dynamic INT8 Quantization Script for MiniLM-L12 ONNX model.
Quantizes FP32 weights to INT8 to reduce disk footprint from ~450MB to ~115MB
and reduce runtime inference memory.
"""

import os
import sys
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ONNX_DIR = PROJECT_ROOT / "backend" / "ml" / "models" / "onnx"
INPUT_MODEL = ONNX_DIR / "model.onnx"
OUTPUT_MODEL = ONNX_DIR / "model_quantized.onnx"


def quantize_model():
    print(f"=== Starting INT8 Dynamic Quantization ===")
    if not INPUT_MODEL.exists():
        raise FileNotFoundError(f"Input ONNX model not found at {INPUT_MODEL}")

    print(f"Quantizing {INPUT_MODEL} -> {OUTPUT_MODEL}...")
    quantize_dynamic(
        model_input=str(INPUT_MODEL),
        model_output=str(OUTPUT_MODEL),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True
    )

    orig_size_mb = INPUT_MODEL.stat().st_size / (1024 * 1024)
    quant_size_mb = OUTPUT_MODEL.stat().st_size / (1024 * 1024)
    reduction = (1.0 - quant_size_mb / orig_size_mb) * 100
    print(f"Quantization complete!")
    print(f"FP32 Model: {orig_size_mb:.2f} MB")
    print(f"INT8 Model: {quant_size_mb:.2f} MB ({reduction:.1f}% reduction)")


if __name__ == "__main__":
    quantize_model()
