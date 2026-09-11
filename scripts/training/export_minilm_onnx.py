"""
Export paraphrase-multilingual-MiniLM-L12-v2 to ONNX FP32 format.
Exports the underlying transformer model and saves tokenizer configuration
to backend/ml/models/onnx/ directory.
"""

import os
import sys
from pathlib import Path

# Fix Windows console encoding for Unicode logging
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import torch
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "backend" / "ml" / "models" / "onnx"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class MiniLMEncoderWrapper(torch.nn.Module):
    """Wrapper to clean signature for ONNX export."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is not None:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        else:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return out.last_hidden_state


def export_to_onnx():
    print(f"=== Exporting {MODEL_NAME} to ONNX ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_model_path = OUTPUT_DIR / "model.onnx"

    print(f"Loading tokenizer and PyTorch model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModel.from_pretrained(MODEL_NAME)
    wrapper_model = MiniLMEncoderWrapper(base_model)
    wrapper_model.eval()

    # Save tokenizer files directly for lightweight runtime (without torch)
    print(f"Saving tokenizer assets to {OUTPUT_DIR}...")
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Dummy input for tracing
    dummy_text = ["This is a test query for ONNX export.", "Mera order kahan hai?"]
    dummy_inputs = tokenizer(
        dummy_text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    input_names = ["input_ids", "attention_mask"]
    input_tuple = (dummy_inputs["input_ids"], dummy_inputs["attention_mask"])
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "last_hidden_state": {0: "batch_size", 1: "sequence_length"}
    }

    if "token_type_ids" in dummy_inputs:
        input_names.append("token_type_ids")
        input_tuple = (dummy_inputs["input_ids"], dummy_inputs["attention_mask"], dummy_inputs["token_type_ids"])
        dynamic_axes["token_type_ids"] = {0: "batch_size", 1: "sequence_length"}

    output_names = ["last_hidden_state"]

    print(f"Exporting PyTorch model to ONNX: {onnx_model_path}...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper_model,
            input_tuple,
            str(onnx_model_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
            dynamo=False
        )

    fp32_size_mb = onnx_model_path.stat().st_size / (1024 * 1024)
    print(f"Successfully exported FP32 ONNX model ({fp32_size_mb:.2f} MB) to {onnx_model_path}")


if __name__ == "__main__":
    export_to_onnx()
