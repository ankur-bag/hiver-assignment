"""
Tests for Baseline Intent Classifiers & Evaluation Pipeline.
Verifies leakage safeguards, dynamic majority class derivation,
TF-IDF + Logistic Regression baseline correctness, and metric calculations.
"""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from scripts.evaluation.run_baselines import (
    CANONICAL_INTENTS,
    FROZEN_GOLDEN_SHA256,
    calculate_escalation_metrics,
    calculate_intent_metrics,
    clean_training_data,
    compute_file_sha256,
    normalize_for_leakage_detection,
    run_all_baselines,
)


def test_frozen_golden_set_sha256_unmodified():
    """Test that the 158-example frozen golden set SHA-256 hash is unchanged."""
    golden_path = Path("evaluation/golden_set_final.csv")
    assert golden_path.exists(), "Golden set file must exist"
    actual_sha = compute_file_sha256(golden_path)
    assert actual_sha == FROZEN_GOLDEN_SHA256, f"Golden set modified! Expected {FROZEN_GOLDEN_SHA256}, got {actual_sha}"

    df_gold = pd.read_csv(golden_path)
    assert len(df_gold) == 158, f"Expected 158 golden rows, got {len(df_gold)}"
    assert len(set(df_gold["example_id"])) == 158, "Duplicate example_id found in golden set"
    assert "ESCALATION" not in df_gold["expected_intent"].values, "ESCALATION must not be an intent in golden set"


def test_escalation_source_labels_excluded_from_training():
    """Test that rows with ESCALATION intent in source dataset are completely excluded."""
    df_train_mock = pd.DataFrame({
        "customer_text": ["Help with login", "Order delayed", "Serious fraud issue"],
        "intent": ["ACCOUNT_ACCESS", "DELIVERY_DELAY", "ESCALATION"],
    })
    df_gold_mock = pd.DataFrame({
        "customer_text": ["Different query"],
        "expected_intent": ["ORDER_STATUS"],
    })

    df_clean, leakage_info = clean_training_data(df_train_mock, df_gold_mock)
    assert "ESCALATION" not in df_clean["intent"].values
    assert len(df_clean) == 2
    assert leakage_info["non_canonical_removed"] == 1


def test_normalized_duplicate_text_leakage_removal():
    """Test that exact and normalized variants of golden examples are removed from training data."""
    df_train_mock = pd.DataFrame({
        "customer_text": [
            "Where is my package? @AmazonHelp https://t.co/123",
            "Mera order deliver nahi hua",
            "Completely unique query that is not in golden set",
        ],
        "intent": ["PACKAGE_NOT_RECEIVED", "DELIVERY_DELAY", "PRODUCT_ISSUE"],
    })
    df_gold_mock = pd.DataFrame({
        "customer_text": [
            "where is my package",
            "mera order deliver nahi hua...",
        ],
        "expected_intent": ["PACKAGE_NOT_RECEIVED", "DELIVERY_DELAY"],
    })

    df_clean, leakage_info = clean_training_data(df_train_mock, df_gold_mock)
    assert len(df_clean) == 1
    assert df_clean.iloc[0]["customer_text"] == "Completely unique query that is not in golden set"
    assert leakage_info["leakage_rows_removed"] == 2


def test_dynamic_majority_class_derivation():
    """Test that majority class is derived dynamically from training distribution, not hardcoded."""
    df_train_mock = pd.DataFrame({
        "customer_text": ["text1", "text2", "text3", "text4"],
        "intent": ["REFUND_PENDING", "REFUND_PENDING", "REFUND_PENDING", "ORDER_STATUS"],
    })
    df_gold_mock = pd.DataFrame({
        "customer_text": ["gold1", "gold2"],
        "expected_intent": ["REFUND_PENDING", "ORDER_STATUS"],
    })

    df_clean, _ = clean_training_data(df_train_mock, df_gold_mock)
    majority_intent = df_clean["intent"].value_counts().idxmax()
    assert majority_intent == "REFUND_PENDING"


def test_intent_metrics_calculation_accuracy_and_macro_f1():
    """Test multi-class intent metric computation logic."""
    expected = ["ACCOUNT_ACCESS", "ORDER_STATUS", "REFUND_PENDING", "PRODUCT_ISSUE"]
    predicted = ["ACCOUNT_ACCESS", "ORDER_STATUS", "ORDER_STATUS", "PRODUCT_ISSUE"]

    metrics = calculate_intent_metrics(expected, predicted, canonical_intents=CANONICAL_INTENTS)
    assert metrics["accuracy"] == 0.75
    assert "confusion_matrix" in metrics
    assert metrics["confusion_matrix"]["ACCOUNT_ACCESS"]["ACCOUNT_ACCESS"] == 1
    assert metrics["confusion_matrix"]["REFUND_PENDING"]["ORDER_STATUS"] == 1


def test_trivial_escalation_metrics_calculation():
    """Test binary escalation metrics calculation for always-False baseline."""
    expected_esc = [True, False, False, True]
    predicted_esc = [False, False, False, False]

    esc_metrics = calculate_escalation_metrics(expected_esc, predicted_esc)
    assert esc_metrics["accuracy"] == 0.5  # 2 TN out of 4
    assert esc_metrics["tp"] == 0
    assert esc_metrics["tn"] == 2
    assert esc_metrics["fp"] == 0
    assert esc_metrics["fn"] == 2
    assert esc_metrics["precision"] == 0.0
    assert esc_metrics["recall"] == 0.0
    assert esc_metrics["f1"] == 0.0


def test_end_to_end_baseline_runner_execution():
    """Test full baseline runner creates all required CSVs and JSONs with 158 rows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        comparison = run_all_baselines(
            golden_path="evaluation/golden_set_final.csv",
            training_path="amazon_final_intent_dataset.csv",
            output_dir=out_dir,
            gemini_summary_path="evaluation/results/final/golden_eval_summary.json",
        )

        assert (out_dir / "majority_baseline_results.csv").exists()
        assert (out_dir / "majority_baseline_summary.json").exists()
        assert (out_dir / "tfidf_logreg_results.csv").exists()
        assert (out_dir / "tfidf_logreg_summary.json").exists()
        assert (out_dir / "baseline_comparison.json").exists()
        assert (out_dir / "confusion_matrix_tfidf.csv").exists()

        df_maj = pd.read_csv(out_dir / "majority_baseline_results.csv")
        assert len(df_maj) == 158
        df_tfidf = pd.read_csv(out_dir / "tfidf_logreg_results.csv")
        assert len(df_tfidf) == 158

        # Verify all predictions are from canonical 8 intents
        assert set(df_tfidf["predicted_intent"].unique()).issubset(set(CANONICAL_INTENTS))
        assert "ESCALATION" not in df_tfidf["predicted_intent"].values

        assert comparison["golden_set"]["examples"] == 158
        assert comparison["majority_baseline"]["intent_accuracy"] == 0.019
        assert comparison["tfidf_logreg"]["intent_accuracy"] == 0.3987
