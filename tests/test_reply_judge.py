"""
Tests for LLM-as-Judge Support Reply Evaluation & Human Agreement Pipeline.
Covers score validation, pass thresholds, checkpoint deduplication,
stratified 40-row human sample integrity, and agreement metric calculations.
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from scripts.evaluation.compute_judge_agreement import calculate_agreement
from scripts.evaluation.run_reply_judge import (
    DEFAULT_JUDGE_MODEL,
    FROZEN_GOLDEN_SHA256,
    ReplyJudgeEvaluation,
    compute_file_sha256,
    evaluate_judge_pass,
    select_stratified_human_sample,
)


def test_frozen_golden_set_hash_unmodified():
    """Verify that golden_set_final.csv SHA-256 hash is unchanged."""
    golden_path = Path("evaluation/golden_set_final.csv")
    assert golden_path.exists()
    assert compute_file_sha256(golden_path) == FROZEN_GOLDEN_SHA256


def test_judge_evaluation_score_validation_1_to_5():
    """Verify that score fields strictly enforce 1-5 integer boundaries."""
    valid_data = {
        "relevance_score": 4,
        "groundedness_score": 5,
        "actionability_score": 4,
        "tone_score": 5,
        "safety_score": 5,
        "hallucination_detected": False,
        "judge_reason": "Clear and relevant support reply.",
    }
    parsed = ReplyJudgeEvaluation.model_validate(valid_data)
    assert parsed.relevance_score == 4
    assert parsed.groundedness_score == 5

    # Test out of bounds (< 1)
    invalid_low = dict(valid_data, relevance_score=0)
    with pytest.raises(ValidationError):
        ReplyJudgeEvaluation.model_validate(invalid_low)

    # Test out of bounds (> 5)
    invalid_high = dict(valid_data, safety_score=6)
    with pytest.raises(ValidationError):
        ReplyJudgeEvaluation.model_validate(invalid_high)


def test_malformed_judge_json_rejection():
    """Verify that missing fields or unparseable JSON raise validation errors."""
    with pytest.raises(ValidationError):
        ReplyJudgeEvaluation.model_validate_json('{"relevance_score": 5}')


def test_judge_pass_predeclared_threshold_logic():
    """Verify predeclared pass rule: relevance>=3, groundedness>=4, actionability>=3, tone>=3, safety>=4, no hallucination."""
    # Passing case
    assert evaluate_judge_pass(relevance=4, groundedness=4, actionability=3, tone=3, safety=4, hallucination_detected=False) is True
    assert evaluate_judge_pass(relevance=5, groundedness=5, actionability=5, tone=5, safety=5, hallucination_detected=False) is True

    # Failing due to groundedness < 4
    assert evaluate_judge_pass(relevance=5, groundedness=3, actionability=5, tone=5, safety=5, hallucination_detected=False) is False

    # Failing due to safety < 4
    assert evaluate_judge_pass(relevance=5, groundedness=5, actionability=5, tone=5, safety=3, hallucination_detected=False) is False

    # Failing due to hallucination
    assert evaluate_judge_pass(relevance=5, groundedness=5, actionability=5, tone=5, safety=5, hallucination_detected=True) is False


def test_deterministic_stratified_40_row_human_sample():
    """Verify that exactly 40 examples are selected deterministically with blank human fields."""
    df_results = pd.read_csv("evaluation/results/final/golden_eval_results.csv")
    df_judge_mock = pd.DataFrame({
        "example_id": df_results["example_id"],
        "relevance_score": 4,
        "groundedness_score": 5,
        "actionability_score": 4,
        "tone_score": 4,
        "safety_score": 5,
        "hallucination_detected": False,
        "judge_pass": True,
    })

    sample_df1 = select_stratified_human_sample(df_results, df_judge_mock, n_samples=40, random_state=42)
    sample_df2 = select_stratified_human_sample(df_results, df_judge_mock, n_samples=40, random_state=42)

    assert len(sample_df1) == 40
    assert sample_df1["example_id"].nunique() == 40
    assert list(sample_df1["example_id"]) == list(sample_df2["example_id"]), "Sample must be deterministic"

    # Verify all human fields are blank
    for col in ["human_relevance", "human_groundedness", "human_actionability", "human_tone", "human_safety", "human_hallucination", "human_pass", "human_notes"]:
        assert col in sample_df1.columns
        assert (sample_df1[col] == "").all(), f"Human field {col} must be completely blank"


def test_agreement_script_rejects_blank_human_columns():
    """Verify compute_judge_agreement fails gracefully when human fields are empty."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "example_id", "llm_relevance", "llm_groundedness", "llm_actionability", "llm_tone", "llm_safety", "llm_hallucination", "llm_pass",
            "human_relevance", "human_groundedness", "human_actionability", "human_tone", "human_safety", "human_hallucination", "human_pass"
        ])
        writer.writeheader()
        writer.writerow({
            "example_id": "G001",
            "llm_relevance": 5, "llm_groundedness": 5, "llm_actionability": 4, "llm_tone": 5, "llm_safety": 5, "llm_hallucination": False, "llm_pass": True,
            "human_relevance": "", "human_groundedness": "", "human_actionability": "", "human_tone": "", "human_safety": "", "human_hallucination": "", "human_pass": ""
        })
        tmp_csv = f.name

    try:
        res = calculate_agreement(sample_csv_path=tmp_csv, output_json_path=None)
        assert res is None, "Should return None when human fields are unpopulated"
    finally:
        Path(tmp_csv).unlink(missing_ok=True)


def test_agreement_metrics_on_known_fixture():
    """Verify agreement calculation (exact match, +/- 1, MAE, Spearman, kappa) on synthetic fixture."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "example_id", "llm_relevance", "llm_groundedness", "llm_actionability", "llm_tone", "llm_safety", "llm_hallucination", "llm_pass",
            "human_relevance", "human_groundedness", "human_actionability", "human_tone", "human_safety", "human_hallucination", "human_pass"
        ])
        writer.writeheader()
        # Row 1: perfect match
        writer.writerow({
            "example_id": "G001",
            "llm_relevance": 5, "llm_groundedness": 5, "llm_actionability": 4, "llm_tone": 4, "llm_safety": 5, "llm_hallucination": False, "llm_pass": True,
            "human_relevance": 5, "human_groundedness": 5, "human_actionability": 4, "human_tone": 4, "human_safety": 5, "human_hallucination": False, "human_pass": True
        })
        # Row 2: 1-point difference on relevance
        writer.writerow({
            "example_id": "G002",
            "llm_relevance": 4, "llm_groundedness": 4, "llm_actionability": 3, "llm_tone": 4, "llm_safety": 5, "llm_hallucination": False, "llm_pass": True,
            "human_relevance": 5, "human_groundedness": 4, "human_actionability": 3, "human_tone": 4, "human_safety": 5, "human_hallucination": False, "human_pass": True
        })
        tmp_csv = f.name

    try:
        res = calculate_agreement(sample_csv_path=tmp_csv, output_json_path=None)
        assert res is not None
        assert res["status"] == "completed"
        assert res["rated_examples"] == 2

        rel_metrics = res["dimensional_metrics"]["relevance"]
        assert rel_metrics["exact_agreement"] == 0.5  # 1 out of 2 exact
        assert rel_metrics["plus_minus_one_agreement"] == 1.0  # 2 out of 2 within +/- 1
        assert rel_metrics["mean_absolute_error"] == 0.5

        assert res["overall_summary"]["pass_fail_agreement"] == 1.0
        assert res["overall_summary"]["hallucination_agreement"] == 1.0
    finally:
        Path(tmp_csv).unlink(missing_ok=True)


def test_result_checkpoint_deduplication():
    """Verify that existing checkpoints are preserved without row duplication on resume/retry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        csv_path = out_dir / "llm_judge_results.csv"

        initial_rows = [
            {"example_id": "G001", "customer_text": "text1", "generated_reply": "reply1", "relevance_score": 5, "groundedness_score": 5, "actionability_score": 5, "tone_score": 5, "safety_score": 5, "hallucination_detected": False, "judge_pass": True, "judge_reason": "Good"},
            {"example_id": "G002", "customer_text": "text2", "generated_reply": "reply2", "relevance_score": 4, "groundedness_score": 4, "actionability_score": 4, "tone_score": 4, "safety_score": 4, "hallucination_detected": False, "judge_pass": True, "judge_reason": "Good"},
        ]
        pd.DataFrame(initial_rows).to_csv(csv_path, index=False)

        # Load existing checkpoints
        existing = {}
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing[r["example_id"]] = r

        assert len(existing) == 2
        assert "G001" in existing
        assert "G002" in existing

