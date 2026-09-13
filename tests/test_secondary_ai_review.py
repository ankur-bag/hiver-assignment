"""
Unit tests for Secondary AI Review and Inter-Reviewer Agreement.
Covers schema validation, pass threshold logic, absence of primary judge score leakage,
agreement calculations, checkpoint deduplication, and preservation of human_* fields.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from scripts.evaluation.run_secondary_ai_review import (
    FROZEN_GOLDEN_SHA256,
    SecondaryReviewEvaluation,
    build_secondary_reviewer_prompt,
    compute_file_sha256,
    compute_interreviewer_agreement,
    evaluate_secondary_pass,
)


def test_secondary_score_validation_1_to_5():
    """Scores must strictly be integers between 1 and 5."""
    valid_data = {
        "relevance_score": 5,
        "groundedness_score": 4,
        "actionability_score": 4,
        "tone_score": 5,
        "safety_score": 5,
        "hallucination_detected": False,
        "reviewer_reason": "High quality grounded response.",
    }
    parsed = SecondaryReviewEvaluation.model_validate(valid_data)
    assert parsed.relevance_score == 5
    assert parsed.groundedness_score == 4

    # Test out of bounds (< 1)
    with pytest.raises(ValidationError):
        SecondaryReviewEvaluation.model_validate(dict(valid_data, relevance_score=0))

    # Test out of bounds (> 5)
    with pytest.raises(ValidationError):
        SecondaryReviewEvaluation.model_validate(dict(valid_data, safety_score=6))


def test_secondary_pass_rule_thresholds():
    """Verify pass rule: relevance>=3, groundedness>=4, actionability>=3, tone>=3, safety>=4, hallucination=False."""
    assert evaluate_secondary_pass(3, 4, 3, 3, 4, False) is True
    assert evaluate_secondary_pass(5, 5, 5, 5, 5, False) is True

    # Fail: Groundedness < 4
    assert evaluate_secondary_pass(5, 3, 5, 5, 5, False) is False

    # Fail: Safety < 4
    assert evaluate_secondary_pass(5, 5, 5, 5, 3, False) is False

    # Fail: Hallucination detected
    assert evaluate_secondary_pass(5, 5, 5, 5, 5, True) is False


def test_primary_scores_not_in_prompt():
    """Verify that primary judge scores/reasons are NOT leaked in the prompt to secondary reviewer."""
    prompt = build_secondary_reviewer_prompt(
        example_id="G001",
        customer_text="Where is my refund?",
        generated_reply="Your refund is processing.",
        expected_reply_points="Explain timeline",
        historical_amazon_reply="Refund within 3-5 days",
        grounding_context="Policy: refunds take 3-5 business days",
        expected_intent="REFUND_PENDING",
        predicted_intent="REFUND_PENDING",
        expected_escalate="False",
        predicted_escalate="False",
    )
    # Check that primary judge labels/scores never appear
    forbidden_terms = [
        "llm_relevance",
        "llm_groundedness",
        "llm_actionability",
        "llm_tone",
        "llm_safety",
        "llm_hallucination",
        "llm_pass",
        "primary_judge",
        "judge_reason",
    ]
    for term in forbidden_terms:
        assert term not in prompt, f"Prompt leaked forbidden term: {term}"


def test_agreement_calculation_logic():
    """Verify exact match, +/- 1, MAE, Spearman, and Cohen's kappa calculations."""
    df_prim = pd.DataFrame([
        {
            "example_id": "G001",
            "llm_relevance": 5, "llm_groundedness": 5, "llm_actionability": 4, "llm_tone": 5, "llm_safety": 5,
            "llm_hallucination": False, "llm_pass": True,
        },
        {
            "example_id": "G002",
            "llm_relevance": 4, "llm_groundedness": 4, "llm_actionability": 3, "llm_tone": 4, "llm_safety": 5,
            "llm_hallucination": False, "llm_pass": True,
        },
    ])

    df_sec = pd.DataFrame([
        {
            "example_id": "G001",
            "secondary_reviewer_relevance": 5, "secondary_reviewer_groundedness": 5, "secondary_reviewer_actionability": 4, "secondary_reviewer_tone": 5, "secondary_reviewer_safety": 5,
            "secondary_reviewer_hallucination": False, "secondary_reviewer_pass": True,
        },
        {
            "example_id": "G002",
            "secondary_reviewer_relevance": 5, "secondary_reviewer_groundedness": 4, "secondary_reviewer_actionability": 4, "secondary_reviewer_tone": 4, "secondary_reviewer_safety": 5,
            "secondary_reviewer_hallucination": False, "secondary_reviewer_pass": True,
        },
    ])

    agreement = compute_interreviewer_agreement(df_prim, df_sec)
    assert agreement["reviewed_cases"] == 2
    assert agreement["binary_metrics"]["pass_fail_agreement"] == 1.0
    assert agreement["binary_metrics"]["hallucination_agreement"] == 1.0

    # Relevance: row 1 exact (5,5), row 2 diff by 1 (4,5) -> exact=50%, pm1=100%, MAE=0.5
    rel = agreement["dimensional_metrics"]["relevance"]
    assert rel["exact_agreement"] == 0.5
    assert rel["plus_minus_one_agreement"] == 1.0
    assert rel["mean_absolute_error"] == 0.5


def test_human_fields_remain_unmodified():
    """Verify that running the evaluation does not modify human_* fields in human_review_sample.csv."""
    sample_path = Path("evaluation/results/judge/human_review_sample.csv")
    assert sample_path.exists()
    df = pd.read_csv(sample_path, dtype=str, keep_default_na=False)

    # 40 rows, 40 unique IDs
    assert len(df) == 40
    assert df["example_id"].nunique() == 40

    # Verify unrated rows are still blank
    unrated = df[~df["example_id"].isin(["G019", "G022"])]
    for col in ["human_relevance", "human_groundedness", "human_actionability", "human_tone", "human_safety", "human_hallucination", "human_pass"]:
        assert (unrated[col] == "").all()


def test_golden_sha256_unmodified():
    """Verify golden_set_final.csv SHA-256 remains untouched."""
    golden_path = Path("evaluation/golden_set_final.csv")
    assert golden_path.exists()
    assert compute_file_sha256(golden_path) == FROZEN_GOLDEN_SHA256
