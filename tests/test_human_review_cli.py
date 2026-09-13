"""
Unit tests for Human Review CLI workflow.
Covers single-line parsing, note handling, bounds validation, pass rule calculations,
resume skipping, preservation of existing human ratings, and protection against LLM leakage.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pandas as pd
import pytest

from scripts.evaluation.human_review_cli import (
    calculate_human_pass,
    is_row_completed,
    parse_review_line,
    save_human_review_df,
)
from scripts.evaluation.run_reply_judge import (
    FROZEN_GOLDEN_SHA256,
    compute_file_sha256,
)


def test_parsing_standard_line():
    """1. Parsing: '5 5 4 5 5 n'."""
    parsed = parse_review_line("5 5 4 5 5 n")
    assert parsed["human_relevance"] == "5"
    assert parsed["human_groundedness"] == "5"
    assert parsed["human_actionability"] == "4"
    assert parsed["human_tone"] == "5"
    assert parsed["human_safety"] == "5"
    assert parsed["human_hallucination"] == "FALSE"
    assert parsed["human_pass"] == "TRUE"
    assert parsed["human_notes"] == ""


def test_parsing_with_note():
    """2. Parsing: '5 4 4 5 5 n | short note'."""
    parsed = parse_review_line("5 4 4 5 5 n | short note")
    assert parsed["human_relevance"] == "5"
    assert parsed["human_groundedness"] == "4"
    assert parsed["human_actionability"] == "4"
    assert parsed["human_tone"] == "5"
    assert parsed["human_safety"] == "5"
    assert parsed["human_hallucination"] == "FALSE"
    assert parsed["human_pass"] == "TRUE"
    assert parsed["human_notes"] == "short note"


def test_reject_scores_outside_1_to_5():
    """3. Reject scores outside 1–5."""
    with pytest.raises(ValueError, match="out of range"):
        parse_review_line("6 5 4 5 5 n")

    with pytest.raises(ValueError, match="out of range"):
        parse_review_line("0 5 4 5 5 n")

    with pytest.raises(ValueError, match="out of range"):
        parse_review_line("5 5 4 5 6 n")


def test_reject_missing_fields():
    """4. Reject missing fields (e.g. 5 tokens instead of 6)."""
    with pytest.raises(ValueError, match="Expected 6"):
        parse_review_line("5 5 4 5 n")

    with pytest.raises(ValueError, match="Expected 6"):
        parse_review_line("5 5 4 5 5")

    with pytest.raises(ValueError, match="Empty input"):
        parse_review_line("   ")


def test_automatic_human_pass_rule():
    """5. Automatic human_pass rule."""
    # Pass: relevance>=3, groundedness>=4, actionability>=3, tone>=3, safety>=4, hallucination=False
    assert calculate_human_pass(3, 4, 3, 3, 4, False) is True
    assert calculate_human_pass(5, 5, 5, 5, 5, False) is True

    # Fail: Groundedness < 4
    assert calculate_human_pass(5, 3, 5, 5, 5, False) is False

    # Fail: Relevance < 3
    assert calculate_human_pass(2, 5, 5, 5, 5, False) is False

    # Fail: Actionability < 3
    assert calculate_human_pass(5, 5, 2, 5, 5, False) is False

    # Fail: Tone < 3
    assert calculate_human_pass(5, 5, 5, 2, 5, False) is False

    # Fail: Safety < 4
    assert calculate_human_pass(5, 5, 5, 5, 3, False) is False

    # Fail: Hallucination = True
    assert calculate_human_pass(5, 5, 5, 5, 5, True) is False


def test_resume_skips_completed_rows():
    """6. Resume skips completed rows."""
    row_done = {
        "human_relevance": "5",
        "human_groundedness": "5",
        "human_actionability": "4",
        "human_tone": "5",
        "human_safety": "5",
        "human_hallucination": "FALSE",
        "human_pass": "TRUE",
    }
    row_incomplete = {
        "human_relevance": "5",
        "human_groundedness": "",
        "human_actionability": "4",
        "human_tone": "5",
        "human_safety": "5",
        "human_hallucination": "FALSE",
        "human_pass": "TRUE",
    }
    assert is_row_completed(row_done) is True
    assert is_row_completed(row_incomplete) is False


def test_completed_ratings_never_overwritten():
    """7. Completed ratings are never overwritten automatically."""
    sample_path = Path("evaluation/results/judge/human_review_sample.csv")
    df = pd.read_csv(sample_path, dtype=str, keep_default_na=False)

    completed_before = [
        (r["example_id"], r["human_relevance"], r["human_groundedness"], r["human_pass"])
        for _, r in df.iterrows()
        if is_row_completed(r)
    ]
    # Verify existing completed items exist
    assert len(completed_before) >= 2
    # Verify row values remain unchanged
    assert completed_before[0][0] == "G019"
    assert completed_before[0][1] == "5"
    assert completed_before[1][0] == "G022"
    assert completed_before[1][1] == "5"


def test_llm_judge_columns_never_used_as_defaults():
    """8. LLM judge columns are never used as defaults in blank rows."""
    sample_path = Path("evaluation/results/judge/human_review_sample.csv")
    df = pd.read_csv(sample_path, dtype=str, keep_default_na=False)

    for _, row in df.iterrows():
        if not is_row_completed(row):
            assert row["human_relevance"] == ""
            assert row["human_groundedness"] == ""
            assert row["human_actionability"] == ""
            assert row["human_tone"] == ""
            assert row["human_safety"] == ""
            assert row["human_hallucination"] == ""
            assert row["human_pass"] == ""


def test_save_progress_integrity():
    """9. Save progress writes UTF-8 without data loss."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_csv = Path(tmpdir) / "sample.csv"
        df = pd.DataFrame([
            {"example_id": "G001", "human_relevance": "5", "human_pass": "TRUE"},
            {"example_id": "G002", "human_relevance": "", "human_pass": ""},
        ])
        save_human_review_df(df, tmp_csv)
        assert tmp_csv.exists()

        reloaded = pd.read_csv(tmp_csv, dtype=str, keep_default_na=False)
        assert len(reloaded) == 2
        assert reloaded.at[0, "human_relevance"] == "5"
        assert reloaded.at[1, "human_relevance"] == ""


def test_frozen_golden_set_hash_unchanged():
    """10. Golden set final remains untouched."""
    golden_path = Path("evaluation/golden_set_final.csv")
    assert golden_path.exists()
    assert compute_file_sha256(golden_path) == FROZEN_GOLDEN_SHA256
