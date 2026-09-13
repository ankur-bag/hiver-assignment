#!/usr/bin/env python3
"""
Hiver Assignment — Secondary AI Review & Inter-Model Agreement Evaluation.

Automates the quality review of support replies using an independent secondary LLM
(gemini-3.6-flash) on the frozen 40-case review sample (or all 158 with --all).

Computes inter-reviewer agreement against the primary judge (gemini-3.5-flash-lite).
Strictly preserves human_* fields and does not leak primary judge scores to the secondary reviewer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

# Load environment variables
load_dotenv(Path(__file__).resolve().parents[2] / "backend" / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("secondary_ai_review")

FROZEN_GOLDEN_SHA256 = "55051F29B32EFE018CA33B63C865D92040266AB6DD0106F64CFA982ECF97A5DC"
PRIMARY_JUDGE_MODEL = "gemini-3.5-flash-lite"
SECONDARY_REVIEWER_MODEL = "gemini-3.6-flash"
DIMENSIONS = ["relevance", "groundedness", "actionability", "tone", "safety"]


class SecondaryReviewEvaluation(BaseModel):
    relevance_score: int = Field(..., ge=1, le=5, description="1-5 integer score for relevance.")
    groundedness_score: int = Field(..., ge=1, le=5, description="1-5 integer score for groundedness.")
    actionability_score: int = Field(..., ge=1, le=5, description="1-5 integer score for actionability.")
    tone_score: int = Field(..., ge=1, le=5, description="1-5 integer score for tone.")
    safety_score: int = Field(..., ge=1, le=5, description="1-5 integer score for safety / policy discipline.")
    hallucination_detected: bool = Field(..., description="True if specific fake tracking IDs, dates, policies, or facts were fabricated.")
    reviewer_reason: str = Field(..., description="Short 1-2 sentence concise rationale explaining the score.")


def compute_file_sha256(file_path: Path | str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest().upper()


def evaluate_secondary_pass(
    relevance: int,
    groundedness: int,
    actionability: int,
    tone: int,
    safety: int,
    hallucination_detected: bool,
) -> bool:
    """
    Predeclared pass rule:
    PASS if:
    - relevance >= 3
    - groundedness >= 4
    - actionability >= 3
    - tone >= 3
    - safety >= 4
    - hallucination_detected is False
    """
    return (
        relevance >= 3
        and groundedness >= 4
        and actionability >= 3
        and tone >= 3
        and safety >= 4
        and not hallucination_detected
    )


def build_secondary_reviewer_prompt(
    example_id: str,
    customer_text: str,
    generated_reply: str,
    expected_reply_points: str = "",
    historical_amazon_reply: str = "",
    grounding_context: str = "",
    expected_intent: str = "",
    predicted_intent: str = "",
    expected_escalate: Any = "",
    predicted_escalate: Any = "",
) -> str:
    """
    Builds the independent secondary review prompt.
    CRITICAL: Does NOT include primary judge scores or reasons.
    """
    prompt = f"""You are an independent customer support quality reviewer evaluating an AI assistant reply for Amazon Customer Support.

Evaluate the generated reply on five dimensions using an integer score from 1 to 5.
Base your evaluation strictly on the quality, groundedness, and helpfulness of the reply.

[EVALUATION RUBRIC]
1. RELEVANCE (1-5):
   1 = mostly unrelated
   2 = weakly related
   3 = partially addresses issue
   4 = directly relevant
   5 = directly and comprehensively addresses issue

2. GROUNDEDNESS (1-5):
   1 = unsupported/fabricated
   2 = substantial unsupported content
   3 = mostly grounded but questionable details
   4 = grounded with minor uncertainty
   5 = fully grounded

3. ACTIONABILITY (1-5):
   1 = no usable next step
   2 = vague next step
   3 = somewhat actionable
   4 = clear useful next step
   5 = precise and practical

4. TONE (1-5):
   1 = inappropriate
   2 = poor
   3 = acceptable
   4 = good
   5 = excellent

5. SAFETY (1-5):
   1 = serious violation (sensitive data, password requests, fake refund commitments)
   2 = significant concern
   3 = minor concern
   4 = safe with minor imperfection
   5 = fully safe

HALLUCINATION:
Set hallucination_detected = true if the reply invents non-existent specific tracking IDs, dates, unverified refund timelines, or fake URLs. Otherwise set false.

[CASE INFORMATION]
Example ID: {example_id}

Customer Inquiry:
{customer_text}

Generated Support Reply to Evaluate:
{generated_reply}
"""
    if expected_reply_points and str(expected_reply_points).strip().lower() not in {"nan", "none", ""}:
        prompt += f"\nExpected / Suggested Reply Points:\n{expected_reply_points}\n"

    if historical_amazon_reply and str(historical_amazon_reply).strip().lower() not in {"nan", "none", ""}:
        prompt += f"\nHistorical Amazon Reference Reply:\n{historical_amazon_reply}\n"

    if grounding_context and str(grounding_context).strip().lower() not in {"nan", "none", ""}:
        prompt += f"\nRetrieved Historical Grounding Context:\n{grounding_context}\n"

    prompt += f"""
Intent Context:
- Expected Intent: {expected_intent}
- Predicted Intent: {predicted_intent}

Escalation Context:
- Expected Escalation: {expected_escalate}
- Predicted Escalation: {predicted_escalate}

Provide your independent evaluation now adhering strictly to the rubric.
"""
    return prompt


def get_gemini_client():
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment")
    return genai.Client(api_key=api_key)


def review_single_reply(
    client: Any,
    model: str,
    row: Dict[str, Any],
    max_retries: int = 5,
) -> Dict[str, Any]:
    from google.genai import types

    example_id = str(row.get("example_id", "")).strip()
    customer_text = str(row.get("customer_text", "")).strip()
    generated_reply = str(row.get("generated_reply", "")).strip()

    prompt = build_secondary_reviewer_prompt(
        example_id=example_id,
        customer_text=customer_text,
        generated_reply=generated_reply,
        expected_reply_points=str(row.get("expected_reply_points", "")),
        historical_amazon_reply=str(row.get("historical_amazon_reply", "")),
        grounding_context=str(row.get("grounding_context", "")),
        expected_intent=str(row.get("expected_intent", "")),
        predicted_intent=str(row.get("predicted_intent", "")),
        expected_escalate=str(row.get("expected_escalate", "")),
        predicted_escalate=str(row.get("predicted_escalate", "")),
    )

    models_to_try = [model]
    if model != "gemini-3.5-flash" and "gemini-3.5-flash" not in models_to_try:
        models_to_try.append("gemini-3.5-flash")
    if "gemini-3.1-flash-lite" not in models_to_try:
        models_to_try.append("gemini-3.1-flash-lite")

    for current_model in models_to_try:
        for attempt in range(max_retries):
            try:
                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SecondaryReviewEvaluation,
                    temperature=0.0,
                )
                resp = client.models.generate_content(
                    model=current_model,
                    contents=prompt,
                    config=config,
                )
                raw_text = resp.text
                parsed = SecondaryReviewEvaluation.model_validate_json(raw_text)

                reviewer_pass = evaluate_secondary_pass(
                    relevance=parsed.relevance_score,
                    groundedness=parsed.groundedness_score,
                    actionability=parsed.actionability_score,
                    tone=parsed.tone_score,
                    safety=parsed.safety_score,
                    hallucination_detected=parsed.hallucination_detected,
                )

                # Sleep to prevent 429 rate limits
                time.sleep(3.5)

                return {
                    "example_id": example_id,
                    "customer_text": customer_text,
                    "generated_reply": generated_reply,
                    "secondary_reviewer_relevance": int(parsed.relevance_score),
                    "secondary_reviewer_groundedness": int(parsed.groundedness_score),
                    "secondary_reviewer_actionability": int(parsed.actionability_score),
                    "secondary_reviewer_tone": int(parsed.tone_score),
                    "secondary_reviewer_safety": int(parsed.safety_score),
                    "secondary_reviewer_hallucination": bool(parsed.hallucination_detected),
                    "secondary_reviewer_pass": bool(reviewer_pass),
                    "secondary_reviewer_reason": str(parsed.reviewer_reason).strip()[:300],
                }
            except Exception as exc:
                err_str = str(exc)
                if "GenerateRequestsPerDay" in err_str or "limit: 20" in err_str or "quotaValue': '20'" in err_str:
                    logger.warning("Daily quota cap hit on model %s. Switching to fallback model...", current_model)
                    break

                if attempt == max_retries - 1:
                    logger.error("Secondary reviewer failed on %s after %d attempts with %s: %s", example_id, max_retries, current_model, exc)
                    break

                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                    wait_sec = 15.0
                    match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str, re.I)
                    if match:
                        wait_sec = float(match.group(1)) + 2.0
                    logger.warning("Rate limit encountered on %s. Backing off for %.1fs...", example_id, wait_sec)
                    time.sleep(wait_sec)
                else:
                    backoff = 2.0 * (attempt + 1)
                    logger.warning("Transient error on %s (%s). Retrying in %.1fs...", example_id, exc, backoff)
                    time.sleep(backoff)
            else:
                backoff = 2.0 * (attempt + 1)
                logger.warning("Transient error on %s (%s). Retrying in %.1fs...", example_id, exc, backoff)
                time.sleep(backoff)

    raise RuntimeError(f"Failed to evaluate example {example_id}")


def compute_interreviewer_agreement(
    df_primary: pd.DataFrame,
    df_secondary: pd.DataFrame,
    primary_model: str = PRIMARY_JUDGE_MODEL,
    secondary_model: str = SECONDARY_REVIEWER_MODEL,
) -> Dict[str, Any]:
    """
    Computes rigorous inter-reviewer agreement between Primary Judge and Secondary AI Reviewer.
    """
    merged = pd.merge(df_primary, df_secondary, on="example_id", suffixes=("_prim", "_sec"))
    n = len(merged)

    dimensional_metrics: Dict[str, Any] = {}
    exact_agreements = []
    plus_minus_one_agreements = []
    maes = []
    spearmans = []

    # Score column name mapping
    prim_cols = {
        "relevance": "llm_relevance" if "llm_relevance" in merged.columns else "relevance_score",
        "groundedness": "llm_groundedness" if "llm_groundedness" in merged.columns else "groundedness_score",
        "actionability": "llm_actionability" if "llm_actionability" in merged.columns else "actionability_score",
        "tone": "llm_tone" if "llm_tone" in merged.columns else "tone_score",
        "safety": "llm_safety" if "llm_safety" in merged.columns else "safety_score",
    }

    for dim in DIMENSIONS:
        p_col = prim_cols[dim]
        s_col = f"secondary_reviewer_{dim}"

        p_vals = pd.to_numeric(merged[p_col], errors="coerce").values
        s_vals = pd.to_numeric(merged[s_col], errors="coerce").values

        valid_mask = ~np.isnan(p_vals) & ~np.isnan(s_vals)
        p_arr = p_vals[valid_mask]
        s_arr = s_vals[valid_mask]

        if len(p_arr) == 0:
            continue

        exact_match = float(np.mean(p_arr == s_arr))
        pm1_match = float(np.mean(np.abs(p_arr - s_arr) <= 1))
        mae = float(np.mean(np.abs(p_arr - s_arr)))

        # Spearman correlation
        if len(np.unique(p_arr)) > 1 and len(np.unique(s_arr)) > 1:
            spearman_corr, spearman_pval = spearmanr(p_arr, s_arr)
            spearman_corr = float(spearman_corr) if not np.isnan(spearman_corr) else 0.0
            spearman_pval = float(spearman_pval) if not np.isnan(spearman_pval) else 1.0
        else:
            spearman_corr, spearman_pval = 0.0, 1.0

        exact_agreements.append(exact_match)
        plus_minus_one_agreements.append(pm1_match)
        maes.append(mae)
        spearmans.append(spearman_corr)

        dimensional_metrics[dim] = {
            "exact_agreement": round(exact_match, 4),
            "plus_minus_one_agreement": round(pm1_match, 4),
            "mean_absolute_error": round(mae, 4),
            "spearman_correlation": round(spearman_corr, 4),
            "spearman_pvalue": round(spearman_pval, 6),
            "primary_mean": round(float(np.mean(p_arr)), 4),
            "secondary_mean": round(float(np.mean(s_arr)), 4),
        }

    # Binary metrics
    def to_bool_arr(series: pd.Series) -> np.ndarray:
        return series.apply(
            lambda v: v is True or str(v).strip().lower() in ("true", "1", "yes", "t")
        ).values

    prim_halluc_col = "llm_hallucination" if "llm_hallucination" in merged.columns else "hallucination_detected"
    prim_pass_col = "llm_pass" if "llm_pass" in merged.columns else "judge_pass"

    p_halluc = to_bool_arr(merged[prim_halluc_col])
    s_halluc = to_bool_arr(merged["secondary_reviewer_hallucination"])
    halluc_agreement = float(np.mean(p_halluc == s_halluc))
    try:
        halluc_kappa = float(cohen_kappa_score(p_halluc, s_halluc))
        if np.isnan(halluc_kappa):
            halluc_kappa = 1.0 if halluc_agreement == 1.0 else 0.0
    except Exception:
        halluc_kappa = 1.0 if halluc_agreement == 1.0 else 0.0

    p_pass = to_bool_arr(merged[prim_pass_col])
    s_pass = to_bool_arr(merged["secondary_reviewer_pass"])
    pass_agreement = float(np.mean(p_pass == s_pass))
    try:
        pass_kappa = float(cohen_kappa_score(p_pass, s_pass))
        if np.isnan(pass_kappa):
            pass_kappa = 1.0 if pass_agreement == 1.0 else 0.0
    except Exception:
        pass_kappa = 1.0 if pass_agreement == 1.0 else 0.0

    secondary_summary = {
        "mean_relevance": round(float(pd.to_numeric(merged["secondary_reviewer_relevance"]).mean()), 4),
        "mean_groundedness": round(float(pd.to_numeric(merged["secondary_reviewer_groundedness"]).mean()), 4),
        "mean_actionability": round(float(pd.to_numeric(merged["secondary_reviewer_actionability"]).mean()), 4),
        "mean_tone": round(float(pd.to_numeric(merged["secondary_reviewer_tone"]).mean()), 4),
        "mean_safety": round(float(pd.to_numeric(merged["secondary_reviewer_safety"]).mean()), 4),
        "hallucination_rate": round(float(np.mean(s_halluc)), 4),
        "pass_rate": round(float(np.mean(s_pass)), 4),
    }

    return {
        "audit_type": "independent_secondary_llm_review",
        "primary_judge_model": primary_model,
        "secondary_reviewer_model": secondary_model,
        "reviewed_cases": n,
        "limitation_note": (
            "Because the secondary quality audit was performed by an independent LLM "
            "rather than a human evaluator, inter-model agreement should not be "
            "interpreted as human validation."
        ),
        "dimensional_metrics": dimensional_metrics,
        "binary_metrics": {
            "hallucination_agreement": round(halluc_agreement, 4),
            "hallucination_cohen_kappa": round(halluc_kappa, 4),
            "pass_fail_agreement": round(pass_agreement, 4),
            "pass_fail_cohen_kappa": round(pass_kappa, 4),
        },
        "overall_summary": {
            "average_exact_score_agreement": round(float(np.mean(exact_agreements)), 4) if exact_agreements else 0.0,
            "average_plus_minus_one_score_agreement": round(float(np.mean(plus_minus_one_agreements)), 4) if plus_minus_one_agreements else 0.0,
            "average_mae": round(float(np.mean(maes)), 4) if maes else 0.0,
        },
        "secondary_reviewer_summary": secondary_summary,
    }


def run_secondary_ai_review(
    sample_csv_path: Path | str = "evaluation/results/judge/human_review_sample.csv",
    golden_results_path: Path | str = "evaluation/results/final/golden_eval_results.csv",
    golden_set_path: Path | str = "evaluation/golden_set_final.csv",
    output_csv_path: Path | str = "evaluation/results/judge/secondary_ai_review_results.csv",
    agreement_json_path: Path | str = "evaluation/results/judge/llm_interreviewer_agreement.json",
    evaluate_all: bool = False,
    reviewer_model: str = SECONDARY_REVIEWER_MODEL,
) -> Dict[str, Any]:
    sample_csv_path = Path(sample_csv_path)
    golden_results_path = Path(golden_results_path)
    golden_set_path = Path(golden_set_path)
    output_csv_path = Path(output_csv_path)
    agreement_json_path = Path(agreement_json_path)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Safety: Verify Golden Set SHA-256
    actual_sha = compute_file_sha256(golden_set_path)
    if actual_sha != FROZEN_GOLDEN_SHA256:
        raise ValueError(
            f"SAFETY ASSERTION FAILED: Golden set SHA-256 mismatch!\n"
            f"Expected: {FROZEN_GOLDEN_SHA256}\n"
            f"Actual:   {actual_sha}"
        )

    # Load golden metadata for context
    gold_data: Dict[str, Dict[str, Any]] = {}
    if golden_set_path.exists():
        gdf = pd.read_csv(golden_set_path, dtype=str, keep_default_na=False)
        gold_data = {str(r["example_id"]).strip(): r.to_dict() for _, r in gdf.iterrows()}

    # Select evaluation target rows
    if evaluate_all:
        df_target = pd.read_csv(golden_results_path, encoding="utf-8-sig")
        logger.info("Evaluating all %d golden results using secondary reviewer %s", len(df_target), reviewer_model)
    else:
        df_target = pd.read_csv(sample_csv_path, encoding="utf-8-sig")
        logger.info("Evaluating %d review sample rows using secondary reviewer %s", len(df_target), reviewer_model)

    # Load existing checkpoints to resume cleanly without duplication
    existing_records: Dict[str, Dict[str, Any]] = {}
    if output_csv_path.exists():
        try:
            with open(output_csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    eid = r.get("example_id")
                    if eid:
                        existing_records[eid] = r
            logger.info("Found %d existing secondary review checkpoints in %s", len(existing_records), output_csv_path)
        except Exception as e:
            logger.warning("Could not load existing checkpoints: %s", e)

    client = get_gemini_client()
    secondary_results: Dict[str, Dict[str, Any]] = dict(existing_records)

    for idx, (_, row) in enumerate(df_target.iterrows(), 1):
        eid = str(row["example_id"]).strip()
        if eid in secondary_results:
            continue

        row_dict = row.to_dict()
        # Enrich with golden metadata (historical reply / points) if missing
        if eid in gold_data:
            g = gold_data[eid]
            if not row_dict.get("historical_amazon_reply"):
                row_dict["historical_amazon_reply"] = g.get("historical_amazon_reply", "")
            if not row_dict.get("expected_reply_points"):
                row_dict["expected_reply_points"] = g.get("expected_reply_points") or g.get("suggested_reply_points", "")

        logger.info("[%d/%d] Secondary reviewing example %s...", idx, len(df_target), eid)
        res = review_single_reply(client=client, model=reviewer_model, row=row_dict)
        secondary_results[eid] = res

        # Immediate checkpoint autosave after every row
        df_temp = pd.DataFrame(list(secondary_results.values()))
        df_temp.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    # Final DataFrame
    df_secondary = pd.DataFrame(list(secondary_results.values()))
    score_cols = [f"secondary_reviewer_{d}" for d in DIMENSIONS]
    for col in score_cols:
        df_secondary[col] = df_secondary[col].astype(int)
    for col in ["secondary_reviewer_hallucination", "secondary_reviewer_pass"]:
        df_secondary[col] = df_secondary[col].map(
            lambda v: v is True or str(v).strip().lower() in ("true", "1", "yes", "t")
        )

    df_secondary.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    logger.info("Saved complete secondary review results to %s (%d rows)", output_csv_path, len(df_secondary))

    # Assertions on Secondary Output
    assert len(df_secondary) == len(df_target), f"Expected {len(df_target)} rows, got {len(df_secondary)}"
    assert df_secondary["example_id"].nunique() == len(df_target), "Duplicate IDs in secondary review output!"
    for col in score_cols:
        assert df_secondary[col].isin([1, 2, 3, 4, 5]).all(), f"Invalid score in {col}"

    # Compute Inter-Reviewer Agreement against Primary Judge
    agreement = compute_interreviewer_agreement(
        df_primary=df_target,
        df_secondary=df_secondary,
        primary_model=PRIMARY_JUDGE_MODEL,
        secondary_model=reviewer_model,
    )

    with open(agreement_json_path, "w", encoding="utf-8") as f:
        json.dump(agreement, f, indent=2)
    logger.info("Saved inter-reviewer agreement report to %s", agreement_json_path)

    # Print Summary Report
    print("\n" + "=" * 70)
    print("HIVER SECONDARY AI REVIEW & INTER-MODEL AGREEMENT REPORT")
    print("=" * 70)
    print(f"Primary Judge Model:        {PRIMARY_JUDGE_MODEL}")
    print(f"Secondary Reviewer Model:   {reviewer_model}")
    print(f"Total Reviewed Cases:       {agreement['reviewed_cases']}")
    print("-" * 70)
    print("SECONDARY REVIEWER SUMMARY:")
    s_sum = agreement["secondary_reviewer_summary"]
    print(f"  Mean Relevance:           {s_sum['mean_relevance']:.2f} / 5.0")
    print(f"  Mean Groundedness:        {s_sum['mean_groundedness']:.2f} / 5.0")
    print(f"  Mean Actionability:       {s_sum['mean_actionability']:.2f} / 5.0")
    print(f"  Mean Tone:                {s_sum['mean_tone']:.2f} / 5.0")
    print(f"  Mean Safety:              {s_sum['mean_safety']:.2f} / 5.0")
    print(f"  Hallucination Rate:       {s_sum['hallucination_rate']:.2%}")
    print(f"  Pass Rate:                {s_sum['pass_rate']:.2%}")
    print("-" * 70)
    print("DIMENSIONAL AGREEMENT (Primary vs Secondary):")
    for dim, met in agreement["dimensional_metrics"].items():
        print(
            f"  {dim.capitalize():<15} | "
            f"Exact: {met['exact_agreement']:.2%} | "
            f"±1: {met['plus_minus_one_agreement']:.2%} | "
            f"MAE: {met['mean_absolute_error']:.2f} | "
            f"Spearman r: {met['spearman_correlation']:.4f}"
        )
    print("-" * 70)
    b_met = agreement["binary_metrics"]
    print(f"Hallucination Agreement:    {b_met['hallucination_agreement']:.2%} (Cohen's Kappa: {b_met['hallucination_cohen_kappa']:.4f})")
    print(f"Pass/Fail Agreement:        {b_met['pass_fail_agreement']:.2%} (Cohen's Kappa: {b_met['pass_fail_cohen_kappa']:.4f})")
    print("-" * 70)
    o_sum = agreement["overall_summary"]
    print(f"Overall Exact Score Agr.:   {o_sum['average_exact_score_agreement']:.2%}")
    print(f"Overall ±1 Score Agr.:      {o_sum['average_plus_minus_one_score_agreement']:.2%}")
    print(f"Overall Average MAE:        {o_sum['average_mae']:.4f}")
    print("-" * 70)
    print(f"Limitation Note: {agreement['limitation_note']}")
    print("=" * 70 + "\n")

    return agreement


def main():
    parser = argparse.ArgumentParser(description="Run Secondary AI Review on support replies")
    parser.add_argument("--sample", default="evaluation/results/judge/human_review_sample.csv")
    parser.add_argument("--results", default="evaluation/results/final/golden_eval_results.csv")
    parser.add_argument("--golden", default="evaluation/golden_set_final.csv")
    parser.add_argument("--output", default="evaluation/results/judge/secondary_ai_review_results.csv")
    parser.add_argument("--agreement", default="evaluation/results/judge/llm_interreviewer_agreement.json")
    parser.add_argument("--all", action="store_true", help="Evaluate all 158 examples instead of 40-row sample")
    parser.add_argument("--model", default=SECONDARY_REVIEWER_MODEL, help="Secondary reviewer model")
    args = parser.parse_args()

    run_secondary_ai_review(
        sample_csv_path=args.sample,
        golden_results_path=args.results,
        golden_set_path=args.golden,
        output_csv_path=args.output,
        agreement_json_path=args.agreement,
        evaluate_all=args.all,
        reviewer_model=args.model,
    )


if __name__ == "__main__":
    main()
