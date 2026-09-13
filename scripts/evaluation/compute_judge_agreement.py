"""
Hiver Assignment - Judge vs Human Agreement Computation.

Reads the completed human review sample (human_review_sample.csv) and computes:
1. Exact agreement for each 1-5 dimension
2. Agreement within +/- 1 point
3. Mean Absolute Error (MAE)
4. Spearman correlation for each dimension
5. Hallucination binary agreement
6. Pass/fail agreement & Cohen's kappa
7. Overall average exact and +/-1 agreement

Fails gracefully if human fields are blank without fabricating fake metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("compute_judge_agreement")

DIMENSIONS = ["relevance", "groundedness", "actionability", "tone", "safety"]


def parse_bool_val(val: Any) -> Optional[bool]:
    if pd.isna(val) or val is None or str(val).strip() == "":
        return None
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("true", "1", "yes", "t", "pass"):
        return True
    if s in ("false", "0", "no", "f", "fail"):
        return False
    return None


def calculate_agreement(
    sample_csv_path: Path | str = "evaluation/results/judge/human_review_sample.csv",
    output_json_path: Optional[Path | str] = "evaluation/results/judge/judge_human_agreement.json",
) -> Optional[Dict[str, Any]]:
    sample_csv_path = Path(sample_csv_path)

    if not sample_csv_path.exists():
        logger.error("Human review sample CSV not found at: %s", sample_csv_path)
        print(f"Error: Sample file not found at {sample_csv_path}")
        return None

    df = pd.read_csv(sample_csv_path, encoding="utf-8-sig")

    # Required columns
    human_cols = [f"human_{d}" for d in DIMENSIONS] + ["human_hallucination", "human_pass"]
    llm_cols = [f"llm_{d}" for d in DIMENSIONS] + ["llm_hallucination", "llm_pass"]

    missing_cols = [c for c in human_cols + llm_cols if c not in df.columns]
    if missing_cols:
        logger.error("Sample CSV missing required columns: %s", missing_cols)
        print(f"Error: Missing columns {missing_cols}")
        return None

    # Check if human fields are populated
    human_populated = False
    for col in human_cols:
        non_empty = df[col].dropna().astype(str).str.strip()
        if (non_empty != "").sum() > 0:
            human_populated = True
            break

    if not human_populated:
        print("\n" + "=" * 60)
        print("JUDGE-HUMAN AGREEMENT STATUS")
        print("=" * 60)
        print("Human ratings have not been completed yet.")
        print(f"Please fill out human scores in: {sample_csv_path}")
        print("=" * 60 + "\n")
        return None

    # Filter to completed rows
    valid_rows = []
    for idx, row in df.iterrows():
        is_complete = True
        for d in DIMENSIONS:
            val = row.get(f"human_{d}")
            if pd.isna(val) or str(val).strip() == "":
                is_complete = False
                break
        if is_complete:
            valid_rows.append(row)

    if not valid_rows:
        print("\nHuman ratings have not been completed yet (all rows incomplete).\n")
        return None

    df_valid = pd.DataFrame(valid_rows)
    n_rated = len(df_valid)
    logger.info("Computing agreement over %d completed human review rows (out of %d total)", n_rated, len(df))

    dimensional_metrics: Dict[str, Any] = {}
    exact_agreements = []
    plus_minus_one_agreements = []
    maes = []

    for dim in DIMENSIONS:
        llm_vals = pd.to_numeric(df_valid[f"llm_{dim}"], errors="coerce").values
        human_vals = pd.to_numeric(df_valid[f"human_{dim}"], errors="coerce").values

        # Remove any NaN pair
        valid_mask = ~np.isnan(llm_vals) & ~np.isnan(human_vals)
        l_arr = llm_vals[valid_mask]
        h_arr = human_vals[valid_mask]

        if len(l_arr) == 0:
            continue

        exact_match = float(np.mean(l_arr == h_arr))
        pm1_match = float(np.mean(np.abs(l_arr - h_arr) <= 1))
        mae = float(np.mean(np.abs(l_arr - h_arr)))

        # Spearman correlation
        if len(np.unique(l_arr)) > 1 and len(np.unique(h_arr)) > 1:
            spearman_corr, spearman_pval = spearmanr(l_arr, h_arr)
            spearman_corr = float(spearman_corr) if not np.isnan(spearman_corr) else 0.0
            spearman_pval = float(spearman_pval) if not np.isnan(spearman_pval) else 1.0
        else:
            spearman_corr, spearman_pval = 0.0, 1.0

        exact_agreements.append(exact_match)
        plus_minus_one_agreements.append(pm1_match)
        maes.append(mae)

        dimensional_metrics[dim] = {
            "exact_agreement": round(exact_match, 4),
            "plus_minus_one_agreement": round(pm1_match, 4),
            "mean_absolute_error": round(mae, 4),
            "spearman_correlation": round(spearman_corr, 4),
            "spearman_pvalue": round(spearman_pval, 4),
        }

    # Binary metrics: Hallucination & Pass
    llm_halluc = [parse_bool_val(v) for v in df_valid["llm_hallucination"]]
    human_halluc = [parse_bool_val(v) for v in df_valid["human_hallucination"]]
    halluc_pairs = [(l, h) for l, h in zip(llm_halluc, human_halluc) if l is not None and h is not None]
    halluc_agreement = (
        sum(1 for l, h in halluc_pairs if l == h) / len(halluc_pairs) if halluc_pairs else 0.0
    )

    llm_pass = [parse_bool_val(v) for v in df_valid["llm_pass"]]
    human_pass = [parse_bool_val(v) for v in df_valid["human_pass"]]
    pass_pairs = [(l, h) for l, h in zip(llm_pass, human_pass) if l is not None and h is not None]

    pass_agreement = (
        sum(1 for l, h in pass_pairs if l == h) / len(pass_pairs) if pass_pairs else 0.0
    )

    if pass_pairs:
        lp_list = [p[0] for p in pass_pairs]
        hp_list = [p[1] for p in pass_pairs]
        try:
            kappa = float(cohen_kappa_score(lp_list, hp_list))
            kappa = kappa if not np.isnan(kappa) else 0.0
        except Exception:
            kappa = 0.0
    else:
        kappa = 0.0

    avg_exact = float(np.mean(exact_agreements)) if exact_agreements else 0.0
    avg_pm1 = float(np.mean(plus_minus_one_agreements)) if plus_minus_one_agreements else 0.0
    avg_mae = float(np.mean(maes)) if maes else 0.0

    results = {
        "status": "completed",
        "sample_size": len(df),
        "rated_examples": n_rated,
        "overall_summary": {
            "average_exact_agreement": round(avg_exact, 4),
            "average_plus_minus_one_agreement": round(avg_pm1, 4),
            "average_mae": round(avg_mae, 4),
            "pass_fail_agreement": round(pass_agreement, 4),
            "pass_fail_cohen_kappa": round(kappa, 4),
            "hallucination_agreement": round(halluc_agreement, 4),
        },
        "dimensional_metrics": dimensional_metrics,
    }

    if output_json_path:
        out_p = Path(output_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved agreement metrics to %s", out_p)

    # Print Report
    print("\n" + "=" * 70)
    print(f"JUDGE-HUMAN AGREEMENT REPORT ({n_rated} RATED EXAMPLES)")
    print("=" * 70)
    print(f"Overall Average Exact Agreement:    {avg_exact:.2%}")
    print(f"Overall Average +/- 1 Agreement:    {avg_pm1:.2%}")
    print(f"Overall Average MAE:                {avg_mae:.3f}")
    print(f"Pass/Fail Exact Agreement:          {pass_agreement:.2%}")
    print(f"Pass/Fail Cohen's Kappa:            {kappa:.4f}")
    print(f"Hallucination Binary Agreement:     {halluc_agreement:.2%}")
    print("-" * 70)
    print(f"{'Dimension':<20} | {'Exact Match':<12} | {'+/- 1 Match':<12} | {'MAE':<8} | {'Spearman r':<10}")
    print("-" * 70)
    for dim, met in dimensional_metrics.items():
        print(f"{dim.capitalize():<20} | {met['exact_agreement']:<12.2%} | {met['plus_minus_one_agreement']:<12.2%} | {met['mean_absolute_error']:<8.3f} | {met['spearman_correlation']:<10.4f}")
    print("=" * 70 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute agreement metrics between LLM judge and human ratings.")
    parser.add_argument(
        "--input",
        default="evaluation/results/judge/human_review_sample.csv",
        help="Path to human review sample CSV",
    )
    parser.add_argument(
        "--output",
        default="evaluation/results/judge/judge_human_agreement.json",
        help="Path to output agreement JSON",
    )
    args = parser.parse_args()

    calculate_agreement(
        sample_csv_path=args.input,
        output_json_path=args.output,
    )


if __name__ == "__main__":
    main()
