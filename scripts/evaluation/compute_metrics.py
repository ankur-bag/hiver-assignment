"""
Metrics calculation module for Hiver Golden-Set Evaluation.
Calculates Intent, Escalation, Grounding, and Latency metrics with breakdowns.
Outputs summary JSON and prints terminal report.
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def calculate_metrics_from_csv(
    csv_path: str | Path,
    max_expected_examples: Optional[int] = 200,
    strict_no_duplicates: bool = True,
) -> Dict[str, Any]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    raw_rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            raw_rows.append(r)

    if not raw_rows:
        return {"error": "Empty results CSV"}

    # Validate invariant 1: No duplicate example_id in results CSV
    unique_ids = [r.get("example_id") for r in raw_rows if r.get("example_id")]
    if strict_no_duplicates and len(raw_rows) != len(set(unique_ids)):
        raise ValueError(
            f"Invariant violation: results CSV contains duplicate example_ids "
            f"({len(raw_rows)} rows vs {len(set(unique_ids))} unique IDs). Failing loudly."
        )

    # Validate invariant 2: Result count must not exceed max allowed golden examples
    if max_expected_examples is not None and len(raw_rows) > max_expected_examples:
        raise ValueError(
            f"Invariant violation: results row count ({len(raw_rows)}) exceeds max allowed "
            f"golden set count ({max_expected_examples}). Failing loudly."
        )

    # Defensively deduplicate by example_id, preserving the most recent (latest) occurrence
    deduped_by_id: Dict[str, Dict[str, Any]] = {}
    rows_without_id: List[Dict[str, Any]] = []

    for r in raw_rows:
        eid = r.get("example_id")
        if eid:
            deduped_by_id[eid] = r
        else:
            rows_without_id.append(r)

    rows = list(deduped_by_id.values()) + rows_without_id
    total_examples = len(rows)

    successful_rows = [r for r in rows if r.get("request_status") == "SUCCESS"]
    failed_rows = [r for r in rows if r.get("request_status") != "SUCCESS"]

    # Classify failures into infrastructure, provider, and model/eval failures
    infra_fail_types = {
        "ENDPOINT_NOT_FOUND", "HTTP_404", "CONNECT_ERROR", "CONNECTION_REFUSED",
        "DNS_ERROR", "HTTP_502", "HTTP_503", "HTTP_504", "NETWORK_ERROR"
    }
    infra_failed = []
    provider_failed = []
    other_failed = []

    for r in failed_rows:
        cat = (r.get("error_category") or "").upper()
        err_type = (r.get("error_type") or "").upper()
        if cat == "INFRASTRUCTURE" or err_type in infra_fail_types or "HTTP_404" in err_type:
            infra_failed.append(r)
        elif cat == "PROVIDER" or "PROVIDER" in err_type or "QUOTA" in err_type:
            provider_failed.append(r)
        else:
            other_failed.append(r)

    n_total = total_examples
    n_success = len(successful_rows)
    n_failed = len(failed_rows)

    # Error analysis
    error_counts = Counter()
    for r in failed_rows:
        err_type = r.get("error_type") or "UNKNOWN_ERROR"
        error_counts[err_type] += 1

    # Latency Metrics (from successful requests only)
    ttfts = []
    totals = []
    for r in successful_rows:
        try:
            t = float(r.get("ttft_ms", 0))
            if t > 0:
                ttfts.append(t)
        except (ValueError, TypeError):
            pass
        try:
            tot = float(r.get("total_latency_ms", 0))
            if tot > 0:
                totals.append(tot)
        except (ValueError, TypeError):
            pass

    def percentile(data: List[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_data[int(k)]
        d0 = sorted_data[int(f)] * (c - k)
        d1 = sorted_data[int(c)] * (k - f)
        return d0 + d1

    latency_metrics = {
        "median_ttft_ms": round(percentile(ttfts, 50), 2),
        "p90_ttft_ms": round(percentile(ttfts, 90), 2),
        "p95_ttft_ms": round(percentile(ttfts, 95), 2),
        "median_total_ms": round(percentile(totals, 50), 2),
        "p90_total_ms": round(percentile(totals, 90), 2),
        "p95_total_ms": round(percentile(totals, 95), 2),
        "min_total_ms": round(min(totals), 2) if totals else 0.0,
        "max_total_ms": round(max(totals), 2) if totals else 0.0,
    }

    # Intent Metrics
    all_intents = sorted(list(set(
        [r.get("expected_intent", "") for r in rows if r.get("expected_intent")] +
        [r.get("predicted_intent", "") for r in successful_rows if r.get("predicted_intent")]
    )))
    all_intents = [i for i in all_intents if i]

    # Confusion Matrix: row = expected, col = predicted
    confusion_matrix = {exp: {pred: 0 for pred in all_intents} for exp in all_intents}
    per_intent_tp = Counter()
    per_intent_fp = Counter()
    per_intent_fn = Counter()
    per_intent_support = Counter()

    intent_correct_count = 0
    for r in successful_rows:
        exp = r.get("expected_intent", "")
        pred = r.get("predicted_intent", "")
        if not exp:
            continue
        per_intent_support[exp] += 1
        if exp == pred:
            intent_correct_count += 1
            per_intent_tp[exp] += 1
        else:
            per_intent_fp[pred] += 1
            per_intent_fn[exp] += 1

        if exp in confusion_matrix and pred in confusion_matrix[exp]:
            confusion_matrix[exp][pred] += 1

    intent_accuracy = (intent_correct_count / n_success) if n_success > 0 else 0.0

    per_intent_metrics = {}
    precisions, recalls, f1s, weights = [], [], [], []

    for intent in all_intents:
        tp = per_intent_tp[intent]
        fp = per_intent_fp[intent]
        fn = per_intent_fn[intent]
        support = per_intent_support[intent]

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_intent_metrics[intent] = {
            "support": support,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

        if support > 0:
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
            weights.append(support)

    macro_prec = sum(precisions) / len(precisions) if precisions else 0.0
    macro_rec = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    weighted_f1 = (
        sum(f * w for f, w in zip(f1s, weights)) / sum(weights)
        if sum(weights) > 0
        else 0.0
    )

    intent_summary = {
        "accuracy": round(intent_accuracy, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_intent": per_intent_metrics,
        "confusion_matrix": confusion_matrix,
    }

    # Escalation Metrics (Positive class = True)
    esc_tp = 0
    esc_fp = 0
    esc_tn = 0
    esc_fn = 0
    false_positives = []
    false_negatives = []

    def parse_bool(val: Any) -> bool:
        if isinstance(val, bool):
            return val
        return str(val).strip().lower() in ("true", "1", "yes", "t")

    for r in successful_rows:
        eid = r.get("example_id", "")
        exp_esc = parse_bool(r.get("expected_escalate", False))
        pred_esc = parse_bool(r.get("predicted_escalate", False))

        if exp_esc and pred_esc:
            esc_tp += 1
        elif not exp_esc and not pred_esc:
            esc_tn += 1
        elif not exp_esc and pred_esc:
            esc_fp += 1
            false_positives.append(eid)
        elif exp_esc and not pred_esc:
            esc_fn += 1
            false_negatives.append(eid)

    esc_total = esc_tp + esc_tn + esc_fp + esc_fn
    esc_accuracy = (esc_tp + esc_tn) / esc_total if esc_total > 0 else 0.0
    esc_precision = esc_tp / (esc_tp + esc_fp) if (esc_tp + esc_fp) > 0 else 0.0
    esc_recall = esc_tp / (esc_tp + esc_fn) if (esc_tp + esc_fn) > 0 else 0.0
    esc_f1 = (
        (2 * esc_precision * esc_recall) / (esc_precision + esc_recall)
        if (esc_precision + esc_recall) > 0
        else 0.0
    )

    escalation_summary = {
        "accuracy": round(esc_accuracy, 4),
        "precision": round(esc_precision, 4),
        "recall": round(esc_recall, 4),
        "f1": round(esc_f1, 4),
        "tp": esc_tp,
        "tn": esc_tn,
        "fp": esc_fp,
        "fn": esc_fn,
        "false_positive_ids": false_positives,
        "false_negative_ids": false_negatives,
    }

    # Grounding / Retrieval Metrics
    grounded_count = 0
    grounding_by_intent = defaultdict(lambda: {"total": 0, "grounded": 0})
    grounding_by_difficulty = defaultdict(lambda: {"total": 0, "grounded": 0})
    grounding_by_lang = defaultdict(lambda: {"total": 0, "grounded": 0})

    for r in successful_rows:
        cnt = int(r.get("retrieved_context_count") or 0)
        avail = parse_bool(r.get("retrieved_context_available", False)) or (cnt > 0)
        exp_intent = r.get("expected_intent", "UNKNOWN")
        diff = r.get("difficulty", "unknown").lower()
        lang = r.get("language", "unknown").lower()

        if avail:
            grounded_count += 1

        grounding_by_intent[exp_intent]["total"] += 1
        grounding_by_difficulty[diff]["total"] += 1
        grounding_by_lang[lang]["total"] += 1

        if avail:
            grounding_by_intent[exp_intent]["grounded"] += 1
            grounding_by_difficulty[diff]["grounded"] += 1
            grounding_by_lang[lang]["grounded"] += 1

    overall_grounding_rate = (grounded_count / n_success) if n_success > 0 else 0.0

    grounding_summary = {
        "success_rate": round(overall_grounding_rate, 4),
        "grounded_count": grounded_count,
        "total_successful": n_success,
        "by_intent": {
            k: round(v["grounded"] / v["total"], 4) if v["total"] > 0 else 0.0
            for k, v in grounding_by_intent.items()
        },
        "by_difficulty": {
            k: round(v["grounded"] / v["total"], 4) if v["total"] > 0 else 0.0
            for k, v in grounding_by_difficulty.items()
        },
        "by_language": {
            k: round(v["grounded"] / v["total"], 4) if v["total"] > 0 else 0.0
            for k, v in grounding_by_lang.items()
        },
    }

    # Difficulty Breakdown for Intent Accuracy
    difficulty_intent = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in successful_rows:
        diff = r.get("difficulty", "unknown").lower()
        difficulty_intent[diff]["total"] += 1
        if parse_bool(r.get("intent_correct", False)):
            difficulty_intent[diff]["correct"] += 1

    difficulty_summary = {
        diff: {
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy": round(stats["correct"] / stats["total"], 4) if stats["total"] > 0 else 0.0,
        }
        for diff, stats in sorted(difficulty_intent.items())
    }

    # Language Breakdown for Intent Accuracy
    language_intent = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in successful_rows:
        lang = r.get("language", "unknown").lower()
        language_intent[lang]["total"] += 1
        if parse_bool(r.get("intent_correct", False)):
            language_intent[lang]["correct"] += 1

    # Also calculate English vs non-English
    en_total = sum(v["total"] for k, v in language_intent.items() if k == "en")
    en_correct = sum(v["correct"] for k, v in language_intent.items() if k == "en")
    non_en_total = sum(v["total"] for k, v in language_intent.items() if k != "en")
    non_en_correct = sum(v["correct"] for k, v in language_intent.items() if k != "en")

    language_summary = {
        "english": {
            "total": en_total,
            "correct": en_correct,
            "accuracy": round(en_correct / en_total, 4) if en_total > 0 else 0.0,
        },
        "non_english": {
            "total": non_en_total,
            "correct": non_en_correct,
            "accuracy": round(non_en_correct / non_en_total, 4) if non_en_total > 0 else 0.0,
        },
        "individual": {
            lang: {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": round(stats["correct"] / stats["total"], 4) if stats["total"] > 0 else 0.0,
            }
            for lang, stats in sorted(language_intent.items())
        },
    }

    summary = {
        "n_total": n_total,
        "n_success": n_success,
        "n_failed": n_failed,
        "model_successful_examples": len(successful_rows),
        "provider_failures": len(provider_failed),
        "infrastructure_failures": len(infra_failed),
        "other_failures": len(other_failed),
        "intent": intent_summary,
        "escalation": escalation_summary,
        "grounding": grounding_summary,
        "latency": latency_metrics,
        "difficulty_breakdown": difficulty_summary,
        "language_breakdown": language_summary,
        "error_counts": dict(error_counts),
    }

    return summary


def print_console_report(summary: Dict[str, Any], result_file: Optional[str] = None):
    print("\n" + "=" * 60)
    print("HIVER GOLDEN EVALUATION")
    print("=" * 60)
    print(f"Examples:   {summary.get('n_total', 0)}")
    print(f"Successful: {summary.get('n_success', 0)}")
    print(f"Failed:     {summary.get('n_failed', 0)}")
    if summary.get("infrastructure_failures", 0) > 0 or summary.get("provider_failures", 0) > 0:
        print(f"  - Model Success:          {summary.get('model_successful_examples', 0)}")
        print(f"  - Infrastructure Errors:  {summary.get('infrastructure_failures', 0)}")
        print(f"  - Provider Errors:        {summary.get('provider_failures', 0)}")

    intent = summary.get("intent", {})
    print("\nINTENT")
    print(f"Accuracy:   {intent.get('accuracy', 0.0):.2%}")
    print(f"Macro F1:   {intent.get('macro_f1', 0.0):.4f}")
    print(f"Weighted F1:{intent.get('weighted_f1', 0.0):.4f}")

    esc = summary.get("escalation", {})
    print("\nESCALATION")
    print(f"Accuracy:   {esc.get('accuracy', 0.0):.2%}")
    print(f"Precision:  {esc.get('precision', 0.0):.4f}")
    print(f"Recall:     {esc.get('recall', 0.0):.4f}")
    print(f"F1:         {esc.get('f1', 0.0):.4f}")
    if esc.get("fp", 0) > 0:
        print(f"False Positives ({esc['fp']}): {', '.join(esc.get('false_positive_ids', [])[:10])}")
    if esc.get("fn", 0) > 0:
        print(f"False Negatives ({esc['fn']}): {', '.join(esc.get('false_negative_ids', [])[:10])}")

    grounding = summary.get("grounding", {})
    print("\nGROUNDING")
    print(f"Success rate: {grounding.get('success_rate', 0.0):.2%}")

    latency = summary.get("latency", {})
    print("\nLATENCY")
    print(f"Median TTFT:  {latency.get('median_ttft_ms', 0):.1f} ms")
    print(f"P95 TTFT:     {latency.get('p95_ttft_ms', 0):.1f} ms")
    print(f"Median total: {latency.get('median_total_ms', 0):.1f} ms")
    print(f"P95 total:    {latency.get('p95_total_ms', 0):.1f} ms")

    diff = summary.get("difficulty_breakdown", {})
    if diff:
        print("\nDIFFICULTY ACCURACY")
        for d, s in diff.items():
            print(f"  {d:<10} {s.get('accuracy', 0.0):.2%} ({s.get('correct')}/{s.get('total')})")

    lang = summary.get("language_breakdown", {})
    if lang:
        print("\nLANGUAGE ACCURACY")
        en = lang.get("english", {})
        non_en = lang.get("non_english", {})
        print(f"  English:     {en.get('accuracy', 0.0):.2%} ({en.get('correct')}/{en.get('total')})")
        print(f"  Non-English: {non_en.get('accuracy', 0.0):.2%} ({non_en.get('correct')}/{non_en.get('total')})")

    errs = summary.get("error_counts", {})
    if errs:
        print("\nERRORS")
        for e_type, count in errs.items():
            print(f"  {e_type}: {count}")

    if result_file:
        print(f"\nRESULT FILE:\n{result_file}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compute metrics from golden evaluation results CSV")
    parser.add_argument(
        "--input",
        default="evaluation/results/golden_eval_results.csv",
        help="Path to results CSV"
    )
    parser.add_argument(
        "--output",
        default="evaluation/results/golden_eval_summary.json",
        help="Path to save summary JSON"
    )
    args = parser.parse_args()

    summary = calculate_metrics_from_csv(args.input)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_console_report(summary, result_file=args.input)
    print(f"Saved summary JSON to: {out_path}")


if __name__ == "__main__":
    main()
