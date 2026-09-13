"""
Generates failure analysis artifacts for Hiver support evaluation:
1. evaluation/results/analysis/confusion_pairs.csv
2. evaluation/results/analysis/escalation_error_analysis.csv
3. evaluation/results/analysis/failure_analysis.json
"""

import json
from collections import Counter
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "evaluation" / "results" / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "evaluation" / "results" / "final" / "golden_eval_results.csv", encoding="utf-8-sig")
gold = pd.read_csv(ROOT / "evaluation" / "golden_set_final.csv", encoding="utf-8-sig")
merged = pd.merge(df, gold, on="example_id", suffixes=("", "_gold"))

# 1. Confusion Pairs CSV
confusions = []
for _, row in df[df["expected_intent"] != df["predicted_intent"]].iterrows():
    confusions.append((row["expected_intent"], row["predicted_intent"]))

confusion_counts = Counter(confusions)
confusion_rows = [
    {"expected_intent": exp, "predicted_intent": pred, "count": cnt}
    for (exp, pred), cnt in confusion_counts.most_common()
]
df_conf = pd.DataFrame(confusion_rows)
df_conf.to_csv(ANALYSIS_DIR / "confusion_pairs.csv", index=False, encoding="utf-8-sig")

# 2. Escalation Error Analysis CSV
def to_bool(val):
    return str(val).strip().lower() in ("true", "1", "yes", "t")

df["exp_esc_bool"] = df["expected_escalate"].apply(to_bool)
df["pred_esc_bool"] = df["predicted_escalate"].apply(to_bool)

escalation_rows = []
for _, row in df.iterrows():
    exp = row["exp_esc_bool"]
    pred = row["pred_esc_bool"]
    if exp and pred:
        cat = "TRUE_POSITIVE"
    elif exp and not pred:
        cat = "FALSE_NEGATIVE"
    elif not exp and pred:
        cat = "FALSE_POSITIVE"
    else:
        cat = "TRUE_NEGATIVE"

    escalation_rows.append({
        "example_id": row["example_id"],
        "customer_text": row["customer_text"],
        "expected_escalate": exp,
        "predicted_escalate": pred,
        "category": cat,
        "expected_escalation_reason": row.get("expected_escalation_reason", ""),
        "predicted_escalation_reason": row.get("predicted_escalation_reason", ""),
        "generated_reply": row.get("generated_reply", ""),
    })

df_esc = pd.DataFrame(escalation_rows)
df_esc.to_csv(ANALYSIS_DIR / "escalation_error_analysis.csv", index=False, encoding="utf-8-sig")

# 3. Comprehensive failure_analysis.json
tp = int(((df["exp_esc_bool"] == True) & (df["pred_esc_bool"] == True)).sum())
fn = int(((df["exp_esc_bool"] == True) & (df["pred_esc_bool"] == False)).sum())
fp = int(((df["exp_esc_bool"] == False) & (df["pred_esc_bool"] == True)).sum())
tn = int(((df["exp_esc_bool"] == False) & (df["pred_esc_bool"] == False)).sum())

clf_rep = classification_report(df["expected_intent"], df["predicted_intent"], output_dict=True)

failure_analysis = {
    "summary": {
        "total_examples": len(df),
        "intent_accuracy": round(float((df["intent_correct"] == True).mean()), 4),
        "macro_f1": round(float(clf_rep["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(clf_rep["weighted avg"]["f1-score"]), 4),
        "escalation_metrics": {
            "accuracy": round(float((df["exp_esc_bool"] == df["pred_esc_bool"]).mean()), 4),
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "true_negatives": tn,
            "precision": round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0,
            "recall": round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0,
            "f1": round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) > 0 else 0.0,
        },
    },
    "per_class_intent_metrics": {
        intent: {
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1_score": round(metrics["f1-score"], 4),
            "support": int(metrics["support"]),
        }
        for intent, metrics in clf_rep.items()
        if intent not in ("accuracy", "macro avg", "weighted avg")
    },
    "top_confusion_pairs": confusion_rows[:15],
    "top_5_failure_modes": [
        {
            "rank": 1,
            "name": "Severe Escalation False Negatives (Frustration, Prior Failed Contacts, Carrier Impasse)",
            "affected_examples_count": fn,
            "sample_example_ids": ["G005", "G008", "G015", "G017"],
            "description": "Customer experiences severe friction (e.g. 2-3 previous failed contacts, carrier lost package, no resolution), yet the model predicts escalate=False and provides standard self-service steps.",
            "root_cause_hypothesis": "The prompt's escalation guidelines prioritize safety and self-service default without a dedicated sentiment/re-contact detection classifier or calibrated threshold.",
            "mitigation": "Add explicit few-shot rules for prior-contact signals ('2nd/3rd time contacting', 'nobody knows', 'unresolved') and implement a dedicated multi-stage escalation classifier with higher sensitivity."
        },
        {
            "rank": 2,
            "name": "ORDER_STATUS Underprediction & Confusion with DELIVERY_DELAY / REFUND_PENDING",
            "affected_examples_count": 18,
            "sample_example_ids": ["G002", "G003", "G071", "G073"],
            "description": "Customers inquiring about overall order state or shipments before dispatch are misclassified into downstream execution states like DELIVERY_DELAY or REFUND_PENDING.",
            "root_cause_hypothesis": "Keywords like 'dispatch', 'prepared for shipment', or 'cancel' trigger specific operational branch prompts rather than general status lookups.",
            "mitigation": "Clarify taxonomy definition: pre-shipment timeline inquiries belong to ORDER_STATUS, whereas confirmed past-due shipments belong to DELIVERY_DELAY."
        },
        {
            "rank": 3,
            "name": "DELIVERY_DELAY vs PACKAGE_NOT_RECEIVED Ambiguity",
            "affected_examples_count": 6,
            "sample_example_ids": ["G032", "G124", "G125", "G126"],
            "description": "In-transit carrier delays (courier not answering, redelivery attempt) get classified as lost packages (PACKAGE_NOT_RECEIVED).",
            "root_cause_hypothesis": "Semantic overlap between 'item has not arrived yet' (delay) and 'package is permanently lost/missing' (not received).",
            "mitigation": "Refine prompt boundary: use PACKAGE_NOT_RECEIVED only when carrier marks delivered or explicitly lost; use DELIVERY_DELAY when tracking indicates active transit/redelivery."
        },
        {
            "rank": 4,
            "name": "PRODUCT_ISSUE Confusion with CUSTOMER_SERVICE_CONTACT / ACCOUNT_SUPPORT",
            "affected_examples_count": 10,
            "sample_example_ids": ["G001", "G033", "G035", "G092"],
            "description": "Customers describing software/hardware defects (app crashing, review submission failure, server errors) are classified as general contact or account issues.",
            "root_cause_hypothesis": "When customers ask 'can someone help with this error in the app', the contact intent overrides the technical defect intent.",
            "mitigation": "Prioritize domain-specific technical defects over generic contact intent when an explicit app/device malfunction is described."
        },
        {
            "rank": 5,
            "name": "REFUND_PENDING Overprediction on Return / Order Status Queries (Low Precision = 31.25%)",
            "affected_examples_count": 11,
            "sample_example_ids": ["G002", "G003", "G075", "G105"],
            "description": "Any customer message mentioning return, cancellation, or money is prematurely classified as REFUND_PENDING even before a return or refund has been initiated.",
            "root_cause_hypothesis": "High semantic salience of refund keywords in the embedding/prompt matching logic.",
            "mitigation": "Restrict REFUND_PENDING to cases where a refund has already been authorized or return received, routing pre-return questions to PRODUCT_ISSUE or ORDER_STATUS."
        }
    ]
}

with open(ANALYSIS_DIR / "failure_analysis.json", "w", encoding="utf-8") as f:
    json.dump(failure_analysis, f, indent=2)

print("Saved failure analysis artifacts successfully.")
