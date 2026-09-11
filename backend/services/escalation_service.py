"""
Customer Support Escalation Decision Engine.
Evaluates deterministic rules to decide when a conversation must be
transferred or escalated to a human customer support specialist.
"""

import re
from typing import Any, Dict, List, Optional

# Security and high-risk intent keywords
SECURITY_FRAUD_PATTERNS = [
    re.compile(r"\bunauthorized\s+(charge|transaction|order|purchase|access)\b", re.IGNORECASE),
    re.compile(r"\bcredit\s+card\s+fraud\b", re.IGNORECASE),
    re.compile(r"\bidentity\s+theft\b", re.IGNORECASE),
    re.compile(r"\b(hacked|compromised)\b", re.IGNORECASE),
    re.compile(r"\bstolen\s+(card|account|password)\b", re.IGNORECASE),
    re.compile(r"\blegal\s+(action|lawsuit|attorney|lawyer)\b", re.IGNORECASE),
]

MIN_CONFIDENCE_THRESHOLD = 0.40


class EscalationDecision:
    """Represents an escalation decision outcome."""
    def __init__(self, should_escalate: bool, reason: Optional[str] = None):
        self.should_escalate = should_escalate
        self.reason = reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "escalate": self.should_escalate,
            "reason": self.reason
        }


def evaluate_escalation(
    query: str,
    predicted_intent: str,
    confidence: float,
    retrieval_status: str = "success",
    generation_failed: bool = False
) -> EscalationDecision:
    """
    Evaluates multi-factor escalation rules.

    Args:
        query: Customer query string.
        predicted_intent: Intent from classifier.
        confidence: Prediction confidence score.
        retrieval_status: Status from Pinecone retrieval ('success', 'no_match', 'unavailable').
        generation_failed: Flag indicating whether LLM generation encountered an unrecoverable failure.

    Returns:
        EscalationDecision object.
    """
    query_str = query or ""

    # Rule 1: Explicit Escalation Intent
    if predicted_intent == "ESCALATION":
        return EscalationDecision(
            should_escalate=True,
            reason="Customer explicitly requested manager, supervisor, or escalation."
        )

    # Rule 2: Security, fraud, or legal threat keywords
    for pattern in SECURITY_FRAUD_PATTERNS:
        match = pattern.search(query_str)
        if match:
            return EscalationDecision(
                should_escalate=True,
                reason=f"High-risk account issue detected: '{match.group(0)}'."
            )

    # Rule 3: Low classification confidence (ambiguous inquiry)
    if confidence < MIN_CONFIDENCE_THRESHOLD:
        return EscalationDecision(
            should_escalate=True,
            reason=f"Low intent confidence ({confidence:.2f} < {MIN_CONFIDENCE_THRESHOLD:.2f})."
        )

    # Rule 4: Generation failure or safety validator rejection
    if generation_failed:
        return EscalationDecision(
            should_escalate=True,
            reason="Automated response generation could not produce a safe, validated reply."
        )

    return EscalationDecision(should_escalate=False, reason=None)
