"""Deterministic mandatory escalation guardrails."""

import re
from dataclasses import dataclass
from typing import Optional

MANDATORY_PATTERNS = (
    (re.compile(r"\b(hack(?:ed)?|compromis(?:e|ed)|account takeover|identity theft)\b", re.I), "Possible account compromise."),
    (re.compile(r"\b(fraud|unauthori[sz]ed (?:charge|transaction|purchase|order|access))\b", re.I), "Possible fraud or unauthorized activity."),
    (re.compile(r"\b(stolen (?:card|account|password)|unsafe|danger|threat)\b", re.I), "Security or safety-sensitive issue."),
    (re.compile(r"\b(human|agent|manager|supervisor|representative|escalat(?:e|ion))\b", re.I), "Customer explicitly requested human assistance."),
)


@dataclass(frozen=True)
class EscalationDecision:
    should_escalate: bool
    reason: Optional[str] = None


def evaluate_escalation(query: str, model_escalate: bool = False, model_reason: Optional[str] = None, generation_failed: bool = False) -> EscalationDecision:
    for pattern, reason in MANDATORY_PATTERNS:
        if pattern.search(query or ""):
            return EscalationDecision(True, reason)
    if generation_failed:
        return EscalationDecision(True, "Automated support is temporarily unavailable.")
    if model_escalate:
        return EscalationDecision(True, model_reason or "This request requires human review.")
    return EscalationDecision(False, None)
