"""
Context Builder for Retrieval-Augmented Generation (RAG).
Formats retrieved historical Amazon support cases into clean, structured prompts
tailored for Phase 3 Gemini text generation.
"""

from typing import Any, Dict, List, Optional


def build_rag_context(
    retrieved_cases: List[Dict[str, Any]],
    predicted_intent: Optional[str] = None,
    max_cases: int = 3
) -> str:
    """
    Constructs a structured historical resolution context block for Gemini RAG prompting.

    Args:
        retrieved_cases: List of dictionaries containing 'customer_text', 'amazon_reply',
                         'intent', and 'score'.
        predicted_intent: The intent predicted by the classifier (e.g. DELIVERY_DELAY).
        max_cases: Maximum number of historical examples to include in the context.

    Returns:
        Formatted string ready to be injected into the LLM prompt.
    """
    if not retrieved_cases:
        intent_header = f"Predicted Customer Intent: {predicted_intent}\n\n" if predicted_intent else ""
        return f"{intent_header}No historical resolution records found for this query."

    cases_to_include = retrieved_cases[:max_cases]
    intent_label = predicted_intent or cases_to_include[0].get("intent", "GENERAL_SUPPORT")

    lines = [
        "### HISTORICAL SUPPORT RESOLUTIONS CONTEXT",
        f"Verified Intent: {intent_label}",
        f"Relevant Historical Cases Retrieved: {len(cases_to_include)}",
        ""
    ]

    for idx, case in enumerate(cases_to_include, start=1):
        cust_query = case.get("customer_text", "").strip()
        resolution = case.get("amazon_reply", "").strip()
        score = case.get("score", 0.0)
        case_intent = case.get("intent", intent_label)

        lines.append(f"--- Historical Example {idx} (Relevance Score: {score:.2f}, Intent: {case_intent}) ---")
        lines.append(f"Customer Inquiry: \"{cust_query}\"")
        lines.append(f"Amazon Official Resolution: \"{resolution}\"")
        lines.append("")

    lines.append("### INSTRUCTIONS FOR RESPONSE GENERATION:")
    lines.append("- Adopt the helpful, professional, and empathetic tone of the historical Amazon resolutions above.")
    lines.append("- Provide direct, actionable next steps based on the verified intent.")
    lines.append("- Maintain brand consistency and safety policies.")

    return "\n".join(lines)
