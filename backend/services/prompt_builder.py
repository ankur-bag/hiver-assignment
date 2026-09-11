"""
Production Prompt Builder for Gemini Customer Support RAG.
Constructs grounded, enterprise-grade prompts instructing Gemini to act
as an empathetic, accurate customer support representative based on verified intent
and retrieved historical Amazon resolutions.
"""

from typing import Any, Dict, List, Optional


def build_support_prompt(
    customer_query: str,
    detected_intent: str,
    confidence_score: float,
    historical_cases: List[Dict[str, Any]],
    target_language: str = "English"
) -> str:
    """
    Constructs an enterprise customer support prompt for Gemini.

    Args:
        customer_query: The customer's incoming message.
        detected_intent: Canonical intent from Phase 1 classifier (e.g. DELIVERY_DELAY).
        confidence_score: Classification confidence score (0.0 to 1.0).
        historical_cases: List of matching historical cases from Pinecone.
        target_language: Detected or requested user language (e.g. English, Hindi, Spanish).

    Returns:
        Full prompt string ready for LLM generation.
    """
    # Format historical examples
    examples_block = []
    if historical_cases:
        for idx, case in enumerate(historical_cases[:3], start=1):
            q = case.get("customer_text", "").strip()
            r = case.get("amazon_reply", "").strip()
            sc = case.get("score", 0.0)
            examples_block.append(
                f"Historical Case {idx} [Relevance: {sc:.2f}]:\n"
                f"  Customer asked: \"{q}\"\n"
                f"  Support resolved: \"{r}\""
            )
        formatted_examples = "\n\n".join(examples_block)
    else:
        formatted_examples = "No direct historical records retrieved. Provide standard, helpful support guidance for this category."

    prompt = f"""You are a professional, empathetic, and knowledgeable Customer Support Specialist for an enterprise e-commerce platform.

### CONTEXT & METADATA:
- Customer Query: "{customer_query}"
- Detected Category/Intent: {detected_intent} (Confidence: {confidence_score:.2f})
- Response Language: Respond naturally in {target_language}.

### VERIFIED HISTORICAL SUPPORT RESOLUTIONS:
{formatted_examples}

### STRICT OPERATIONAL GUIDELINES:
1. Grounding: Rely on the official resolutions above to formulate the appropriate policy, next steps, or link guidance.
2. Tone: Helpful, courteous, concise, and professional. Acknowledge frustration empathetically without over-promising.
3. Clarity: Keep your reply focused (2 to 4 sentences). Give direct next steps or ask for clarification if critical info (like order ID) is missing.
4. Accuracy & Hallucination Prevention:
   - Do NOT invent specific account details, tracking numbers, refund amounts, or fake URLs.
   - You may refer customers to their "Your Orders" page or official account settings.
5. STRICT NEGATIVE CONSTRAINTS:
   - NEVER mention that you are an AI, an LLM, a bot, or an automated system.
   - NEVER mention prompts, instructions, datasets, training data, vectors, Pinecone, or technical systems.
   - NEVER use meta-language like "Based on the examples provided..." or "According to my database...".
   - Respond directly to the customer as their support agent.

CUSTOMER RESPONSE:"""

    return prompt.strip()
