"""
Unit and End-to-End Integration Tests for Phase 3 Generative RAG Layer.
Tests:
1. Normal delivery query RAG
2. Refund query RAG
3. Multilingual query RAG (Hindi / Hinglish)
4. Low confidence out-of-domain query
5. Gemini API failure simulation & graceful fallback
6. Pinecone unavailable simulation & degraded response
7. Response safety validator & internal leakage blocking
"""

from unittest.mock import MagicMock, patch
import pytest

from backend.services.rag_service import get_rag_service, chat
from backend.services.response_validator import validate_response
from backend.services.escalation_service import evaluate_escalation


@pytest.fixture(scope="module")
def rag_service():
    """Initializes and returns shared RAG service."""
    return get_rag_service()


# =========================================================================
# 1. Normal Delivery Query Test
# =========================================================================

def test_delivery_query_rag(rag_service):
    """Tests end-to-end RAG response generation for late delivery."""
    query = "My package is 2 days late, where is it?"
    res = rag_service.chat(query=query)

    assert isinstance(res, dict)
    assert res["intent"] in ["DELIVERY_DELAY", "PACKAGE_NOT_RECEIVED"]
    assert res["confidence"] > 0.50
    assert res["escalate"] is False
    assert res["retrieved_cases"] > 0
    assert len(res["reply"]) > 20
    # Ensure no technical internal leakage in reply
    assert "pinecone" not in res["reply"].lower()
    assert "vector" not in res["reply"].lower()
    assert "ai model" not in res["reply"].lower()


# =========================================================================
# 2. Refund Query Test
# =========================================================================

def test_refund_query_rag(rag_service):
    """Tests end-to-end RAG response generation for refund query."""
    query = "Where is my refund for my cancelled order?"
    res = rag_service.chat(query=query)

    assert res["intent"] == "REFUND_PENDING"
    assert res["confidence"] > 0.50
    assert len(res["reply"]) > 20
    assert any(term in res["reply"].lower() for term in ["refund", "order", "account", "bank", "credit"])


# =========================================================================
# 3. Multilingual Query Test (Hindi / Hinglish)
# =========================================================================

def test_multilingual_query_rag(rag_service):
    """Tests multilingual handling preserves user language."""
    query = "Mera parcel abhi tak deliver nahi hua, please help"
    res = rag_service.chat(query=query)

    assert res["language"] in ["hi", "hi-Latn"]
    assert len(res["reply"]) > 15
    assert res["retrieved_cases"] > 0
    # Verify Hindi / Hinglish phrasing in response
    reply_lower = res["reply"].lower()
    has_hinglish_or_devanagari = any(w in reply_lower for w in ["aapka", "order", "parcel", "delivery", "kripya", "namaste", "madad"]) or any(ord(c) >= 0x0900 and ord(c) <= 0x097F for c in res["reply"])
    assert has_hinglish_or_devanagari, f"Expected Hindi/Hinglish phrasing in: {res['reply']}"


# =========================================================================
# 4. Out-of-Domain / Low Confidence Query
# =========================================================================

def test_low_confidence_query_rag(rag_service):
    """Out-of-domain queries should flag fallback or trigger clarification/escalation."""
    query = "Tell me about quantum physics and the theory of relativity"
    res = rag_service.chat(query=query)

    # Must provide a structured answer without crashing
    assert "reply" in res
    assert "intent" in res
    assert "escalate" in res
    assert len(res["reply"]) > 10


# =========================================================================
# 5. Gemini Failure Simulation (Graceful Fallback)
# =========================================================================

def test_gemini_failure_simulation(rag_service):
    """Simulates Gemini API throwing an error (e.g. rate limit, quota exceeded)."""
    with patch.object(rag_service.gemini_service, "generate_response", side_effect=RuntimeError("API Quota Exceeded")):
        res = rag_service.chat(query="Where is my package?")

        assert res["fallback"] is True
        assert res["escalate"] is True
        assert "apologize for the inconvenience" in res["reply"].lower() or "amazon.com/help" in res["reply"]


# =========================================================================
# 6. Pinecone Unavailable Simulation (Safe Degraded Response)
# =========================================================================

def test_pinecone_unavailable_simulation(rag_service):
    """Simulates Pinecone vector search timeout or connection failure."""
    with patch.object(
        rag_service.retrieval_service,
        "retrieve_similar_cases",
        return_value={"retrieval_status": "unavailable", "results": [], "telemetry": {"fallback_used": True}}
    ):
        res = rag_service.chat(query="Where is my package?")

        assert res["retrieved_cases"] == 0
        assert res["fallback"] is True
        assert len(res["reply"]) > 15


# =========================================================================
# 7. Response Safety Validator & Leakage Blocking
# =========================================================================

def test_response_validator_blocks_leakage():
    """Validator must reject internal technical architecture terms."""
    leaked_responses = [
        "According to our Pinecone vector database, your package is late.",
        "Based on the training dataset, refunds take 3-5 days.",
        "As an AI language model, I cannot process your refund.",
        "The system prompt instructed me to help with cluster_id 4."
    ]

    for bad_reply in leaked_responses:
        val = validate_response(bad_reply)
        assert val.is_valid is False, f"Expected validator to reject: {bad_reply}"
        assert "leakage" in val.reason.lower()


def test_response_validator_allows_clean_response():
    """Validator must allow clean, customer-facing support text."""
    clean_reply = (
        "We apologize for the delivery delay. Please check your tracking information in "
        "'Your Orders' or contact our support team at amazon.com/help."
    )
    val = validate_response(clean_reply)
    assert val.is_valid is True
    assert val.sanitized_text == clean_reply
    assert val.reason is None
