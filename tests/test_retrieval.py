"""
Unit and Integration Tests for Retrieval Pipeline.
Tests:
1. Delivery query retrieval
2. Refund query retrieval
3. Account query retrieval
4. Multilingual retrieval (Hindi, Spanish, German)
5. Empty and whitespace query handling
6. Pinecone unavailable / failure recovery handling
7. Context builder structure validation
"""

from unittest.mock import MagicMock, patch
import pytest
from backend.services.pinecone_service import PineconeRetrievalService, get_pinecone_service
from backend.services.context_builder import build_rag_context


@pytest.fixture(scope="module")
def retrieval_service():
    """Initializes and returns shared Pinecone retrieval service."""
    return get_pinecone_service()


# =========================================================================
# 1. Normal Query Retrieval Tests (Delivery, Refund, Account)
# =========================================================================

def test_delivery_query_retrieval(retrieval_service):
    """Tests delivery problem retrieval returns relevant cases with high similarity."""
    query = "My package has not arrived yet, it was delayed"
    res = retrieval_service.retrieve_similar_cases(query=query, top_k=3)

    assert "retrieval_status" in res
    assert res["retrieval_status"] in ["success", "no_match"]

    if res["retrieval_status"] == "success":
        results = res["results"]
        assert len(results) > 0
        top = results[0]
        assert "customer_text" in top
        assert "amazon_reply" in top
        assert top["intent"] in ["DELIVERY_DELAY", "PACKAGE_NOT_RECEIVED"]
        assert top["score"] > 0.50


def test_refund_query_retrieval(retrieval_service):
    """Tests refund query returns refund-specific historical cases."""
    query = "I need a refund for my cancelled order"
    res = retrieval_service.retrieve_similar_cases(query=query, top_k=3)

    assert res["retrieval_status"] in ["success", "no_match"]
    if res["retrieval_status"] == "success":
        results = res["results"]
        assert len(results) > 0
        assert results[0]["intent"] == "REFUND_PENDING"


def test_account_query_retrieval(retrieval_service):
    """Tests account access query retrieves password/login troubleshooting cases."""
    query = "I cannot log into my account, password reset link is invalid"
    res = retrieval_service.retrieve_similar_cases(query=query, top_k=3)

    assert res["retrieval_status"] in ["success", "no_match"]
    if res["retrieval_status"] == "success":
        results = res["results"]
        assert len(results) > 0
        assert results[0]["intent"] in ["ACCOUNT_ACCESS", "ACCOUNT_SUPPORT"]


# =========================================================================
# 2. Multilingual Retrieval Tests
# =========================================================================

@pytest.mark.parametrize(
    "query,expected_intents",
    [
        ("Mera order deliver nahi hua abhi tak", ["DELIVERY_DELAY", "PACKAGE_NOT_RECEIVED", "ORDER_STATUS", "CUSTOMER_SERVICE_CONTACT"]),
        ("Mi paquete no ha llegado todavía", ["DELIVERY_DELAY", "PACKAGE_NOT_RECEIVED"]),
        ("Wo ist meine Bestellung? Das Paket ist verspätet", ["DELIVERY_DELAY", "PACKAGE_NOT_RECEIVED", "ORDER_STATUS"]),
    ]
)
def test_multilingual_retrieval(retrieval_service, query, expected_intents):
    """Cross-lingual semantic retrieval aligns non-English queries to English resolutions."""
    res = retrieval_service.retrieve_similar_cases(query=query, top_k=3)
    assert res["retrieval_status"] in ["success", "no_match"]
    if res["retrieval_status"] == "success":
        results = res["results"]
        assert len(results) > 0
        assert results[0]["intent"] in expected_intents


# =========================================================================
# 3. Empty & Whitespace Query Handling
# =========================================================================

def test_empty_query_handling(retrieval_service):
    """Empty query should gracefully return no_match status without throwing exceptions."""
    res = retrieval_service.retrieve_similar_cases(query="")
    assert res["retrieval_status"] == "no_match"
    assert res["results"] == []
    assert res["telemetry"]["number_of_results"] == 0


def test_whitespace_query_handling(retrieval_service):
    """Whitespace-only query should return no_match."""
    res = retrieval_service.retrieve_similar_cases(query="   \n\t   ")
    assert res["retrieval_status"] == "no_match"
    assert res["results"] == []


# =========================================================================
# 4. Pinecone Failure & Unavailable Handling (Mocked)
# =========================================================================

def test_pinecone_unavailable_handling():
    """Simulates connection failure or unreachable Pinecone service."""
    service = PineconeRetrievalService(api_key="mock_key", index_name="mock_index")

    # Mock index.query to raise an exception
    mock_index = MagicMock()
    mock_index.query.side_effect = ConnectionError("Pinecone endpoint unreachable (DNS timeout)")
    service._index = mock_index

    res = service.retrieve_similar_cases(query="Where is my order?")
    assert res["retrieval_status"] == "unavailable"
    assert res["results"] == []
    assert "error" in res
    assert "telemetry" in res
    assert res["telemetry"]["number_of_results"] == 0


# =========================================================================
# 5. Gemini RAG Context Builder Validation
# =========================================================================

def test_context_builder_formatting():
    """Verifies that the RAG context block properly structures retrieved examples."""
    mock_cases = [
        {
            "customer_text": "Where is my package?",
            "amazon_reply": "Please check tracking on your account page.",
            "intent": "DELIVERY_DELAY",
            "score": 0.9412
        },
        {
            "customer_text": "Late delivery status",
            "amazon_reply": "Carrier shows a slight delay.",
            "intent": "DELIVERY_DELAY",
            "score": 0.8871
        }
    ]

    context = build_rag_context(mock_cases, predicted_intent="DELIVERY_DELAY")
    assert "HISTORICAL SUPPORT RESOLUTIONS CONTEXT" in context
    assert "Verified Intent: DELIVERY_DELAY" in context
    assert "Historical Example 1" in context
    assert "Historical Example 2" in context
    assert "Please check tracking" in context
    assert "INSTRUCTIONS FOR RESPONSE GENERATION" in context


def test_context_builder_empty_cases():
    """Empty cases list should return graceful fallback message."""
    context = build_rag_context([], predicted_intent="REFUND_PENDING")
    assert "No historical resolution records found" in context
