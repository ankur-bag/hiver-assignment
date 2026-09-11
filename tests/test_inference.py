"""
Unit Tests for Intent Inference Pipeline.
Validates:
- Standard customer queries across distinct intents
- Multilingual and cross-lingual customer queries (Hindi, Spanish, German, French)
- Empty, whitespace-only, and invalid edge case inputs
- Out-of-domain / unknown queries and fallback thresholding
- Batch inference consistency
- Output schema contracts
"""

import pytest
from backend.ml.config import config
from backend.ml.inference import get_intent_classifier


@pytest.fixture(scope="module")
def classifier():
    """Initializes and caches the singleton classifier instance for all test cases."""
    return get_intent_classifier()


# =========================================================================
# 1. Output Schema Contract Tests
# =========================================================================

def test_output_schema_contract(classifier):
    """Verifies that the output strictly adheres to required contract."""
    query = "My package has not arrived yet"
    result = classifier.predict(query)

    assert isinstance(result, dict), "Result must be a dictionary"
    assert "intent" in result, "Result dictionary must contain 'intent'"
    assert "confidence" in result, "Result dictionary must contain 'confidence'"

    assert isinstance(result["intent"], str), "Intent must be a string"
    assert isinstance(result["confidence"], (float, int)), "Confidence must be a numeric float"
    assert 0.0 <= result["confidence"] <= 1.0, "Confidence must be between 0.0 and 1.0"
    assert result["intent"] in config.SUPPORTED_INTENTS, f"Intent {result['intent']} not recognized"


# =========================================================================
# 2. Normal Customer Queries
# =========================================================================

@pytest.mark.parametrize(
    "query,expected_intents",
    [
        ("My package has not arrived yet", ["PACKAGE_NOT_RECEIVED", "DELIVERY_DELAY"]),
        ("Where is my order tracking number?", ["ORDER_STATUS"]),
        ("I need a refund for my cancelled order", ["REFUND_PENDING"]),
        ("I cannot login to my account, password reset link not working", ["ACCOUNT_ACCESS"]),
        ("The item is defective and damaged", ["PRODUCT_ISSUE", "REFUND_PENDING"]),
        ("I want to speak with a human agent or manager immediately", ["ESCALATION", "CUSTOMER_SERVICE_CONTACT"]),
    ]
)
def test_normal_customer_queries(classifier, query, expected_intents):
    """Ensures standard English customer queries are classified into correct intents."""
    result = classifier.predict(query, include_metadata=True)
    assert result["intent"] in expected_intents, (
        f"Query '{query}' classified as '{result['intent']}', expected one of {expected_intents}"
    )
    assert result["confidence"] >= 0.40, (
        f"Query '{query}' produced lower than expected confidence: {result['confidence']}"
    )


# =========================================================================
# 3. Multilingual Queries (Cross-lingual generalization)
# =========================================================================

@pytest.mark.parametrize(
    "query,language,expected_intents",
    [
        ("Mera package delay ho gaya hai abhi tak", "Hindi/Hinglish", ["DELIVERY_DELAY", "PACKAGE_NOT_RECEIVED"]),
        ("Mera order deliver nahi hua", "Hindi/Hinglish", ["ORDER_STATUS", "DELIVERY_DELAY", "PACKAGE_NOT_RECEIVED"]),
        ("Mujhe refund kab milega mere order ka?", "Hindi/Hinglish", ["REFUND_PENDING"]),
        ("Mi paquete no ha llegado todavía", "Spanish", ["PACKAGE_NOT_RECEIVED", "DELIVERY_DELAY"]),
        ("Necesito un reembolso de mi dinero", "Spanish", ["REFUND_PENDING"]),
        ("Wo ist meine Bestellung?", "German", ["ORDER_STATUS", "PACKAGE_NOT_RECEIVED"]),
        ("Mein Konto ist gesperrt, ich kann mich nicht einloggen", "German", ["ACCOUNT_ACCESS"]),
        ("Mon colis n'est pas encore arrivé", "French", ["PACKAGE_NOT_RECEIVED", "DELIVERY_DELAY"]),
    ]
)
def test_multilingual_queries(classifier, query, language, expected_intents):
    """Verifies that the multilingual sentence transformer handles non-English queries accurately."""
    result = classifier.predict(query)
    assert result["intent"] in expected_intents, (
        f"Multilingual query ({language}) '{query}' classified as '{result['intent']}', "
        f"expected one of {expected_intents}"
    )
    assert result["confidence"] > 0.35, (
        f"Multilingual query ({language}) confidence too low: {result['confidence']}"
    )


# =========================================================================
# 4. Empty, Whitespace, and Edge Case Inputs
# =========================================================================

def test_empty_string_input(classifier):
    """Empty string should return fallback intent with confidence 0.0 without crashing."""
    result = classifier.predict("")
    assert result["intent"] == config.FALLBACK_INTENT
    assert result["confidence"] == 0.0


def test_whitespace_only_input(classifier):
    """Whitespace-only query should return fallback intent with confidence 0.0."""
    result = classifier.predict("     \n\t   ")
    assert result["intent"] == config.FALLBACK_INTENT
    assert result["confidence"] == 0.0


def test_none_input_handling(classifier):
    """None input should be safely caught and return fallback gracefully."""
    result = classifier.predict(None)  # type: ignore
    assert result["intent"] == config.FALLBACK_INTENT
    assert result["confidence"] == 0.0


def test_excessively_long_input(classifier):
    """Excessively long inputs should be sanitized and truncated cleanly."""
    long_text = "My package has not arrived yet. " * 200
    result = classifier.predict(long_text)
    assert result["intent"] in ["PACKAGE_NOT_RECEIVED", "DELIVERY_DELAY"]
    assert result["confidence"] > 0.0


# =========================================================================
# 5. Out-of-Domain / Unknown Queries
# =========================================================================

def test_unknown_out_of_domain_query(classifier):
    """Out-of-domain queries should still produce a valid schema and confidence score."""
    nonsense_query = "who won the 1994 world cup?"
    result = classifier.predict(nonsense_query, include_metadata=True)

    # Must provide valid structure
    assert "intent" in result
    assert "confidence" in result
    assert "is_fallback" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["intent"] in config.SUPPORTED_INTENTS



# =========================================================================
# 6. Batch Inference Consistency
# =========================================================================

def test_batch_prediction_consistency(classifier):
    """Batch predictions must match individual sequential predictions."""
    queries = [
        "Where is my package?",
        "I need a refund right now",
        "My password is reset",
        "",
        "Product is broken"
    ]

    batch_results = classifier.predict_batch(queries)
    assert len(batch_results) == len(queries)

    for query, batch_res in zip(queries, batch_results):
        single_res = classifier.predict(query)
        assert batch_res["intent"] == single_res["intent"]
        assert batch_res["confidence"] == single_res["confidence"]
