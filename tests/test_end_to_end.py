"""
End-to-End System Integration Test (Addon 12).
Verifies complete pipeline from simulated frontend request through:
Frontend Request -> API -> Intent Classifier -> Pinecone Retrieval -> Gemini Generation -> Response Validation -> Final Contract Response.
"""

import pytest
from fastapi.testclient import TestClient
from backend.api.app import app

client = TestClient(app)


def test_end_to_end_delivery_delay():
    """
    Simulates a real customer asking about a late order.
    Validates complete pipeline from HTTP request to RAG synthesis and response schema.
    """
    payload = {
        "text": "My package with order #102-998812 was supposed to be delivered yesterday but it hasn't arrived yet.",
        "session_id": "e2e-session-001"
    }

    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200

    data = response.json()

    # 1. Correct Schema & Identifiers
    assert "reply" in data and len(data["reply"]) > 10
    assert "intent" in data
    assert "confidence" in data and isinstance(data["confidence"], (float, int))
    assert "retrieved_cases" in data and isinstance(data["retrieved_cases"], int)
    assert "escalate" in data and isinstance(data["escalate"], bool)
    assert "language" in data
    assert data["session_id"] == "e2e-session-001"
    assert "request_id" in data and len(data["request_id"]) > 0

    # 2. Intent detection matches delivery domain
    assert data["intent"] in ["DELIVERY_DELAY", "ORDER_STATUS", "PACKAGE_NOT_RECEIVED"]

    # 3. Telemetry is populated
    assert "telemetry" in data
    assert "inference_time_ms" in data["telemetry"]
    assert "retrieval_time_ms" in data["telemetry"]
    assert "generation_time_ms" in data["telemetry"]
    assert "total_time_ms" in data["telemetry"]


def test_end_to_end_security_escalation():
    """
    Simulates an unauthorized account access attempt.
    Verifies that pipeline correctly triggers escalation.
    """
    payload = {
        "text": "I cannot log in to my Amazon account, my password was changed and my account was hacked!",
        "session_id": "e2e-session-security"
    }

    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["intent"] in ["ACCOUNT_ACCESS", "CUSTOMER_SERVICE_CONTACT"]
    # Security/fraud queries must escalate
    assert data["escalate"] is True
    assert data["escalation_reason"] is not None
    assert len(data["reply"]) > 0
