"""
FastAPI Route Unit & Integration Tests.
Verifies:
- GET /api/v1/health operational status
- POST /api/v1/chat successful response schema & headers
- Empty and invalid request handling (400 / 422)
- Security headers presence (X-Request-ID, nosniff, etc.)
- Session ID preservation
"""

import pytest
from fastapi.testclient import TestClient
from backend.api.app import app

client = TestClient(app)


def test_root_index():
    """Verify service root metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Hiver AI Customer Support API"
    assert "endpoints" in data


def test_health_endpoint():
    """Verify /api/v1/health returns structured status without failure."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "services" in data
    assert "intent_model" in data["services"]
    assert "pinecone" in data["services"]
    assert "gemini" in data["services"]


def test_security_headers_present():
    """Verify security headers are injected in all responses."""
    response = client.get("/api/v1/health")
    assert "x-request-id" in response.headers
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"


def test_chat_validation_empty_query():
    """Verify empty query string triggers 400 Bad Request."""
    response = client.post("/api/v1/chat", json={"text": "   "})
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "request_id" in data


def test_chat_validation_missing_field():
    """Verify missing 'text' field triggers 422 validation error."""
    response = client.post("/api/v1/chat", json={"session_id": "test-123"})
    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert "request_id" in data


def test_chat_session_id_forwarding():
    """Verify session_id is preserved in the response."""
    test_session = "cust-sess-9988"
    response = client.post(
        "/api/v1/chat",
        json={"text": "Where is my package? It was supposed to arrive yesterday.", "session_id": test_session}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == test_session
    assert "reply" in data
    assert "intent" in data
    assert "confidence" in data
    assert "retrieved_cases" in data
    assert "escalate" in data
    assert "request_id" in data
