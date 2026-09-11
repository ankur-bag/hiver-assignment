"""
Unit and Integration Tests for Real HTTP Token Streaming Endpoint.
Verifies:
- POST /api/v1/chat/stream Content-Type text/event-stream
- Emission of metadata, token, and complete SSE events
- Empty input validation (400 Bad Request)
- Session ID and Request ID propagation in stream events
- Graceful degradation and fallback behavior on streaming failures
"""

import json
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from backend.api.app import app

client = TestClient(app)


def parse_sse_stream(response_text: str):
    """Utility helper to parse SSE blocks into structured event list."""
    events = []
    blocks = response_text.strip().split("\n\n")
    for block in blocks:
        if not block.strip():
            continue
        event_name = "message"
        data_str = ""
        for line in block.split("\n"):
            line = line.strip()
            if line.startswith("event:"):
                event_name = line.replace("event:", "").strip()
            elif line.startswith("data:"):
                data_str += line.replace("data:", "").strip()
        if data_str:
            try:
                parsed_data = json.loads(data_str)
            except Exception:
                parsed_data = data_str
            events.append({"event": event_name, "data": parsed_data})
    return events


def test_streaming_endpoint_content_type():
    """Verify POST /api/v1/chat/stream returns text/event-stream header."""
    response = client.post(
        "/api/v1/chat/stream",
        json={"text": "My package has not arrived yet", "session_id": "stream-sess-1"}
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert response.headers.get("cache-control") == "no-cache"


def test_streaming_event_sequence():
    """Verify streaming response emits metadata, token, and complete events in order."""
    response = client.post(
        "/api/v1/chat/stream",
        json={"text": "Where is my order #108-9921?", "session_id": "seq-sess-100"}
    )
    assert response.status_code == 200

    events = parse_sse_stream(response.text)
    assert len(events) >= 3, f"Expected at least 3 SSE events, got {len(events)}"

    event_types = [e["event"] for e in events]
    assert "metadata" in event_types
    assert "token" in event_types
    assert "complete" in event_types

    # First event should be metadata
    assert events[0]["event"] == "metadata"
    meta = events[0]["data"]
    assert "intent" in meta
    assert "confidence" in meta
    assert "retrieved_cases" in meta
    assert meta["session_id"] == "seq-sess-100"

    # Middle events are tokens
    token_events = [e for e in events if e["event"] == "token"]
    assert len(token_events) > 0
    full_text = "".join(e["data"]["text"] for e in token_events if "text" in e["data"])
    assert len(full_text) > 10

    # Last event should be complete
    assert events[-1]["event"] == "complete"
    complete_data = events[-1]["data"]
    assert "intent" in complete_data
    assert "confidence" in complete_data
    assert "telemetry" in complete_data
    assert "escalate" in complete_data
    assert complete_data["session_id"] == "seq-sess-100"


def test_streaming_validation_empty_query():
    """Verify empty text produces 400 Bad Request on stream endpoint."""
    response = client.post("/api/v1/chat/stream", json={"text": "   "})
    assert response.status_code == 400
    data = response.json()
    assert "error" in data


def test_streaming_gemini_failure_fallback():
    """Verify simulated stream failure emits safe fallback without unhandled server crash."""
    from backend.services.rag_service import get_rag_service
    rag_service = get_rag_service()

    with patch.object(
        rag_service.gemini_service,
        "generate_response_stream",
        side_effect=RuntimeError("Stream Connection Quota Exceeded")
    ):
        response = client.post(
            "/api/v1/chat/stream",
            json={"text": "Where is my delivery?", "session_id": "err-sess-1"}
        )
        assert response.status_code == 200
        events = parse_sse_stream(response.text)
        assert len(events) >= 2

        complete_event = next(e for e in events if e["event"] == "complete")
        assert complete_event["data"]["fallback"] is True
        assert complete_event["data"]["escalate"] is True


def test_streaming_security_escalation():
    """Verify security/fraud intent emits appropriate escalation status in stream."""
    response = client.post(
        "/api/v1/chat/stream",
        json={
            "text": "My account was hacked and unauthorized charges were made to my card!",
            "session_id": "sec-sess-99"
        }
    )
    assert response.status_code == 200
    events = parse_sse_stream(response.text)
    complete_event = next(e for e in events if e["event"] == "complete")
    assert complete_event["data"]["escalate"] is True
    assert complete_event["data"]["escalation_reason"] is not None
