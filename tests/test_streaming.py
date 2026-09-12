from unittest.mock import MagicMock
from services.gemini_service import ProviderTransientError
from services.rag_service import RAGService
from tests.test_rag import FakeGemini, FakeSinglePassGemini


def test_legacy_stream_fallback():
    events = list(RAGService(FakeGemini()).chat_stream("Where is my package?", "en"))
    assert [event["event"] for event in events] == ["metadata", "token", "token", "complete"]
    assert events[0]["data"]["retrieved_context"] == 1
    assert "confidence" not in events[-1]["data"]


def test_stream_can_be_aborted_by_consumer():
    stream = RAGService(FakeGemini()).chat_stream("Where is my package?", "en")
    assert next(stream)["event"] == "metadata"
    stream.close()


def test_progressive_sse_events_sequence():
    events = list(RAGService(FakeSinglePassGemini()).chat_stream("i got wrong order", "en"))
    event_names = [e["event"] for e in events]
    
    # Must contain progressive status, token, metadata, grounding, escalation, and complete
    assert "status" in event_names
    assert "token" in event_names
    assert "grounding" in event_names
    assert "escalation" in event_names
    assert "metadata" in event_names
    assert event_names[-1] == "complete"
    
    # Check progressive status stages
    status_events = [e["data"]["stage"] for e in events if e["event"] == "status"]
    assert "analyzing" in status_events
    assert "retrieving" in status_events
    assert "generating" in status_events
    
    complete_data = events[-1]["data"]
    assert complete_data["intent"] == "PRODUCT_ISSUE"
    assert complete_data["retrieved_cases"] == 1
    assert "ttft_ms" in complete_data["telemetry"]


def test_single_gemini_call_preserved():
    mock_gemini = FakeSinglePassGemini()
    mock_gemini.stream_chat = MagicMock(side_effect=mock_gemini.stream_chat)
    rag = RAGService(mock_gemini)
    
    events = list(rag.chat_stream("Where is my refund?", "en"))
    assert events[-1]["event"] == "complete"
    assert mock_gemini.stream_chat.call_count == 1


def test_provider_error_emits_error_event():
    class ErrorGemini:
        def stream_chat(self, query, language, language_instruction=None):
            raise ProviderTransientError("AI service is temporarily unavailable")
            yield None, [], None

    rag = RAGService(ErrorGemini())
    events = list(rag.chat_stream("My order is delayed", "en"))
    assert any(e["event"] == "error" for e in events)
    error_event = next(e for e in events if e["event"] == "error")
    assert "temporarily unavailable" in error_event["data"]["message"]
