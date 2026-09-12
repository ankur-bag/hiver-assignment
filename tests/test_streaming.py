from unittest.mock import MagicMock
import httpx
from services.gemini_service import GeminiFileSearchService, ProviderTransientError, ProviderError
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
    
    assert "status" in event_names
    assert "token" in event_names
    assert "grounding" in event_names
    assert "escalation" in event_names
    assert "metadata" in event_names
    assert event_names[-1] == "complete"
    
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


def test_primary_readtimeout_before_first_token_triggers_fallback():
    service = GeminiFileSearchService(
        api_key="test_key",
        store_name="fileSearchStores/test",
        model_name="gemini-3.6-flash",
        fallback_model_name="gemini-3.5-flash-lite",
        timeout_seconds=22.0
    )

    call_models = []

    class Chunk:
        def __init__(self, text):
            self.text = text
            self.candidates = []

    def mock_gen_stream(model, contents, config):
        call_models.append(model)
        if model == "gemini-3.6-flash":
            raise httpx.ReadTimeout("The read operation timed out")
        return iter([Chunk("Fallback response token. "), Chunk("\n\n[INTENT: REFUND_PENDING]")])

    mock_client = MagicMock()
    mock_client.models.generate_content_stream = mock_gen_stream
    service._get_client = MagicMock(return_value=mock_client)

    rag = RAGService(service)
    events = list(rag.chat_stream("Where is my refund?", "en"))

    assert call_models == ["gemini-3.6-flash", "gemini-3.5-flash-lite"]
    assert events[-1]["event"] == "complete"
    assert events[-1]["data"]["intent"] == "REFUND_PENDING"


def test_primary_503_before_first_token_triggers_fallback():
    service = GeminiFileSearchService(
        api_key="test_key",
        store_name="fileSearchStores/test",
        model_name="gemini-3.6-flash",
        fallback_model_name="gemini-3.5-flash-lite",
        timeout_seconds=22.0
    )

    call_models = []

    class Chunk:
        def __init__(self, text):
            self.text = text
            self.candidates = []

    def mock_gen_stream(model, contents, config):
        call_models.append(model)
        if model == "gemini-3.6-flash":
            raise Exception("503 Service Unavailable")
        return iter([Chunk("Fallback response. "), Chunk("\n\n[INTENT: PACKAGE_NOT_RECEIVED]")])

    mock_client = MagicMock()
    mock_client.models.generate_content_stream = mock_gen_stream
    service._get_client = MagicMock(return_value=mock_client)

    rag = RAGService(service)
    events = list(rag.chat_stream("Where is my package?", "en"))

    assert call_models == ["gemini-3.6-flash", "gemini-3.5-flash-lite"]
    assert events[-1]["event"] == "complete"
    assert events[-1]["data"]["intent"] == "PACKAGE_NOT_RECEIVED"


def test_primary_failure_after_first_token_does_not_trigger_fallback():
    service = GeminiFileSearchService(
        api_key="test_key",
        store_name="fileSearchStores/test",
        model_name="gemini-3.6-flash",
        fallback_model_name="gemini-3.5-flash-lite",
        timeout_seconds=22.0
    )

    call_models = []

    class Chunk:
        def __init__(self, text):
            self.text = text
            self.candidates = []

    def failing_stream():
        yield Chunk("Partial response already emitted to customer...")
        raise httpx.ReadTimeout("The read operation timed out midstream")

    def mock_gen_stream(model, contents, config):
        call_models.append(model)
        return failing_stream()

    mock_client = MagicMock()
    mock_client.models.generate_content_stream = mock_gen_stream
    service._get_client = MagicMock(return_value=mock_client)

    rag = RAGService(service)
    events = list(rag.chat_stream("Where is my refund?", "en"))

    # Fallback model must NOT have been called because a token was already emitted
    assert call_models == ["gemini-3.6-flash"]
    assert any(e["event"] == "token" for e in events)
    assert events[-1]["event"] == "error"
    assert "interrupted" in events[-1]["data"]["message"].lower()


def test_fallback_failure_emits_error_once():
    service = GeminiFileSearchService(
        api_key="test_key",
        store_name="fileSearchStores/test",
        model_name="gemini-3.6-flash",
        fallback_model_name="gemini-3.5-flash-lite",
        timeout_seconds=22.0
    )

    def mock_gen_stream(model, contents, config):
        raise httpx.ReadTimeout("The read operation timed out")

    mock_client = MagicMock()
    mock_client.models.generate_content_stream = mock_gen_stream
    service._get_client = MagicMock(return_value=mock_client)

    rag = RAGService(service)
    events = list(rag.chat_stream("Where is my refund?", "en"))

    error_events = [e for e in events if e["event"] == "error"]
    assert len(error_events) == 1


def test_primary_first_token_timeout_triggers_fallback_with_no_interleaved_error():
    import time
    service = GeminiFileSearchService(
        api_key="test_key",
        store_name="fileSearchStores/test",
        model_name="gemini-3.6-flash",
        fallback_model_name="gemini-3.5-flash-lite",
        primary_first_token_timeout=0.1,
        fallback_timeout=2.0,
        timeout_seconds=22.0
    )

    class Chunk:
        def __init__(self, text):
            self.text = text
            self.candidates = []

    def hanging_primary():
        time.sleep(0.3)
        yield Chunk("Late primary token")

    def quick_fallback():
        yield Chunk("Immediate fallback token. ")
        yield Chunk("\n\n[INTENT: REFUND_PENDING]")

    def mock_gen_stream(model, contents, config):
        if model == "gemini-3.6-flash":
            return hanging_primary()
        return quick_fallback()

    mock_client = MagicMock()
    mock_client.models.generate_content_stream = mock_gen_stream
    service._get_client = MagicMock(return_value=mock_client)

    rag = RAGService(service)
    events = list(rag.chat_stream("Where is my refund?", "en"))

    event_names = [e["event"] for e in events]
    assert "error" not in event_names
    assert "token" in event_names
    assert events[-1]["event"] == "complete"
    assert events[-1]["data"]["intent"] == "REFUND_PENDING"
