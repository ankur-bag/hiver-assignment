from services.rag_service import RAGService
from tests.test_rag import FakeGemini, FakeSinglePassGemini


def test_real_stream_event_order_and_chunks():
    events = list(RAGService(FakeGemini()).chat_stream("Where is my package?", "en"))
    assert [event["event"] for event in events] == ["metadata", "token", "token", "complete"]
    assert events[0]["data"]["retrieved_context"] == 1
    assert "confidence" not in events[-1]["data"]


def test_stream_can_be_aborted_by_consumer():
    stream = RAGService(FakeGemini()).chat_stream("Where is my package?", "en")
    assert next(stream)["event"] == "metadata"
    stream.close()


def test_single_pass_stream_events_with_semantic_intent():
    events = list(RAGService(FakeSinglePassGemini()).chat_stream("i got wrong order", "en"))
    assert [event["event"] for event in events] == ["metadata", "token", "complete"]
    assert events[0]["data"]["retrieved_context"] == "searching"
    assert events[-1]["data"]["intent"] == "PRODUCT_ISSUE"
    assert events[-1]["data"]["retrieved_cases"] == 1
    assert "ttft_ms" in events[-1]["data"]["telemetry"]
