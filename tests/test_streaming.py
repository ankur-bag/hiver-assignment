from services.rag_service import RAGService
from tests.test_rag import FakeGemini


def test_real_stream_event_order_and_chunks():
    events = list(RAGService(FakeGemini()).chat_stream("Where is my package?", "en"))
    assert [event["event"] for event in events] == ["metadata", "token", "token", "complete"]
    assert events[0]["data"]["retrieved_context"] == 1
    assert "confidence" not in events[-1]["data"]


def test_stream_can_be_aborted_by_consumer():
    stream = RAGService(FakeGemini()).chat_stream("Where is my package?", "en")
    assert next(stream)["event"] == "metadata"
    stream.close()
