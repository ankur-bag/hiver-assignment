from services.gemini_service import CanonicalIntent, SupportAnalysis
from services.rag_service import RAGService


class FakeGemini:
    def analyze(self, query, language):
        return SupportAnalysis(intent=CanonicalIntent.PACKAGE_NOT_RECEIVED, escalate=False, language=language, retrieved_context_available=True), [{"title": "case"}], ["A relevant historical resolution"]

    def stream_response(self, query, analysis, context):
        yield "Please check Your Orders "
        yield "for the latest delivery status."

    def generate_response(self, query, analysis, context):
        return "".join(self.stream_response(query, analysis, context))


def test_file_search_grounding_metadata_and_no_confidence():
    result = RAGService(FakeGemini()).chat("My package has not arrived", "en")
    assert result["intent"] == "PACKAGE_NOT_RECEIVED"
    assert result["retrieved_context"] == 1
    assert "confidence" not in result


def test_mandatory_security_escalation_overrides_model():
    result = RAGService(FakeGemini()).chat("My account was hacked", "en")
    assert result["escalate"] is True
    assert "compromise" in result["escalation_reason"].lower()


def test_multilingual_language_is_preserved():
    result = RAGService(FakeGemini()).chat("Mera parcel abhi tak nahi aaya", "hi-Latn")
    assert result["language"] == "hi-Latn"
