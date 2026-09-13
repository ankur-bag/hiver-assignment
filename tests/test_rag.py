from services.gemini_service import (
    CanonicalIntent,
    SupportAnalysis,
    extract_intent_and_clean_text,
)
from services.rag_service import RAGService


class FakeGemini:
    def analyze(self, query, language):
        return SupportAnalysis(intent=CanonicalIntent.PACKAGE_NOT_RECEIVED, escalate=False, language=language, retrieved_context_available=True), [{"title": "case"}], ["A relevant historical resolution"]

    def stream_response(self, query, analysis, context):
        yield "Please check Your Orders "
        yield "for the latest delivery status."

    def generate_response(self, query, analysis, context):
        return "".join(self.stream_response(query, analysis, context))


class FakeSinglePassGemini:
    def stream_chat(self, query, language, language_instruction=None):
        if "wrong order" in query or "wrong item" in query:
            yield "I am sorry you received the wrong item. ", [{"title": "case1", "uri": "uri1"}], "I am sorry you received the wrong item. "
            yield None, [], "\n\n[INTENT: PRODUCT_ISSUE]"
        elif "hacked" in query:
            yield "Please secure your account immediately. ", [{"title": "case2", "uri": "uri2"}], "Please secure your account immediately. "
            yield None, [], "\n\n[INTENT: ACCOUNT_ACCESS]"
        else:
            yield "Here is your order information. ", [{"title": "case3", "uri": "uri3"}], "Here is your order information. "
            yield None, [], "\n\n[INTENT: ORDER_STATUS]"


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


def test_semantic_intent_extraction_and_cleaning():
    raw = "I am so sorry to hear that you got the wrong item.\n\n[INTENT: PRODUCT_ISSUE]"
    intent, clean = extract_intent_and_clean_text(raw)
    assert intent == CanonicalIntent.PRODUCT_ISSUE
    assert clean == "I am so sorry to hear that you got the wrong item."
    assert "[INTENT:" not in clean


def test_wrong_order_classifies_as_product_issue():
    result = RAGService(FakeSinglePassGemini()).chat("i got wrong order", "en")
    assert result["intent"] == "PRODUCT_ISSUE"
    assert result["escalate"] is False
    assert "[INTENT:" not in result["reply"]


def test_security_issue_classifies_into_issue_intent_with_independent_escalation():
    result = RAGService(FakeSinglePassGemini()).chat("Someone hacked my account and changed my password", "en")
    assert result["intent"] == "ACCOUNT_ACCESS"
    assert result["intent"] != "ESCALATION"
    assert result["escalate"] is True
    assert "compromise" in result["escalation_reason"].lower()

