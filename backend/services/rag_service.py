"""Gemini-native File Search customer-support pipeline."""

import time
import uuid
from threading import Lock
from typing import Any, Optional

from services.escalation_service import evaluate_escalation
from services.gemini_service import ProviderError, SupportAnalysis, get_gemini_service
from services.language_service import detect_language
from services.response_validator import validate_response

SAFE_FALLBACK = "I’m sorry, but support is temporarily unavailable. Please contact a human support representative for help."


class RAGService:
    def __init__(self, gemini_service=None):
        self.gemini_service = gemini_service or get_gemini_service()

    def _analyze(self, query: str, language: Optional[str]):
        started = time.perf_counter()
        analysis, grounding, context = self.gemini_service.analyze(query, language or detect_language(query))
        guardrail = evaluate_escalation(query, analysis.escalate, analysis.escalation_reason)
        analysis.escalate = guardrail.should_escalate
        analysis.escalation_reason = guardrail.reason
        return analysis, grounding, context, (time.perf_counter() - started) * 1000

    @staticmethod
    def _metadata(analysis: SupportAnalysis, grounding: list[dict[str, Any]], request_id: str) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "intent": analysis.intent.value,
            "language": analysis.language,
            "retrieved_context": len(grounding) if grounding else ("available" if analysis.retrieved_context_available else "unavailable"),
            "retrieved_cases": len(grounding),
            "escalate": analysis.escalate,
            "escalation_reason": analysis.escalation_reason,
            "grounding_metadata": grounding,
        }

    def chat(self, query: str, language: Optional[str] = None) -> dict[str, Any]:
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        analysis, grounding, context, analysis_ms = self._analyze(query, language)
        generation_start = time.perf_counter()
        reply = self.gemini_service.generate_response(query, analysis, context)
        validation = validate_response(reply)
        if not validation.is_valid:
            reply = SAFE_FALLBACK
            decision = evaluate_escalation(query, analysis.escalate, analysis.escalation_reason, generation_failed=True)
            analysis.escalate, analysis.escalation_reason = decision.should_escalate, decision.reason
        generation_ms = (time.perf_counter() - generation_start) * 1000
        result = self._metadata(analysis, grounding, request_id)
        result.update({
            "reply": validation.sanitized_text if validation.is_valid else reply,
            "fallback": not validation.is_valid,
            "telemetry": {"request_id": request_id, "analysis_time_ms": round(analysis_ms, 2), "generation_time_ms": round(generation_ms, 2), "total_time_ms": round((time.perf_counter()-started)*1000, 2)},
        })
        return result

    def chat_stream(self, query: str, language: Optional[str] = None):
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        try:
            analysis, grounding, context, analysis_ms = self._analyze(query, language)
            metadata = self._metadata(analysis, grounding, request_id)
            yield {"event": "metadata", "data": metadata}
            generation_start = time.perf_counter()
            full_reply = []
            for token in self.gemini_service.stream_response(query, analysis, context):
                full_reply.append(token)
                yield {"event": "token", "data": {"text": token}}
            validation = validate_response("".join(full_reply))
            if not validation.is_valid:
                raise ProviderError("Generated response did not pass safety validation")
            yield {"event": "complete", "data": {**metadata, "fallback": False, "telemetry": {"request_id": request_id, "analysis_time_ms": round(analysis_ms, 2), "generation_time_ms": round((time.perf_counter()-generation_start)*1000, 2), "total_time_ms": round((time.perf_counter()-started)*1000, 2)}}}
        except ProviderError as exc:
            yield {"event": "error", "data": {"code": exc.code, "message": str(exc), "request_id": request_id, "retryable": exc.retryable}}


_instance = None
_lock = Lock()


def get_rag_service() -> RAGService:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = RAGService()
    return _instance


def chat(query: str, language: Optional[str] = None):
    return get_rag_service().chat(query, language)


def chat_stream(query: str, language: Optional[str] = None):
    return get_rag_service().chat_stream(query, language)
