"""Gemini-native File Search customer-support pipeline."""

import logging
import time
import uuid
from threading import Lock
from typing import Any, Optional

from services.escalation_service import evaluate_escalation
from services.gemini_service import (
    CanonicalIntent,
    ProviderError,
    SupportAnalysis,
    extract_intent_and_clean_text,
    get_gemini_service,
)
from services.language_service import detect_language, get_language_prompt_instruction
from services.response_validator import validate_response

logger = logging.getLogger(__name__)

SAFE_FALLBACK = "I’m sorry, but support is temporarily unavailable. Please contact a human support representative for help."


class RAGService:
    def __init__(self, gemini_service=None):
        self.gemini_service = gemini_service or get_gemini_service()

    def _analyze_legacy(self, query: str, language: Optional[str]):
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

        # If mock/custom gemini service doesn't have single-pass stream_chat, use legacy path
        if not hasattr(self.gemini_service, "stream_chat"):
            analysis, grounding, context, analysis_ms = self._analyze_legacy(query, language)
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
                "telemetry": {
                    "request_id": request_id,
                    "analysis_time_ms": round(analysis_ms, 2),
                    "generation_time_ms": round(generation_ms, 2),
                    "total_time_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            })
            return result

        # Fast Single-Pass Production Path with Semantic Intent Extraction
        lang = language or detect_language(query)
        lang_instruction = get_language_prompt_instruction(lang)
        guardrail = evaluate_escalation(query)

        analysis_ms = (time.perf_counter() - started) * 1000
        generation_start = time.perf_counter()

        raw_chunks = []
        grounding = []
        for token, g_chunks, raw_chunk in self.gemini_service.stream_chat(query, lang, lang_instruction):
            if raw_chunk:
                raw_chunks.append(raw_chunk)
            elif token:
                raw_chunks.append(token)
            if g_chunks:
                grounding.extend(g_chunks)

        full_raw_text = "".join(raw_chunks)
        model_intent, clean_reply = extract_intent_and_clean_text(full_raw_text)

        if guardrail.should_escalate:
            final_intent = CanonicalIntent.ESCALATION if not model_intent else model_intent
        else:
            final_intent = model_intent or CanonicalIntent.CUSTOMER_SERVICE_CONTACT

        validation = validate_response(clean_reply)
        reply = validation.sanitized_text if validation.is_valid else SAFE_FALLBACK
        if not validation.is_valid:
            guardrail = evaluate_escalation(query, generation_failed=True)

        total_ms = (time.perf_counter() - started) * 1000
        gen_ms = (time.perf_counter() - generation_start) * 1000

        logger.info(
            "Chat non-stream telemetry: req_id=%s, total_ms=%.1f, gen_ms=%.1f, cases=%d, intent=%s, escalate=%s",
            request_id, total_ms, gen_ms, len(grounding), final_intent.value, guardrail.should_escalate
        )

        return {
            "request_id": request_id,
            "reply": reply,
            "intent": final_intent.value,
            "language": lang,
            "retrieved_context": len(grounding) if grounding else "available",
            "retrieved_cases": len(grounding),
            "escalate": guardrail.should_escalate,
            "escalation_reason": guardrail.reason,
            "grounding_metadata": grounding,
            "fallback": not validation.is_valid,
            "telemetry": {
                "request_id": request_id,
                "analysis_time_ms": round(analysis_ms, 2),
                "generation_time_ms": round(gen_ms, 2),
                "total_time_ms": round(total_ms, 2),
            },
        }

    def chat_stream(self, query: str, language: Optional[str] = None):
        request_id = str(uuid.uuid4())
        started = time.perf_counter()

        # If mock/custom gemini service doesn't have single-pass stream_chat, use legacy path
        if not hasattr(self.gemini_service, "stream_chat"):
            try:
                analysis, grounding, context, analysis_ms = self._analyze_legacy(query, language)
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
                yield {
                    "event": "complete",
                    "data": {
                        **metadata,
                        "fallback": False,
                        "telemetry": {
                            "request_id": request_id,
                            "analysis_time_ms": round(analysis_ms, 2),
                            "generation_time_ms": round((time.perf_counter() - generation_start) * 1000, 2),
                            "total_time_ms": round((time.perf_counter() - started) * 1000, 2),
                        },
                    },
                }
            except ProviderError as exc:
                yield {"event": "error", "data": {"code": exc.code, "message": str(exc), "request_id": request_id, "retryable": exc.retryable}}
            return

        # Fast Single-Pass Streaming Path with Progressive SSE Events
        try:
            analysis_start = time.perf_counter()
            lang = language or detect_language(query)
            lang_instruction = get_language_prompt_instruction(lang)
            guardrail = evaluate_escalation(query)
            analysis_complete = time.perf_counter()
            analysis_ms = (analysis_complete - analysis_start) * 1000

            # 1. Emit Stage 1: analyzing
            yield {"event": "status", "data": {"stage": "analyzing", "request_id": request_id}}

            # If deterministic security rule fired immediately, emit escalation early
            if guardrail.should_escalate:
                yield {"event": "escalation", "data": {"escalate": True, "reason": guardrail.reason, "request_id": request_id}}

            # 2. Emit Stage 2: retrieving (File Search store query begins)
            yield {"event": "status", "data": {"stage": "retrieving", "request_id": request_id}}

            # 3. Stream tokens directly from Gemini File Search
            generation_start = time.perf_counter()
            first_token_time = None
            raw_text_chunks = []
            grounding_chunks_accum = []
            has_emitted_generating_status = False

            for token, g_chunks, raw_chunk in self.gemini_service.stream_chat(query, lang, lang_instruction):
                if g_chunks:
                    grounding_chunks_accum.extend(g_chunks)
                    yield {
                        "event": "grounding",
                        "data": {
                            "retrieved_context_available": True,
                            "retrieved_context_count": len(grounding_chunks_accum),
                            "grounding_metadata": grounding_chunks_accum,
                            "request_id": request_id,
                        },
                    }

                if raw_chunk:
                    raw_text_chunks.append(raw_chunk)
                elif token:
                    raw_text_chunks.append(token)

                if token:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    if not has_emitted_generating_status:
                        has_emitted_generating_status = True
                        yield {"event": "status", "data": {"stage": "generating", "request_id": request_id}}
                    yield {"event": "token", "data": {"text": token}}

            generation_complete = time.perf_counter()
            full_raw_text = "".join(raw_text_chunks)
            model_intent, clean_reply = extract_intent_and_clean_text(full_raw_text)

            if guardrail.should_escalate:
                final_intent = CanonicalIntent.ESCALATION if not model_intent else model_intent
            else:
                final_intent = model_intent or CanonicalIntent.CUSTOMER_SERVICE_CONTACT

            # 4. Response validation and final escalation check
            validation = validate_response(clean_reply)
            if not validation.is_valid:
                guardrail = evaluate_escalation(query, generation_failed=True)

            total_ms = (generation_complete - started) * 1000
            gen_ms = (generation_complete - generation_start) * 1000
            ttft_ms = (first_token_time - started) * 1000 if first_token_time else total_ms

            logger.info(
                "Chat telemetry: req_id=%s, ttft_ms=%.1f, total_ms=%.1f, analysis_ms=%.1f, gen_ms=%.1f, cases=%d, intent=%s, escalate=%s",
                request_id, ttft_ms, total_ms, analysis_ms, gen_ms, len(grounding_chunks_accum), final_intent.value, guardrail.should_escalate
            )

            # 5. Emit progressive updates for intent, grounding, and escalation
            yield {"event": "metadata", "data": {"request_id": request_id, "intent": final_intent.value, "language": lang}}
            yield {
                "event": "grounding",
                "data": {
                    "retrieved_context_available": bool(grounding_chunks_accum),
                    "retrieved_context_count": len(grounding_chunks_accum),
                    "grounding_metadata": grounding_chunks_accum,
                    "request_id": request_id,
                },
            }
            yield {"event": "escalation", "data": {"escalate": guardrail.should_escalate, "reason": guardrail.reason, "request_id": request_id}}

            # 6. Final complete event
            complete_metadata = {
                "request_id": request_id,
                "intent": final_intent.value,
                "language": lang,
                "retrieved_context": len(grounding_chunks_accum) if grounding_chunks_accum else "available",
                "retrieved_cases": len(grounding_chunks_accum),
                "escalate": guardrail.should_escalate,
                "escalation_reason": guardrail.reason,
                "grounding_metadata": grounding_chunks_accum,
                "fallback": not validation.is_valid,
                "telemetry": {
                    "request_id": request_id,
                    "ttft_ms": round(ttft_ms, 2),
                    "analysis_time_ms": round(analysis_ms, 2),
                    "generation_time_ms": round(gen_ms, 2),
                    "total_time_ms": round(total_ms, 2),
                },
            }
            yield {"event": "complete", "data": complete_metadata}

        except ProviderError as exc:
            yield {
                "event": "error",
                "data": {
                    "code": exc.code,
                    "message": str(exc),
                    "request_id": request_id,
                    "retryable": exc.retryable,
                },
            }


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
