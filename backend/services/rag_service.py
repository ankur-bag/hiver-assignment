"""
End-to-End Production RAG Customer Support Orchestrator.
Coordinates:
1. Input validation & language detection
2. Phase 1 Intent Classification
3. Phase 2 Pinecone Vector Retrieval (Hybrid 2-stage + Fallback)
4. Escalation Decision Engine
5. Prompt Assembly with verified historical cases
6. Gemini Response Synthesis
7. Response Safety & Leakage Validation
8. Non-PII Telemetry & Observability
"""

import logging
import time
import uuid
from threading import Lock
from typing import Any, Dict, Optional

from ml.inference import get_intent_classifier
from services.pinecone_service import get_pinecone_service
from services.gemini_service import get_gemini_service
from services.prompt_builder import build_support_prompt
from services.response_validator import validate_response
from services.language_service import detect_language, get_language_prompt_instruction
from services.escalation_service import evaluate_escalation

logger = logging.getLogger(__name__)

# Fallback customer-facing resolution when AI service is unavailable or rejected
GENERIC_SUPPORT_FALLBACK = (
    "We apologize for the inconvenience. To assist you with your request right away, "
    "please visit your 'Your Orders' page or contact our live customer service team at amazon.com/help."
)


class RAGService:
    """
    Main conversational RAG service coordinating intent prediction, vector retrieval,
    LLM response synthesis, and safety validation.
    """
    _instance = None
    _lock = Lock()

    def __init__(self):
        self.classifier = get_intent_classifier()
        self.retrieval_service = get_pinecone_service()
        self.gemini_service = get_gemini_service()

    def chat(
        self,
        query: str,
        language: Optional[str] = None,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Executes complete end-to-end RAG chat pipeline.

        Args:
            query: Raw customer query text.
            language: Optional manual language code override (e.g. 'en', 'hi', 'es').
            top_k: Number of historical support resolutions to retrieve.

        Returns:
            Dict matching required contract:
            {
                "reply": str,
                "intent": str,
                "confidence": float,
                "escalate": bool,
                "escalation_reason": Optional[str],
                "retrieved_cases": int,
                "fallback": bool,
                "language": str,
                "telemetry": Dict[str, Any]
            }
        """
        request_id = str(uuid.uuid4())[:8]
        start_total = time.time()

        # Step 1: Input Validation
        if not query or not query.strip():
            return {
                "reply": "Hello! How can I assist you with your Amazon orders, account, or deliveries today?",
                "intent": "CUSTOMER_SERVICE_CONTACT",
                "confidence": 0.0,
                "escalate": False,
                "escalation_reason": None,
                "retrieved_cases": 0,
                "fallback": True,
                "language": "en",
                "telemetry": {
                    "request_id": request_id,
                    "total_latency_ms": 0.0,
                    "gemini_latency_ms": 0.0
                }
            }

        # Step 2: Language Detection
        detected_lang_code = language or detect_language(query)
        lang_instruction = get_language_prompt_instruction(detected_lang_code)

        # Step 3: Phase 1 Intent Classification
        t_clf_start = time.time()
        intent_res = self.classifier.predict(query, include_metadata=True)
        predicted_intent = intent_res["intent"]
        confidence = intent_res["confidence"]
        clf_latency = (time.time() - t_clf_start) * 1000

        # Step 4: Phase 2 Vector Retrieval in Pinecone
        t_ret_start = time.time()
        retrieval_data = self.retrieval_service.retrieve_similar_cases(
            query=query,
            intent=predicted_intent,
            top_k=top_k
        )
        ret_latency = (time.time() - t_ret_start) * 1000
        cases = retrieval_data.get("results", [])
        retrieval_status = retrieval_data.get("retrieval_status", "success")
        fallback_retrieval = retrieval_data.get("telemetry", {}).get("fallback_used", False)

        # Step 5: Escalation Engine Evaluation
        escalation = evaluate_escalation(
            query=query,
            predicted_intent=predicted_intent,
            confidence=confidence,
            retrieval_status=retrieval_status
        )

        # If escalation is already triggered by intent or security, provide warm escalation reply
        if escalation.should_escalate and predicted_intent == "ESCALATION":
            total_latency = (time.time() - start_total) * 1000
            reply = (
                "I understand this requires immediate attention. I am transferring your inquiry "
                "to an Amazon customer support manager who will assist you shortly."
            )
            return {
                "reply": reply,
                "intent": predicted_intent,
                "confidence": confidence,
                "escalate": True,
                "escalation_reason": escalation.reason,
                "retrieved_cases": len(cases),
                "fallback": False,
                "language": detected_lang_code,
                "telemetry": {
                    "request_id": request_id,
                    "intent_latency_ms": round(clf_latency, 2),
                    "retrieval_latency_ms": round(ret_latency, 2),
                    "gemini_latency_ms": 0.0,
                    "total_latency_ms": round(total_latency, 2)
                }
            }

        # Step 6: Assemble Gemini Prompt
        prompt = build_support_prompt(
            customer_query=query,
            detected_intent=predicted_intent,
            confidence_score=confidence,
            historical_cases=cases,
            target_language=lang_instruction
        )

        # Step 7: Call Gemini API
        t_gemini_start = time.time()
        generation_failed = False
        raw_response = ""

        try:
            raw_response = self.gemini_service.generate_response(prompt=prompt)
        except Exception as e:
            logger.error(f"[{request_id}] Gemini generation failed: {e}", exc_info=True)
            generation_failed = True

        gemini_latency = (time.time() - t_gemini_start) * 1000

        # Step 8: Validate and Sanitize Response
        if not generation_failed:
            validation = validate_response(raw_response)
            if validation.is_valid:
                final_reply = validation.sanitized_text
            else:
                logger.warning(f"[{request_id}] Response validation failed: {validation.reason}")
                final_reply = GENERIC_SUPPORT_FALLBACK
                generation_failed = True
        else:
            final_reply = GENERIC_SUPPORT_FALLBACK

        # Re-evaluate escalation if generation encountered failure
        if generation_failed:
            escalation = evaluate_escalation(
                query=query,
                predicted_intent=predicted_intent,
                confidence=confidence,
                generation_failed=True
            )

        total_latency = (time.time() - start_total) * 1000

        # Step 9: Structured Telemetry Logging (Zero PII stored)
        logger.info(
            f"RAG Request | id={request_id} | intent={predicted_intent} | conf={confidence:.2f} | "
            f"cases={len(cases)} | escalate={escalation.should_escalate} | "
            f"gemini_ms={gemini_latency:.1f} | total_ms={total_latency:.1f}"
        )

        return {
            "reply": final_reply,
            "intent": predicted_intent,
            "confidence": confidence,
            "escalate": escalation.should_escalate,
            "escalation_reason": escalation.reason,
            "retrieved_cases": len(cases),
            "fallback": fallback_retrieval or generation_failed,
            "language": detected_lang_code,
            "telemetry": {
                "request_id": request_id,
                "intent_latency_ms": round(clf_latency, 2),
                "retrieval_latency_ms": round(ret_latency, 2),
                "gemini_latency_ms": round(gemini_latency, 2),
                "total_latency_ms": round(total_latency, 2)
            }
        }


# Singleton accessor
_rag_service_instance = None
_rag_lock = Lock()


def get_rag_service() -> RAGService:
    """Returns singleton instance of RAGService."""
    global _rag_service_instance
    if _rag_service_instance is None:
        with _rag_lock:
            if _rag_service_instance is None:
                _rag_service_instance = RAGService()
    return _rag_service_instance


def chat(query: str, language: Optional[str] = None) -> Dict[str, Any]:
    """Convenience entrypoint for RAG chat pipeline."""
    return get_rag_service().chat(query=query, language=language)
