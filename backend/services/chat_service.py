"""
Production Chat Service Layer.
Separates API routing from RAG orchestrator logic.
Handles session tracking, message sanitization, and structured response preparation.
"""

import logging
import uuid
from typing import Any, Dict, Optional
from services.rag_service import get_rag_service

logger = logging.getLogger(__name__)


class ChatService:
    """
    Business service layer managing customer chat sessions and RAG pipeline execution.
    """
    def __init__(self):
        self.rag_service = get_rag_service()

    def process_chat_message(
        self,
        query: str,
        session_id: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes business workflow for an incoming customer chat message.

        Args:
            query: Raw user message.
            session_id: Conversation session identifier (generated if absent).
            language: Optional manual language override.

        Returns:
            Dict matching the API response contract.
        """
        active_session_id = session_id or str(uuid.uuid4())[:8]

        # Call RAG orchestration layer
        rag_output = self.rag_service.chat(
            query=query,
            language=language
        )

        telemetry = rag_output.get("telemetry", {})
        request_id = telemetry.get("request_id", str(uuid.uuid4())[:8])

        # Ensure standard latency field names are present for client telemetry
        standardized_telemetry = dict(telemetry)
        if "intent_latency_ms" in standardized_telemetry:
            standardized_telemetry["inference_time_ms"] = standardized_telemetry["intent_latency_ms"]
        if "retrieval_latency_ms" in standardized_telemetry:
            standardized_telemetry["retrieval_time_ms"] = standardized_telemetry["retrieval_latency_ms"]
        if "gemini_latency_ms" in standardized_telemetry:
            standardized_telemetry["generation_time_ms"] = standardized_telemetry["gemini_latency_ms"]
        if "total_latency_ms" in standardized_telemetry:
            standardized_telemetry["total_time_ms"] = standardized_telemetry["total_latency_ms"]

        return {
            "reply": rag_output.get("reply", ""),
            "intent": rag_output.get("intent", "CUSTOMER_SERVICE_CONTACT"),
            "confidence": rag_output.get("confidence", 0.0),
            "retrieved_cases": rag_output.get("retrieved_cases", 0),
            "escalate": rag_output.get("escalate", False),
            "escalation_reason": rag_output.get("escalation_reason"),
            "language": rag_output.get("language", "en"),
            "session_id": active_session_id,
            "request_id": request_id,
            "telemetry": standardized_telemetry
        }

    def process_chat_message_stream(
        self,
        query: str,
        session_id: Optional[str] = None,
        language: Optional[str] = None
    ):
        """
        Executes streaming business workflow yielding event payloads for SSE.

        Args:
            query: Raw customer query string.
            session_id: Conversation session identifier.
            language: Optional manual language override.

        Yields:
            Dict with 'event' and 'data' keys for SSE streaming.
        """
        active_session_id = session_id or str(uuid.uuid4())[:8]

        for sse_event in self.rag_service.chat_stream(query=query, language=language):
            event_type = sse_event.get("event")
            event_data = sse_event.get("data", {})

            if event_type == "metadata":
                event_data["session_id"] = active_session_id
            elif event_type == "complete":
                event_data["session_id"] = active_session_id
                # Standardize telemetry key names for client consumption
                telemetry = event_data.get("telemetry", {})
                if "intent_latency_ms" in telemetry:
                    telemetry["inference_time_ms"] = telemetry["intent_latency_ms"]
                if "retrieval_latency_ms" in telemetry:
                    telemetry["retrieval_time_ms"] = telemetry["retrieval_latency_ms"]
                if "gemini_latency_ms" in telemetry:
                    telemetry["generation_time_ms"] = telemetry["gemini_latency_ms"]
                if "total_latency_ms" in telemetry:
                    telemetry["total_time_ms"] = telemetry["total_latency_ms"]

            yield {
                "event": event_type,
                "data": event_data
            }



_chat_service_instance = None


def get_chat_service() -> ChatService:
    """Singleton getter for ChatService."""
    global _chat_service_instance
    if _chat_service_instance is None:
        _chat_service_instance = ChatService()
    return _chat_service_instance
