import json
import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from services.chat_service import ChatService, get_chat_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])


class ChatRequest(BaseModel):
    """Client request schema for customer chat queries."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The customer's message or query"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional unique session identifier for tracking conversation context"
    )
    language: Optional[str] = Field(
        default=None,
        description="Optional language code override (e.g., 'en', 'hi', 'es')"
    )


class ChatResponse(BaseModel):
    """Standardized API response contract."""
    reply: str
    intent: str
    confidence: float
    retrieved_cases: int
    escalate: bool
    escalation_reason: Optional[str] = None
    language: str
    session_id: str
    request_id: str
    telemetry: Dict[str, Any] = {}


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    payload: ChatRequest,
    request: Request,
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    Core customer support AI chat endpoint.
    Processes query, classifies intent, retrieves past AmazonHelp resolutions,
    synthesizes brand-aligned reply with Gemini, and detects escalation triggers.
    """
    clean_text = payload.text.strip()
    if not clean_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message text cannot be empty or only whitespace."
        )

    # Attach request_id from middleware if available
    req_id = getattr(request.state, "request_id", None)

    try:
        result = chat_service.process_chat_message(
            query=clean_text,
            session_id=payload.session_id,
            language=payload.language
        )
        if req_id:
            result["request_id"] = req_id
            if "telemetry" in result:
                result["telemetry"]["request_id"] = req_id

        return result
    except Exception as exc:
        logger.error("Unhandled error processing chat query: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Support service temporarily unavailable"
        )


@router.post("/chat/stream")
async def chat_stream_endpoint(
    payload: ChatRequest,
    request: Request,
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    Real HTTP token streaming endpoint.
    Returns Server-Sent Events (SSE):
    - event: metadata (immediate intent & retrieval stats)
    - event: token (incremental generated response tokens)
    - event: complete (final telemetry, safety, and escalation data)
    - event: error (if an unrecoverable failure occurs)
    """
    clean_text = payload.text.strip()
    if not clean_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message text cannot be empty or only whitespace."
        )

    req_id = getattr(request.state, "request_id", None)

    async def event_generator():
        try:
            for event in chat_service.process_chat_message_stream(
                query=clean_text,
                session_id=payload.session_id,
                language=payload.language
            ):
                event_name = event.get("event", "message")
                event_data = event.get("data", {})
                if req_id and isinstance(event_data, dict):
                    if "request_id" not in event_data:
                        event_data["request_id"] = req_id
                    if "telemetry" in event_data and isinstance(event_data["telemetry"], dict):
                        event_data["telemetry"]["request_id"] = req_id

                json_str = json.dumps(event_data, ensure_ascii=False)
                yield f"event: {event_name}\ndata: {json_str}\n\n"
        except Exception as exc:
            logger.error("Unhandled error in chat streaming generator: %s", exc, exc_info=True)
            err_data = json.dumps({
                "error": "Support service temporarily unavailable",
                "request_id": req_id or ""
            })
            yield f"event: error\ndata: {err_data}\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "Content-Type": "text/event-stream; charset=utf-8",
    }
    if req_id:
        headers["X-Request-ID"] = req_id

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=headers
    )

