"""
Chat Route Handlers.
Exposes POST /api/v1/chat.
Delegates to ChatService and enforces strict request/response schemas.
"""

import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from backend.services.chat_service import ChatService, get_chat_service

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
