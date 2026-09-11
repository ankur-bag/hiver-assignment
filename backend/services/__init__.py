"""
Backend services package.
Exposes retrieval services, RAG orchestrator, Gemini clients,
context builder, prompt builder, validator, and escalation engine.
"""

from .pinecone_service import (
    PineconeRetrievalService,
    get_pinecone_service,
    retrieve_similar_cases,
)
from .gemini_service import GeminiService, get_gemini_service
from .context_builder import build_rag_context
from .prompt_builder import build_support_prompt
from .response_validator import validate_response, ResponseValidationResult
from .language_service import detect_language, get_language_prompt_instruction
from .escalation_service import evaluate_escalation, EscalationDecision
from .rag_service import RAGService, get_rag_service, chat

__all__ = [
    "PineconeRetrievalService",
    "get_pinecone_service",
    "retrieve_similar_cases",
    "GeminiService",
    "get_gemini_service",
    "build_rag_context",
    "build_support_prompt",
    "validate_response",
    "ResponseValidationResult",
    "detect_language",
    "get_language_prompt_instruction",
    "evaluate_escalation",
    "EscalationDecision",
    "RAGService",
    "get_rag_service",
    "chat",
]
