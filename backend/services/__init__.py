"""
Backend services package.
Exposes retrieval services, RAG context builder, and Pinecone clients.
"""

from .pinecone_service import (
    PineconeRetrievalService,
    get_pinecone_service,
    retrieve_similar_cases,
)
from .context_builder import build_rag_context

__all__ = [
    "PineconeRetrievalService",
    "get_pinecone_service",
    "retrieve_similar_cases",
    "build_rag_context",
]
