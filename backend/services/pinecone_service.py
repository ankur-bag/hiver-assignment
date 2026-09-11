"""
Production Pinecone Retrieval Service.
Implements:
- Two-stage hybrid semantic retrieval (Intent Classification -> Intent-Filtered Vector Search)
- Dynamic fallback to global search if filtered similarity < threshold or intent confidence is low
- Multi-tenant namespace support (default: 'amazon')
- Observability and telemetry logging without storing PII
- Resilient failure recovery (Pinecone unavailable, no match, timeout)
"""

import logging
import os
import time
import uuid
from threading import Lock
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from pinecone import Pinecone

from ml.embeddings.encoder import get_embedding_encoder
from ml.inference import get_intent_classifier

logger = logging.getLogger(__name__)

# Fallback similarity threshold: if filtered search scores below this, try global search
DEFAULT_SIMILARITY_THRESHOLD = 0.60


class PineconeRetrievalService:
    """
    Singleton service managing Pinecone vector search, hybrid intent filtering,
    and fallback semantic retrieval.
    """
    _instance = None
    _lock = Lock()

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        namespace: str = "amazon",
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ):
        load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "amazon-support-resolutions")
        self.namespace = namespace
        self.similarity_threshold = similarity_threshold

        self.encoder = get_embedding_encoder()
        self.classifier = get_intent_classifier()
        self._index = None
        self._pinecone = None

    def _get_index(self):
        """Lazy-load and cache the Pinecone Index connection."""
        if self._index is None:
            if not self.api_key:
                raise ValueError("PINECONE_API_KEY is not configured.")
            self._pinecone = Pinecone(api_key=self.api_key)
            self._index = self._pinecone.Index(self.index_name)
        return self._index

    def retrieve_similar_cases(
        self,
        query: str,
        intent: Optional[str] = None,
        top_k: int = 5,
        namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes two-stage hybrid retrieval:
        1. Classifies query intent (if not explicitly provided).
        2. Queries Pinecone in the target namespace with intent filter.
        3. If no matches or score < threshold, falls back to global search.
        4. Logs structured telemetry (latency, request_id, fallback status).

        Args:
            query: Raw customer query text.
            intent: Optional pre-classified intent override.
            top_k: Number of historical cases to retrieve.
            namespace: Target namespace (defaults to 'amazon').

        Returns:
            Dict containing:
            {
                "retrieval_status": "success" | "no_match" | "unavailable",
                "results": [
                    {
                        "customer_text": str,
                        "amazon_reply": str,
                        "intent": str,
                        "score": float
                    }
                ],
                "telemetry": {
                    "request_id": str,
                    "predicted_intent": str,
                    "top_similarity_score": float,
                    "number_of_results": int,
                    "fallback_used": bool,
                    "latency_ms": float
                }
            }
        """
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        target_namespace = namespace or self.namespace

        # Handle empty/whitespace input
        if not query or not query.strip():
            latency = (time.time() - start_time) * 1000
            return {
                "retrieval_status": "no_match",
                "results": [],
                "telemetry": {
                    "request_id": request_id,
                    "predicted_intent": "UNKNOWN",
                    "top_similarity_score": 0.0,
                    "number_of_results": 0,
                    "fallback_used": False,
                    "latency_ms": round(latency, 2)
                }
            }

        # Step 1: Detect intent and confidence if not provided
        fallback_used = False
        intent_confidence = 1.0
        predicted_intent = intent

        if not predicted_intent:
            pred = self.classifier.predict(query, include_metadata=True)
            predicted_intent = pred["intent"]
            intent_confidence = pred.get("raw_confidence", pred["confidence"])
            if pred.get("is_fallback", False):
                fallback_used = True

        # Step 2: Encode query text into 384d vector
        query_vector = self.encoder.encode(query, show_progress_bar=False).tolist()

        # Step 3: Vector search with Pinecone failure recovery
        try:
            index = self._get_index()
            
            # Decide whether to apply intent filter or skip to global (Addon 9)
            use_filter = (predicted_intent is not None) and (intent_confidence >= 0.40) and not fallback_used
            filter_dict = {"intent": {"$eq": predicted_intent}} if use_filter else None

            # Primary Query
            res = index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=target_namespace,
                filter=filter_dict,
                include_metadata=True
            )

            matches = res.get("matches", [])
            top_score = matches[0]["score"] if matches else 0.0

            # Addon 5: Fallback to global search if top similarity is below threshold
            if (not matches or top_score < self.similarity_threshold) and use_filter:
                logger.info(
                    f"[{request_id}] Filtered search top score ({top_score:.2f}) < threshold "
                    f"({self.similarity_threshold}). Falling back to global search."
                )
                global_res = index.query(
                    vector=query_vector,
                    top_k=top_k,
                    namespace=target_namespace,
                    filter=None,  # No intent filter
                    include_metadata=True
                )
                if global_res.get("matches"):
                    matches = global_res["matches"]
                    top_score = matches[0]["score"]
                    fallback_used = True

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.error(f"[{request_id}] Pinecone query failed: {e}", exc_info=True)
            return {
                "retrieval_status": "unavailable",
                "results": [],
                "error": str(e),
                "telemetry": {
                    "request_id": request_id,
                    "predicted_intent": predicted_intent or "UNKNOWN",
                    "top_similarity_score": 0.0,
                    "number_of_results": 0,
                    "fallback_used": fallback_used,
                    "latency_ms": round(latency, 2)
                }
            }

        # Step 4: Parse results
        results = []
        for m in matches:
            meta = m.get("metadata", {})
            results.append({
                "id": m.get("id"),
                "customer_text": meta.get("customer_text", ""),
                "amazon_reply": meta.get("amazon_reply", ""),
                "intent": meta.get("intent", predicted_intent),
                "score": round(float(m.get("score", 0.0)), 4)
            })

        latency = (time.time() - start_time) * 1000
        status = "success" if results else "no_match"

        # Addon 8: Structured telemetry logging (No PII)
        logger.info(
            f"Retrieval Telemetry | req_id={request_id} | intent={predicted_intent} | "
            f"top_score={round(top_score, 4)} | results={len(results)} | "
            f"fallback={fallback_used} | latency={latency:.2f}ms"
        )

        return {
            "retrieval_status": status,
            "results": results,
            "telemetry": {
                "request_id": request_id,
                "predicted_intent": predicted_intent,
                "top_similarity_score": round(top_score, 4),
                "number_of_results": len(results),
                "fallback_used": fallback_used,
                "latency_ms": round(latency, 2)
            }
        }


# Singleton getter
_pinecone_service_instance = None
_service_lock = Lock()


def get_pinecone_service() -> PineconeRetrievalService:
    """Returns singleton PineconeRetrievalService instance."""
    global _pinecone_service_instance
    if _pinecone_service_instance is None:
        with _service_lock:
            if _pinecone_service_instance is None:
                _pinecone_service_instance = PineconeRetrievalService()
    return _pinecone_service_instance


def retrieve_similar_cases(
    query: str,
    intent: Optional[str] = None,
    top_k: int = 5,
    namespace: str = "amazon"
) -> List[Dict[str, Any]]:
    """Convenience functional wrapper returning just the list of case dicts."""
    service = get_pinecone_service()
    res = service.retrieve_similar_cases(query=query, intent=intent, top_k=top_k, namespace=namespace)
    return res.get("results", [])
