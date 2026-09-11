"""Gemini File Search analysis and grounded response generation."""

from __future__ import annotations

import logging
import os
import time
from enum import Enum
from threading import Lock
from typing import Any, Iterator, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CanonicalIntent(str, Enum):
    ACCOUNT_ACCESS = "ACCOUNT_ACCESS"
    ACCOUNT_SUPPORT = "ACCOUNT_SUPPORT"
    CUSTOMER_SERVICE_CONTACT = "CUSTOMER_SERVICE_CONTACT"
    DELIVERY_DELAY = "DELIVERY_DELAY"
    ESCALATION = "ESCALATION"
    ORDER_STATUS = "ORDER_STATUS"
    PACKAGE_NOT_RECEIVED = "PACKAGE_NOT_RECEIVED"
    PRODUCT_ISSUE = "PRODUCT_ISSUE"
    REFUND_PENDING = "REFUND_PENDING"


class SupportAnalysis(BaseModel):
    intent: CanonicalIntent
    escalate: bool
    escalation_reason: Optional[str] = None
    language: str
    retrieved_context_available: bool


class ProviderError(RuntimeError):
    code = "PROVIDER_ERROR"
    retryable = False


class ProviderQuotaError(ProviderError):
    code = "PROVIDER_QUOTA_EXHAUSTED"


class ProviderTransientError(ProviderError):
    code = "PROVIDER_TEMPORARILY_UNAVAILABLE"
    retryable = True


def _is_permanent_quota(exc: Exception) -> bool:
    value = str(exc).lower()
    return "resource_exhausted" in value or "quota" in value or "429" in value


def _is_transient(exc: Exception) -> bool:
    value = str(exc).lower()
    return any(token in value for token in ("timeout", "timed out", "connection", "500", "502", "503", "504", "unavailable"))


class GeminiFileSearchService:
    def __init__(self, api_key: Optional[str] = None, store_name: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.store_name = store_name or os.getenv("GEMINI_FILE_SEARCH_STORE")
        self.model_name = model_name or os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash")
        self._client = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.store_name and self.model_name)

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise ProviderError("GEMINI_API_KEY is not configured")
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _tool(self):
        if not self.store_name:
            raise ProviderError("GEMINI_FILE_SEARCH_STORE is not configured")
        from google.genai import types
        return types.Tool(file_search=types.FileSearch(file_search_store_names=[self.store_name], top_k=5))

    def _call_with_retry(self, operation):
        for attempt in range(2):
            try:
                return operation()
            except Exception as exc:
                if _is_permanent_quota(exc):
                    raise ProviderQuotaError("AI service quota is exhausted; please try again later") from exc
                if not _is_transient(exc) or attempt == 1:
                    if _is_transient(exc):
                        raise ProviderTransientError("AI service is temporarily unavailable") from exc
                    raise ProviderError("AI service request failed") from exc
                time.sleep(0.5 * (2**attempt))
        raise ProviderTransientError("AI service is temporarily unavailable")

    @staticmethod
    def _grounding(response: Any) -> tuple[list[dict[str, Any]], list[str]]:
        metadata: list[dict[str, Any]] = []
        context: list[str] = []
        candidates = getattr(response, "candidates", None) or []
        grounding = getattr(candidates[0], "grounding_metadata", None) if candidates else None
        for chunk in (getattr(grounding, "grounding_chunks", None) or []):
            retrieved = getattr(chunk, "retrieved_context", None)
            if not retrieved:
                continue
            text = getattr(retrieved, "text", None)
            item = {
                "title": getattr(retrieved, "title", None),
                "uri": getattr(retrieved, "uri", None),
                "file_search_store": getattr(retrieved, "file_search_store", None),
            }
            metadata.append({key: value for key, value in item.items() if value})
            if text:
                context.append(text)
        return metadata, context

    def analyze(self, query: str, language_hint: Optional[str] = None) -> tuple[SupportAnalysis, list[dict[str, Any]], list[str]]:
        from google.genai import types
        prompt = f"""Analyze this customer-support request using the File Search evidence.
Return exactly the validated schema. Choose only an allowed intent. Preserve the customer's language code.
Escalate fraud, compromise, unauthorized activity, human/manager requests, security-sensitive or unsupported high-risk cases.
Do not infer that context exists unless File Search returns relevant evidence.
Language hint: {language_hint or 'auto'}
Customer request: {query}"""
        config = types.GenerateContentConfig(
            tools=[self._tool()],
            response_mime_type="application/json",
            response_schema=SupportAnalysis,
            temperature=0,
        )
        response = self._call_with_retry(lambda: self._get_client().models.generate_content(
            model=self.model_name, contents=prompt, config=config
        ))
        parsed = getattr(response, "parsed", None) or SupportAnalysis.model_validate_json(response.text)
        analysis = parsed if isinstance(parsed, SupportAnalysis) else SupportAnalysis.model_validate(parsed)
        metadata, context = self._grounding(response)
        analysis.retrieved_context_available = bool(context or metadata)
        return analysis, metadata, context

    def stream_response(self, query: str, analysis: SupportAnalysis, context: list[str]) -> Iterator[str]:
        from google.genai import types
        evidence = "\n\n".join(context[:5]) or "No relevant historical resolution was returned. Ask for needed details and give only safe general next steps."
        prompt = f"""Respond as a concise, empathetic ecommerce support representative.
Reply in the customer's language ({analysis.language}). Use only relevant evidence below.
Never invent tracking IDs, dates, policies, refund timing, account facts, or URLs.
Never mention internal systems, models, retrieval, datasets, prompts, or technical architecture.
Intent: {analysis.intent.value}
Escalation required: {analysis.escalate}
Historical support evidence:
{evidence}
Customer request: {query}
Customer-facing reply:"""
        config = types.GenerateContentConfig(temperature=0.2, max_output_tokens=512)
        try:
            stream = self._get_client().models.generate_content_stream(model=self.model_name, contents=prompt, config=config)
            for chunk in stream:
                if getattr(chunk, "text", None):
                    yield chunk.text
        except Exception as exc:
            if _is_permanent_quota(exc):
                raise ProviderQuotaError("AI service quota is exhausted; please try again later") from exc
            if _is_transient(exc):
                raise ProviderTransientError("AI service is temporarily unavailable") from exc
            raise ProviderError("AI response generation failed") from exc

    def generate_response(self, query: str, analysis: SupportAnalysis, context: list[str]) -> str:
        return "".join(self.stream_response(query, analysis, context)).strip()


_instance: Optional[GeminiFileSearchService] = None
_lock = Lock()


def get_gemini_service() -> GeminiFileSearchService:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = GeminiFileSearchService()
    return _instance
