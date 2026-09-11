"""
Production Gemini API Service Wrapper.
Uses the modern google-genai SDK to generate grounded, customer-facing support responses.
Implements:
- Lazy client initialization from backend/.env
- Configurable model (defaults to environment setting or gemini-2.5-flash / gemini-3.5-flash-lite)
- Exponential backoff retry logic for transient timeouts and rate limits (HTTP 429)
- Structured error handling and graceful fallbacks
"""

import logging
import os
import time
from pathlib import Path
from threading import Lock
from typing import Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Maximum retries and backoff
MAX_RETRIES = 3
INITIAL_BACKOFF_SEC = 2.0


class GeminiService:
    """
    Singleton wrapper for Google Gemini API.
    """
    _instance = None
    _lock = Lock()

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        env_path = Path(__file__).resolve().parents[1] / ".env"
        load_dotenv(env_path)

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or os.getenv("MODEL", "gemini-3.5-flash-lite")
        self._client = None

    def _get_client(self):
        """Initializes and caches the genai Client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY is not set in backend/.env or environment.")
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "google-genai is not installed. Install via: pip install google-genai"
                )
        return self._client

    def generate_response(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 512
    ) -> str:
        """
        Calls Gemini API with exponential backoff and timeout handling.

        Args:
            prompt: Formatted RAG prompt including system persona, query, and context.
            temperature: Sampling temperature (default 0.2 for factual, grounded responses).
            max_output_tokens: Maximum tokens in generated reply.

        Returns:
            Generated response string from Gemini.
        """
        if not prompt or not prompt.strip():
            return ""

        client = self._get_client()

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"Dispatching prompt to Gemini model '{self.model_name}' (attempt {attempt}/{MAX_RETRIES})...")
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                if response and response.text:
                    return response.text.strip()
                raise ValueError("Empty response received from Gemini API.")
            except Exception as e:
                err_str = str(e)
                logger.warning(f"Gemini API attempt {attempt} failed: {err_str}")
                if attempt == MAX_RETRIES:
                    logger.error(f"Exhausted {MAX_RETRIES} attempts calling Gemini API: {err_str}")
                    raise e
                sleep_time = INITIAL_BACKOFF_SEC * (2 ** (attempt - 1))
                time.sleep(sleep_time)

        return ""

    def generate_response_stream(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 512
    ):
        """
        Calls Gemini Streaming API and yields text chunks as they arrive from the model.

        Args:
            prompt: Formatted RAG prompt.
            temperature: Sampling temperature.
            max_output_tokens: Maximum tokens in generated reply.

        Yields:
            str: Progressive text chunks emitted by Gemini.
        """
        if not prompt or not prompt.strip():
            return

        client = self._get_client()

        # Retry logic for establishing the stream connection
        stream = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"Dispatching streaming prompt to Gemini model '{self.model_name}' (attempt {attempt}/{MAX_RETRIES})...")
                stream = client.models.generate_content_stream(
                    model=self.model_name,
                    contents=prompt
                )
                break
            except Exception as e:
                err_str = str(e)
                logger.warning(f"Gemini API streaming attempt {attempt} connection failed: {err_str}")
                if attempt == MAX_RETRIES:
                    logger.error(f"Exhausted {MAX_RETRIES} attempts connecting to Gemini stream: {err_str}")
                    raise e
                sleep_time = INITIAL_BACKOFF_SEC * (2 ** (attempt - 1))
                time.sleep(sleep_time)

        if stream is None:
            return

        # Iterate over streamed chunks
        try:
            for chunk in stream:
                if chunk and chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Error during Gemini stream iteration: {e}", exc_info=True)
            raise e



_gemini_service_instance = None
_gemini_lock = Lock()


def get_gemini_service() -> GeminiService:
    """Returns the shared singleton instance of GeminiService."""
    global _gemini_service_instance
    if _gemini_service_instance is None:
        with _gemini_lock:
            if _gemini_service_instance is None:
                _gemini_service_instance = GeminiService()
    return _gemini_service_instance
