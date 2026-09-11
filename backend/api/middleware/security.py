"""
Security, Rate Limiting, and Request ID Logging Middleware.
Provides:
1. Request ID generation and structured request logging (request_id, endpoint, latency, status, error_type).
2. In-memory sliding-window IP rate limiter (default 60 req/min per IP).
3. Security response headers (nosniff, DENY, XSS protection, HSTS).
Strictly obeys data privacy: Never logs customer private queries, API keys, or raw conversations.
"""

import logging
import hmac
import os
import time
import uuid
from collections import defaultdict
from threading import Lock
from typing import Dict, List
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("api.access")

DEFAULT_RATE_LIMIT_REQUESTS = 60
DEFAULT_WINDOW_SECONDS = 60


class RateLimiter:
    """Sliding-window IP rate limiter."""
    def __init__(self, max_requests: int = DEFAULT_RATE_LIMIT_REQUESTS, window_sec: int = DEFAULT_WINDOW_SECONDS):
        self.max_requests = max_requests
        self.window_sec = window_sec
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.lock = Lock()

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        window_start = now - self.window_sec
        with self.lock:
            timestamps = self.requests[client_ip]
            valid_timestamps = [t for t in timestamps if t > window_start]
            self.requests[client_ip] = valid_timestamps

            if len(valid_timestamps) >= self.max_requests:
                return False

            self.requests[client_ip].append(now)
            return True


_global_rate_limiter = RateLimiter()


class RequestContextAndSecurityMiddleware(BaseHTTPMiddleware):
    """
    Middleware handling:
    - Injects unique request_id per incoming request.
    - Applies sliding-window rate limiting on chat endpoints.
    - Emits structured operational logs (request_id, endpoint, latency, status, error_type).
    - Injects hardened security HTTP response headers.
    """
    async def dispatch(self, request: Request, call_next):
        # 1. Generate or forward request ID
        req_id = request.headers.get("x-request-id") or str(uuid.uuid4())[:8]
        request.state.request_id = req_id

        # Render chat routes are private behind the Cloudflare Worker in production.
        is_chat = request.method == "POST" and request.url.path in ("/api/v1/chat", "/api/v1/chat/stream")
        local_mode = os.getenv("APP_ENV", "development").lower() in ("development", "local", "test")
        expected_secret = os.getenv("EDGE_SHARED_SECRET", "")
        if is_chat and not local_mode:
            supplied_secret = request.headers.get("x-hiver-edge-auth", "")
            if not expected_secret or not hmac.compare_digest(supplied_secret, expected_secret):
                return JSONResponse(status_code=401, content={"error": "Unauthorized edge request", "request_id": req_id})

        # 2. Extract client IP
        client_ip = request.client.host if request.client else "127.0.0.1"
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()

        # 3. Apply rate limiting to /api/v1/chat and /api/v1/chat/stream
        if (request.url.path.endswith("/chat") or request.url.path.endswith("/chat/stream")) and request.method == "POST":
            if not _global_rate_limiter.is_allowed(client_ip):
                logger.warning(
                    '{"request_id": "%s", "endpoint": "%s", "latency_ms": 0, "status": 429, "error_type": "RateLimitExceeded"}',
                    req_id,
                    request.url.path,
                )
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate limit exceeded. Please wait a moment before sending more messages.",
                        "request_id": req_id,
                    },
                    headers={"X-Request-ID": req_id}
                )

        # 4. Process request and measure latency
        start_time = time.time()
        error_type = None
        status_code = 500

        try:
            response: Response = await call_next(request)
            status_code = response.status_code

            # 5. Inject response headers
            response.headers["X-Request-ID"] = req_id
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            return response
        except Exception as exc:
            error_type = exc.__class__.__name__
            raise exc
        finally:
            latency_ms = round((time.time() - start_time) * 1000, 2)
            if status_code >= 400 and not error_type:
                error_type = f"HTTP_{status_code}"

            # Structured logging (NO customer queries, NO keys, NO full conversations)
            log_data = (
                f'{{"request_id": "{req_id}", "endpoint": "{request.url.path}", '
                f'"latency_ms": {latency_ms}, "status": {status_code}, '
                f'"error_type": "{error_type or "none"}"}}'
            )
            if status_code >= 500:
                logger.error(log_data)
            elif status_code >= 400:
                logger.warning(log_data)
            else:
                logger.info(log_data)
