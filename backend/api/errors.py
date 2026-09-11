"""
Centralized Production Error Handling for FastAPI.
Formats validation, service, and unexpected runtime errors into clean, structured JSON.
Never exposes internal stack traces or database connection strings to clients.
"""

import logging
import uuid
from typing import Optional
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    """
    Registers custom exception handlers on the FastAPI application.
    """

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        req_id = getattr(request.state, "request_id", None) or str(uuid.uuid4())[:8]
        logger.warning(
            "HTTP %d error on %s: %s [req_id=%s]",
            exc.status_code,
            request.url.path,
            exc.detail,
            req_id,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail if isinstance(exc.detail, str) else "Request failed",
                "request_id": req_id,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        req_id = getattr(request.state, "request_id", None) or str(uuid.uuid4())[:8]
        logger.warning(
            "Validation error on %s: %s [req_id=%s]",
            request.url.path,
            exc.errors(),
            req_id,
        )
        # Construct clean client error message without technical internal schemas
        error_msgs = []
        for err in exc.errors():
            field = " -> ".join(str(loc) for loc in err.get("loc", []))
            msg = err.get("msg", "Invalid value")
            error_msgs.append(f"{field}: {msg}")
        clean_msg = "; ".join(error_msgs) if error_msgs else "Invalid request payload format"

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": clean_msg,
                "request_id": req_id,
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        req_id = getattr(request.state, "request_id", None) or str(uuid.uuid4())[:8]
        logger.error(
            "Unhandled server exception on %s [req_id=%s]: %s",
            request.url.path,
            req_id,
            exc,
            exc_info=True,
        )
        # Production safeguard: never leak raw Python stack traces or internal secrets
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Support service temporarily unavailable. Please retry shortly.",
                "request_id": req_id,
            },
        )
