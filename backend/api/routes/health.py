"""
Health and System Readiness Routes.
Provides GET /api/v1/health for deployment liveness and container orchestrator probes.
"""

import os
from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check():
    """
    Returns operational status of the service and its AI subsystems.
    Does not run expensive model inference on health checks to ensure fast response.
    """
    gemini_configured = bool(os.getenv("GEMINI_API_KEY"))
    store_configured = bool(os.getenv("GEMINI_FILE_SEARCH_STORE"))
    is_healthy = gemini_configured and store_configured

    return {
        "status": "healthy" if is_healthy else "degraded",
        "services": {
            "application": "ready",
            "file_search_store": "configured" if store_configured else "not_configured",
            "gemini": "configured" if gemini_configured else "not_configured"
        }
    }
