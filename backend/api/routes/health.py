"""
Health and System Readiness Routes.
Provides GET /api/v1/health for deployment liveness and container orchestrator probes.
"""

import os
from fastapi import APIRouter
from ml.config import config as ml_config

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check():
    """
    Returns operational status of the service and its AI subsystems.
    Does not run expensive model inference on health checks to ensure fast response.
    """
    model_loaded = ml_config.WEIGHTS_PATH.exists() or ml_config.CLASSIFIER_PATH.exists()
    pinecone_configured = bool(os.getenv("PINECONE_API_KEY"))
    gemini_configured = bool(os.getenv("GEMINI_API_KEY"))

    is_healthy = model_loaded and pinecone_configured and gemini_configured

    return {
        "status": "healthy" if is_healthy else "degraded",
        "services": {
            "intent_model": "ready" if model_loaded else "missing_model_file",
            "pinecone": "connected" if pinecone_configured else "not_configured",
            "gemini": "available" if gemini_configured else "not_configured"
        }
    }
