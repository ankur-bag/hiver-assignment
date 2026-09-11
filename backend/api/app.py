"""
FastAPI Production Application Entry Point.
Initializes FastAPI, attaches security and CORS middlewares,
registers centralized error handlers, and mounts versioned API routers.
"""

import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Load environment variables from .env if present
load_dotenv()

from api.errors import register_error_handlers
from api.middleware.cors import setup_cors
from api.middleware.security import RequestContextAndSecurityMiddleware
from api.routes.chat import router as chat_router
from api.routes.health import router as health_router
from ml.config import config as ml_config
from ml.embeddings.encoder import get_embedding_encoder
from ml.inference import get_intent_classifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for warm-up and teardown tasks.
    Pre-warms the ML embedding encoder and classifier on startup to minimize first-request latency.
    """
    logger.info("Initializing Hiver AI Support Backend...")
    try:
        # Warm up singleton ML models
        get_embedding_encoder()
        get_intent_classifier()
        logger.info("ML Inference subsystems warmed up successfully.")
    except Exception as exc:
        logger.warning("ML subsystem warm-up warning (will load on first request): %s", exc)

    yield

    logger.info("Shutting down Hiver AI Support Backend...")


app = FastAPI(
    title="Hiver AI Customer Support API",
    description="Enterprise-grade RAG and Intent Classification API for Automated Customer Support",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# 1. Register security headers, rate limiting, and request ID tracking
app.add_middleware(RequestContextAndSecurityMiddleware)

# 2. Setup CORS for Next.js & Cloudflare
setup_cors(app)

# 3. Register centralized production error handlers (Addon 4)
register_error_handlers(app)

# 4. Mount versioned API routes under /api/v1 (Addon 2)
app.include_router(chat_router, prefix="/api/v1")
app.include_router(health_router, prefix="/api/v1")

# Also include root-level redirect/alias for convenience
@app.get("/")
def root_index():
    return {
        "service": "Hiver AI Customer Support API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "chat": "/api/v1/chat",
            "health": "/api/v1/health"
        }
    }
