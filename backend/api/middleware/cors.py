"""
CORS (Cross-Origin Resource Sharing) Configuration Middleware.
Allows secure cross-origin communication between the FastAPI backend,
the Next.js frontend (localhost and Vercel domains), and Cloudflare Workers.
"""

import os
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def get_allowed_origins() -> List[str]:
    """
    Parses ALLOWED_ORIGINS from environment, providing safe defaults
    for local Next.js development and production Vercel previews.
    """
    env_origins = os.getenv("ALLOWED_ORIGINS", "")
    origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ]
    if env_origins:
        for orig in env_origins.split(","):
            cleaned = orig.strip()
            if cleaned and cleaned not in origins:
                origins.append(cleaned)
    return origins


def setup_cors(app: FastAPI) -> None:
    """Configures CORSMiddleware on the FastAPI application."""
    allowed_origins = get_allowed_origins()
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_origin_regex=r"^https://.*\.vercel\.app$",  # Supports all Vercel deployment URLs
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=3600
    )
