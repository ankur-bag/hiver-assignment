# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    WEB_CONCURRENCY=1

WORKDIR /app

# Install minimal system runtime dependencies (curl & ca-certificates for secure build-time download)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight Python production dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy backend source code and tracked assets
COPY . .

# Build arguments for Cloudflare R2 model artifact download
ARG MODEL_ONNX_URL=""
ARG MODEL_ONNX_SHA256="646856e76f58fd58a4a45a6e11d2efdb328cbda98190b895955a78d37d011c94"

# Enforce that local ONNX binary is absent from build context, create directory, download from R2, and verify SHA256
RUN test ! -f ml/models/onnx/model.onnx || \
    (echo "ERROR: model.onnx unexpectedly entered Docker build context (.dockerignore failed)" && exit 1) && \
    test -n "$MODEL_ONNX_URL" || \
    (echo "ERROR: MODEL_ONNX_URL build argument is required to fetch the production ONNX model from Cloudflare R2" && exit 1) && \
    mkdir -p ml/models/onnx && \
    echo "Downloading production model.onnx from Cloudflare R2: $MODEL_ONNX_URL..." && \
    curl --fail --location --retry 3 "$MODEL_ONNX_URL" --output ml/models/onnx/model.onnx && \
    echo "Verifying SHA-256 checksum of ml/models/onnx/model.onnx..." && \
    echo "$MODEL_ONNX_SHA256  ml/models/onnx/model.onnx" | sha256sum -c - && \
    echo "Production ONNX model successfully downloaded and verified."

# Expose API port
EXPOSE 8000

# Health check probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/v1/health || exit 1

# Launch single-worker uvicorn production server
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]

