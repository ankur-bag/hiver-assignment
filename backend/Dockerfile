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

# Build-time download of model.onnx from Cloudflare R2 if not present in build context
ARG MODEL_ONNX_URL=""
ARG MODEL_ONNX_SHA256="646856e76f58fd58a4a45a6e11d2efdb328cbda98190b895955a78d37d011c94"

RUN mkdir -p ml/models/onnx && \
    if [ ! -s ml/models/onnx/model.onnx ]; then \
        if [ -z "$MODEL_ONNX_URL" ]; then \
            echo "ERROR: ml/models/onnx/model.onnx is missing and MODEL_ONNX_URL build arg is not provided." && exit 1; \
        fi && \
        echo "Downloading production model.onnx from R2: $MODEL_ONNX_URL..." && \
        curl -fSL "$MODEL_ONNX_URL" -o ml/models/onnx/model.onnx || \
        (echo "ERROR: Failed to download model.onnx from $MODEL_ONNX_URL" && exit 1); \
    fi && \
    echo "Verifying SHA-256 checksum of ml/models/onnx/model.onnx..." && \
    echo "$MODEL_ONNX_SHA256  ml/models/onnx/model.onnx" | sha256sum -c - || \
    (echo "ERROR: SHA-256 checksum mismatch for ml/models/onnx/model.onnx!" && exit 1) && \
    echo "Production model artifact verified successfully."

# Expose API port
EXPOSE 8000

# Health check probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/v1/health || exit 1

# Launch single-worker uvicorn production server
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]

