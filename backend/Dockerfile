# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install system runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-prod.txt .

RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy backend source code
COPY . .

# Expose API port
EXPOSE 8000

# Health check probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Launch uvicorn production server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
