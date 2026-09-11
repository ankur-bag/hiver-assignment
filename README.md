# Hiver AI Customer Support

Production customer-support assistant using Gemini File Search for managed ingestion, embeddings, indexing, semantic retrieval, intent extraction, escalation analysis, and grounded response generation.

## Architecture

Browser → Vercel Next.js → Cloudflare Worker → Render FastAPI → Gemini + Gemini File Search → SSE response

The Cloudflare Worker owns CORS, request IDs, security headers, Turnstile verification, origin authentication, and unbuffered SSE passthrough. Render contains only the thin FastAPI application. Production has no Pinecone, local embedding model, ONNX runtime, classifier weights, dataset, or Cloudflare R2 dependency.

## Production configuration

Render:

```text
APP_ENV=production
GEMINI_API_KEY=...
GEMINI_FILE_SEARCH_STORE=fileSearchStores/...
GEMINI_GENERATION_MODEL=gemini-2.5-flash
ALLOWED_ORIGINS=https://your-frontend.example
EDGE_SHARED_SECRET=...
WEB_CONCURRENCY=1
```

Cloudflare Worker variables and secrets:

```text
BACKEND_API_URL=https://your-render-service.example
EDGE_SHARED_SECRET=...
TURNSTILE_ENABLED=true
TURNSTILE_SECRET_KEY=...
ALLOWED_ORIGINS=https://your-frontend.example
```

The browser receives only `NEXT_PUBLIC_API_URL`, pointing to the Worker. It never receives the Render URL or shared secret. For local development use `APP_ENV=development` and `TURNSTILE_ENABLED=false`.

## File Search setup

The source dataset remains outside Git and Render. Prepare deterministic UTF-8 text shards:

```bash
python scripts/dataset/prepare_file_search_dataset.py
```

With `GEMINI_API_KEY` set, create and populate the managed store, then run multilingual smoke tests:

```bash
python scripts/dataset/manage_file_search.py --create --upload --verify
```

Persist the printed `fileSearchStores/...` resource as `GEMINI_FILE_SEARCH_STORE`. Upload operations are polled to completion, retried with bounded backoff, and the store is rejected as ready if documents remain pending or failed.

## Local verification

```bash
python -m pytest tests/ -v
cd frontend && npm run build
docker build -t hiver-backend backend
```

`GET /api/v1/health` reports application and configuration readiness without calling Gemini or spending quota. Chat metadata contains canonical intent, real grounding availability/count, and escalation state; no synthetic confidence percentage is exposed.
