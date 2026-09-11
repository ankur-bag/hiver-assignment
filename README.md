# Enterprise Multilingual AI Customer Support Agent (Hiver Assignment)

A production-grade, end-to-end AI Customer Support Agent built for the **Hiver** engineering assignment. The system ingests and analyzes customer support conversations from Twitter Customer Support (TWCS) for **AmazonHelp**, trains a high-precision multilingual intent classification engine, performs hybrid vector retrieval over a Pinecone Vector Database, synthesizes brand-compliant responses via Google Gemini LLM, and serves real-time conversational assistance through a FastAPI backend and Next.js reactive frontend with live ML telemetry.

---

##  System Architecture

```
                                  +-----------------------------+
                                  |     Next.js 16 Frontend     |
                                  |  (React 19 + Tailwind CSS)  |
                                  +--------------+--------------+
                                                 |
                                         (REST / JSON / CORS)
                                                 v
                                  +-----------------------------+
                                  |  Cloudflare Worker Gateway  |
                                  |    (Edge Proxy + DDoS)      |
                                  +--------------+--------------+
                                                 |
                                                 v
                                  +-----------------------------+
                                  |       FastAPI Backend       |
                                  |      /api/v1/chat & health  |
                                  +--------------+--------------+
                                                 |
                                                 v
                                  +-----------------------------+
                                  |     Chat Service Layer      |
                                  |  Session & Latency Tracking |
                                  +--------------+--------------+
                                                 |
                                                 v
                       +--------------------------------------------------+
                       |             RAG Orchestrator Layer               |
                       +-------------------------+------------------------+
                                                 |
         +---------------------------------------+---------------------------------------+
         |                                       |                                       |
         v                                       v                                       v
+------------------------+             +------------------------+             +------------------------+
|   Intent Classifier    |             | Pinecone Vector Store  |             |  Escalation Engine     |
| MiniLM-L12 + LogReg    |             | 52,124 Amazon Vectors  |             | Multi-factor Policies  |
| 98.07% Test Accuracy   |             | 384d Cosine Similarity |             | Security/Fraud Checks  |
+------------------------+             +------------------------+             +------------------------+
         |                                       |                                       |
         +---------------------------------------+---------------------------------------+
                                                 |
                                                 v
                                  +-----------------------------+
                                  |      Gemini LLM Layer       |
                                  |   gemini-2.5-flash-lite     |
                                  | Brand-Constrained Synthesis |
                                  +--------------+--------------+
                                                 |
                                                 v
                                  +-----------------------------+
                                  | Output Safety & Validator   |
                                  |  PII / Internal Scrubbing   |
                                  +-----------------------------+
```

---

##  Key Features

1. **Multilingual Intent Intelligence (Phase 1):**
   - Trained on **52,124** clean customer support conversations from AmazonHelp.
   - `paraphrase-multilingual-MiniLM-L12-v2` dense 384-dimensional embeddings + Logistic Regression classifier.
   - **98.07% test accuracy** across 7,819 holdout samples covering 9 canonical e-commerce intents: `DELIVERY_DELAY`, `PACKAGE_NOT_RECEIVED`, `ORDER_STATUS`, `REFUND_PENDING`, `ACCOUNT_ACCESS`, `RETURNS_EXCHANGE`, `SUBSCRIPTION_INQUIRY`, `CUSTOMER_SERVICE_CONTACT`, and `ESCALATION`.
   - Automated language identification supporting English (`en`), Hindi (`hi`), Hinglish (`hi-Latn`), German (`de`), Spanish (`es`), and French (`fr`).

2. **Enterprise Hybrid Retrieval Layer (Phase 2):**
   - **52,124** vectors indexed in a Pinecone Serverless Index (`amazon-support-resolutions`) in AWS `us-east-1` under namespace `amazon`.
   - Two-stage hybrid search: Intent-filtered semantic search with automatic fallback to global cosine similarity if score is below `0.60`.
   - Sub-150ms retrieval latency with cosine metric normalization.

3. **Brand-Aligned Response Synthesis (Phase 3):**
   - Google Gemini 2.5 Flash Lite engine conditioned on retrieved ground-truth historical resolutions.
   - Strict system negative constraints: Never invents internal tracking numbers or guarantees, maintains AmazonHelp's signature empathetic, concise, and actionable tone.
   - Comprehensive safety validator: Blocks leakage of internal database terms (`Pinecone`, `vector store`, `embedding`, `LLM`, `model weights`).
   - Deterministic multi-factor escalation engine: Immediately flags account hacking, unauthorized charges, fraud, low intent confidence (<0.40), or unrecoverable generation errors.

4. **Production API & Edge Gateway (Phase 4):**
   - **Backend Service Layer Separation:** Complete separation between HTTP transport, business chat service, and RAG orchestrator.
   - **API Versioning & Streaming:** Strict `/api/v1/chat`, `/api/v1/chat/stream` (real-time SSE streaming), and `/api/v1/health` contracts.
   - **Memory-Optimized ONNX Runtime:** Lightweight `onnxruntime` + Rust `tokenizers` embedding pipeline with standalone NumPy classifier (`classifier_weights.npz`), reducing memory consumption by over 75% for Render Free (<512 MiB limit, zero PyTorch/SentenceTransformers runtime dependencies).
   - **Structured Logging:** Zero-PII logging of `request_id`, `endpoint`, `latency_ms`, `status`, and `error_type`.
   - **Centralized Error Handling:** Clean client-safe JSON responses, masking internal stack traces.
   - **Sliding-Window Rate Limiting:** 60 requests/minute per client IP.
   - **Cloudflare Worker Adapter:** Edge gateway forwarding requests with zero-latency preflight CORS handling, unbuffered SSE streaming passthrough, and optional Turnstile verification.
   - **Dockerized Container:** Lightweight `backend/Dockerfile` based on `python:3.11-slim` with zero CUDA dependencies.
   - **Interactive Next.js Frontend:** Real-time split-screen chat interface displaying live inference telemetry, intent classification badges, confidence progress bars, Pinecone case match counters, and human escalation alerts.

---

##  Repository Structure

```
hiver-assignment/
├── backend/
│   ├── api/
│   │   ├── middleware/
│   │   │   ├── cors.py                # Dynamic CORS for localhost & Vercel
│   │   │   └── security.py            # Rate limiting, security headers & structured logging
│   │   ├── routes/
│   │   │   ├── chat.py                # POST /api/v1/chat
│   │   │   └── health.py              # GET /api/v1/health
│   │   ├── errors.py                  # Production centralized error handlers
│   │   └── app.py                     # FastAPI application entry point
│   ├── cloudflare/
│   │   ├── worker.js                  # Cloudflare Worker Edge proxy adapter
│   │   └── wrangler.toml              # Cloudflare Worker deployment configuration
│   ├── ml/
│   │   ├── embeddings/
│   │   │   └── encoder.py             # Singleton SentenceTransformer 384d encoder
│   │   ├── models/
│   │   │   ├── embedding_intent_classifier.pkl
│   │   │   ├── label_map.json
│   │   │   └── model_metadata.json
│   │   ├── config.py                  # ML thresholds, paths, canonical intents
│   │   ├── inference.py               # Single & batch intent classifier service
│   │   └── preprocessing.py           # Sanitizer, regex cleaners, text validator
│   ├── services/
│   │   ├── chat_service.py            # Business service layer (session tracking)
│   │   ├── context_builder.py         # Formats Pinecone retrieved pairs for LLM
│   │   ├── escalation_service.py      # Multi-factor human escalation rules
│   │   ├── gemini_service.py          # Google GenAI client with retry backoff
│   │   ├── language_service.py        # Multilingual detector (en, hi, hi-Latn, etc.)
│   │   ├── pinecone_service.py        # Two-stage hybrid Pinecone retrieval
│   │   ├── prompt_builder.py          # Enterprise support system prompts
│   │   ├── rag_service.py             # Full RAG pipeline orchestrator
│   │   └── response_validator.py      # Output guardrails & PII scrubber
│   ├── Dockerfile                     # Production container definition
│   └── requirements.txt               # Backend Python dependencies
├── frontend/
│   ├── app/
│   │   ├── globals.css                # Tailwind CSS styles
│   │   ├── layout.tsx                 # Root HTML layout
│   │   └── page.tsx                   # Main page rendering ChatWindow
│   ├── components/
│   │   ├── ChatWindow.tsx         # Enterprise SaaS workspace (70/30 layout)
│   │   ├── ChatComposer.tsx       # Bottom floating input composer with language selector
│   │   ├── MessageBubble.tsx      # High-contrast customer and AI message typography
│   │   ├── OperationsPanel.tsx    # Right 30% AI Operations panel (Intent, Retrieval, Gemini, Safety)
│   │   ├── ConfidenceBadge.tsx    # Progress-bar confidence indicator
│   │   ├── EscalationAlert.tsx    # Human supervisor alert
│   │   └── IntentCard.tsx         # Telemetry intent diagnostics row
│   ├── hooks/
│   │   └── useChat.ts                 # React hook managing chat state & retries
│   ├── services/
│   │   └── api.ts                     # HTTP client communicating with backend API
│   ├── types/
│   │   └── chat.ts                    # TypeScript schemas for chat & telemetry
│   ├── .env.example                   # Frontend environment template
│   └── package.json                   # Next.js 16 + React 19 dependencies
├── scripts/
│   ├── training/                      # Model training & hyperparameter tuning
│   └── vector/                        # Dataset deduplication & Pinecone uploaders
├── tests/
│   ├── test_api.py                    # FastAPI route & security validation tests
│   ├── test_end_to_end.py             # Complete simulated request-to-response test
│   ├── test_inference.py              # ML classifier & preprocessing tests
│   ├── test_rag.py                    # RAG orchestrator & validator tests
│   └── test_retrieval.py              # Pinecone retrieval & fallback tests
├── reports/                           # Machine-readable evaluation reports (JSON, CSV, MD)
└── README.md                          # Comprehensive project documentation
```

---

##  Installation & Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.11 - 3.14)
- Node.js 18+ and npm
- Pinecone API Key (Serverless Index)
- Google Gemini API Key

### 2. Backend Setup

```bash
# Clone repository
cd hiver-assignment

# Create virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables in backend/.env
cat <<EOF > backend/.env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=amazon-support-resolutions
PINECONE_NAMESPACE=amazon
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash-lite
ALLOWED_ORIGINS=http://localhost:3000
EOF

# Start FastAPI backend server
uvicorn backend.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Backend API will be live at `http://localhost:8000`. Interactive Swagger docs are available at `http://localhost:8000/docs`.

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure environment variable
cp .env.example .env.local

# Run Next.js development server
npm run dev
```

Open `http://localhost:3000` in your browser to interact with the support assistant.

---

##  API Reference

### Health Check
`GET /api/v1/health`

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "intent_model": "ready",
    "pinecone": "connected",
    "gemini": "available"
  }
}
```

### Chat Completion
`POST /api/v1/chat`

**Request Body:**
```json
{
  "text": "My package with order #102-998812 was supposed to be delivered yesterday but it has not arrived yet.",
  "session_id": "sess-abc123",
  "language": "en"
}
```

**Response Body:**
```json
{
  "reply": "I understand your package hasn't arrived as scheduled. Could you please provide your tracking ID or check your delivery updates via 'Your Orders'? We'll be glad to look into this for you.",
  "intent": "DELIVERY_DELAY",
  "confidence": 0.99,
  "retrieved_cases": 3,
  "escalate": false,
  "escalation_reason": null,
  "language": "en",
  "session_id": "sess-abc123",
  "request_id": "8fa91d2",
  "telemetry": {
    "request_id": "8fa91d2",
    "intent_latency_ms": 18.5,
    "retrieval_latency_ms": 142.3,
    "gemini_latency_ms": 612.8,
    "total_latency_ms": 775.1,
    "inference_time_ms": 18.5,
    "retrieval_time_ms": 142.3,
    "generation_time_ms": 612.8,
    "total_time_ms": 775.1
  }
}
```

### Real-time SSE Streaming Chat
`POST /api/v1/chat/stream`

Returns Server-Sent Events (`text/event-stream`) delivering:
1. `event: metadata` – Intent, confidence, retrieval count, escalation status, request ID.
2. `event: token` – Progressive token chunks from Gemini synthesis.
3. `event: complete` – Full response summary and latency telemetry.

---

##  Automated Testing

The repository contains 48 unit, integration, and end-to-end tests covering all layers.

Run all tests:
```bash
python -m pytest tests/ -v
```

Test coverage includes:
- `tests/test_inference.py`: Intent classification, preprocessing, confidence calculation, batch prediction.
- `tests/test_retrieval.py`: Pinecone hybrid querying, fallback mechanism, context formatting.
- `tests/test_rag.py`: Gemini prompt construction, output validation, multilingual handling, escalation rules.
- `tests/test_api.py`: FastAPI routes, input validation, 400/422 handling, security headers.
- `tests/test_end_to_end.py`: Complete pipeline integration test verifying request-to-response behavior and escalation.

---

##  Docker & Cloudflare R2 Production Deployment

The production backend runs a memory-optimized ONNX FP32 runtime (<272 MB peak RSS, zero PyTorch/CUDA dependencies).

Because the FP32 ONNX model binary (`model.onnx`, 448.51 MB) exceeds standard GitHub repository size limits, it is stored durably in **Cloudflare R2** and downloaded during the **Docker build phase** with strict SHA-256 verification, ensuring zero runtime download latency on Render startup.

### Artifact Metadata:
- **R2 Bucket Path:** `hiver-models/production/paraphrase-multilingual-MiniLM-L12-v2/model.onnx`
- **File Size:** 470,295,205 bytes (448.51 MB)
- **SHA-256 Checksum:** `646856e76f58fd58a4a45a6e11d2efdb328cbda98190b895955a78d37d011c94`

### Docker Build with R2 Model Download:
```bash
# Build Docker image fetching model artifact from Cloudflare R2 during build:
docker build \
  --build-arg MODEL_ONNX_URL="https://pub-r2.example.com/hiver-models/production/paraphrase-multilingual-MiniLM-L12-v2/model.onnx" \
  --build-arg MODEL_ONNX_SHA256="646856e76f58fd58a4a45a6e11d2efdb328cbda98190b895955a78d37d011c94" \
  -t hiver-ai-support backend/

# Run container with environment variables
docker run -d -p 8000:8000 \
  -e PINECONE_API_KEY="your_pinecone_key" \
  -e GEMINI_API_KEY="your_gemini_key" \
  -e WEB_CONCURRENCY=1 \
  hiver-ai-support
```

---

##  Real-Time Streaming Architecture
The production streaming pipeline is built with real-time progressive token delivery:
```
Gemini Native Streaming (stream_generate_content)
       ↓
FastAPI StreamingResponse (text/event-stream)
       ↓
SSE Protocol (metadata → token → complete)
       ↓
Cloudflare Worker (unbuffered backendResponse.body passthrough)
       ↓
Next.js Frontend (ReadableStream reader & React state update)
```

---

##  Cloudflare Worker Edge Proxy

Deploy the Cloudflare Worker edge adapter to provide global low-latency proxying, CORS preflight handling, and optional Turnstile verification:

```bash
cd backend/cloudflare

# Authenticate with Cloudflare
npx wrangler login

# Deploy to Cloudflare Edge
npx wrangler deploy
```

---

##  Evaluation & Benchmark Reports

Machine-readable evaluation reports are available in `reports/`:
- `reports/accuracy_report.md`: 98.07% accuracy, precision, recall, and F1-scores for intent classification.
- `reports/confusion_matrix.png`: Visual confusion matrix across all 9 classes.
- `reports/retrieval_metrics.md`: Recall@K and latency metrics for Pinecone vector retrieval.
- `reports/rag_quality_report.md`: Benchmark validation across 10 diverse customer queries (delivery, refunds, security, Hinglish, German).
