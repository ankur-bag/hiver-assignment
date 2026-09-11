/**
 * Cloudflare Worker API Gateway for Hiver AI Customer Support.
 * 
 * Responsibilities:
 * 1. Low-latency edge proxy routing requests to the Render FastAPI backend.
 * 2. Real Cloudflare D1 conversation and message persistence.
 * 3. Owner-isolated conversation CRUD (/api/v1/conversations*).
 * 4. Unbuffered real-time SSE token streaming passthrough with edge aggregation for D1 saving.
 * 5. Edge CORS preflight handling with strict security headers.
 * 6. Edge/origin shared secret authentication.
 * 7. Optional Cloudflare Turnstile token validation.
 */

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // 1. CORS Preflight Handling (OPTIONS)
    if (request.method === "OPTIONS") {
      return handleOptions(request, env);
    }

    // 2. Health check shortcut at Edge
    if (url.pathname === "/edge-health") {
      return jsonResponse({ status: "edge_healthy", timestamp: Date.now() }, 200, request, env);
    }

    // 3. Conversation CRUD Routes (D1 Persistence)
    if (url.pathname.startsWith("/api/v1/conversations")) {
      return handleConversationRoutes(request, env, url);
    }

    // 4. Chat Endpoints (/api/v1/chat and /api/v1/chat/stream)
    const isChatEndpoint = url.pathname === "/api/v1/chat" || url.pathname === "/api/v1/chat/stream";
    if (isChatEndpoint && request.method === "POST") {
      return handleChatRequest(request, env, ctx, url);
    }

    // 5. Proxy all other routes directly to backend
    return proxyToBackend(request, env, url);
  },
};

// =========================================================================
// Conversation CRUD Handlers (Cloudflare D1)
// =========================================================================

function getOwnerId(request) {
  const header = request.headers.get("X-Hiver-Client-ID") || request.headers.get("x-hiver-client-id");
  if (!header || typeof header !== "string") return null;
  const trimmed = header.trim();
  if (/^[a-zA-Z0-9_-]{8,128}$/.test(trimmed)) {
    return trimmed;
  }
  return null;
}

async function handleConversationRoutes(request, env, url) {
  if (!env.DB) {
    return jsonResponse({ error: "D1 database binding 'DB' is not configured." }, 503, request, env);
  }

  const ownerId = getOwnerId(request);
  if (!ownerId) {
    return jsonResponse({ error: "Valid X-Hiver-Client-ID header is required (8-128 chars alphanumeric/hyphen/underscore)." }, 401, request, env);
  }

  const pathParts = url.pathname.replace(/^\/api\/v1\/conversations\/?/, "").split("/").filter(Boolean);

  // GET /api/v1/conversations -> List all conversations for owner
  if (pathParts.length === 0 && request.method === "GET") {
    try {
      const { results } = await env.DB.prepare(
        "SELECT id, title, created_at, updated_at FROM conversations WHERE owner_id = ? ORDER BY updated_at DESC LIMIT 50"
      ).bind(ownerId).all();
      return jsonResponse({ conversations: results || [] }, 200, request, env);
    } catch (err) {
      return jsonResponse({ error: "Failed to list conversations", details: err.message }, 500, request, env);
    }
  }

  // POST /api/v1/conversations -> Create new conversation
  if (pathParts.length === 0 && request.method === "POST") {
    try {
      const body = await request.json().catch(() => ({}));
      const id = body.id || crypto.randomUUID();
      const title = (body.title || "New conversation").slice(0, 100).trim();
      const now = new Date().toISOString();

      await env.DB.prepare(
        "INSERT INTO conversations (id, owner_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)"
      ).bind(id, ownerId, title, now, now).run();

      return jsonResponse({ id, title, created_at: now, updated_at: now }, 201, request, env);
    } catch (err) {
      return jsonResponse({ error: "Failed to create conversation", details: err.message }, 500, request, env);
    }
  }

  const conversationId = pathParts[0];

  // GET /api/v1/conversations/:id -> Get conversation details and messages
  if (pathParts.length === 1 && request.method === "GET") {
    try {
      const conv = await env.DB.prepare(
        "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ? AND owner_id = ?"
      ).bind(conversationId, ownerId).first();

      if (!conv) {
        return jsonResponse({ error: "Conversation not found" }, 404, request, env);
      }

      const { results: messages } = await env.DB.prepare(
        "SELECT id, role, content, intent, escalated, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at ASC"
      ).bind(conversationId).all();

      return jsonResponse({ conversation: conv, messages: messages || [] }, 200, request, env);
    } catch (err) {
      return jsonResponse({ error: "Failed to get conversation", details: err.message }, 500, request, env);
    }
  }

  // DELETE /api/v1/conversations/:id -> Delete conversation and cascading messages
  if (pathParts.length === 1 && request.method === "DELETE") {
    try {
      const conv = await env.DB.prepare(
        "SELECT id FROM conversations WHERE id = ? AND owner_id = ?"
      ).bind(conversationId, ownerId).first();

      if (!conv) {
        return jsonResponse({ error: "Conversation not found" }, 404, request, env);
      }

      // Explicit cascading deletion for robustness
      await env.DB.batch([
        env.DB.prepare("DELETE FROM messages WHERE conversation_id = ?").bind(conversationId),
        env.DB.prepare("DELETE FROM conversations WHERE id = ? AND owner_id = ?").bind(conversationId, ownerId),
      ]);

      return jsonResponse({ success: true, id: conversationId }, 200, request, env);
    } catch (err) {
      return jsonResponse({ error: "Failed to delete conversation", details: err.message }, 500, request, env);
    }
  }

  return jsonResponse({ error: "Method not allowed or endpoint not found" }, 405, request, env);
}

// =========================================================================
// Chat & Streaming Proxy with Asynchronous D1 Persistence
// =========================================================================

async function handleChatRequest(request, env, ctx, url) {
  // 1. Turnstile Check if enabled
  if (env.TURNSTILE_ENABLED === "true") {
    const turnstileToken = request.headers.get("CF-Turnstile-Token") || request.headers.get("x-turnstile-token");
    const clientIp = request.headers.get("CF-Connecting-IP") || "127.0.0.1";

    if (!turnstileToken) {
      return jsonResponse({ error: "Security verification required. Missing Turnstile token." }, 403, request, env);
    }

    if (env.TURNSTILE_SECRET_KEY) {
      const formData = new FormData();
      formData.append("secret", env.TURNSTILE_SECRET_KEY);
      formData.append("response", turnstileToken);
      formData.append("remoteip", clientIp);

      const verifyRes = await fetch("https://challenges.cloudflare.com/turnstile/v0/siteverify", {
        method: "POST",
        body: formData,
      });
      const verifyData = await verifyRes.json();
      if (!verifyData.success) {
        return jsonResponse({ error: "Turnstile security verification failed." }, 403, request, env);
      }
    }
  }

  // 2. Read request body once
  let requestBodyJson = null;
  let rawBodyText = "";
  try {
    rawBodyText = await request.text();
    requestBodyJson = JSON.parse(rawBodyText);
  } catch (err) {
    return jsonResponse({ error: "Invalid JSON request body." }, 400, request, env);
  }

  const queryText = (requestBodyJson.text || "").trim();
  const sessionId = requestBodyJson.session_id || crypto.randomUUID();
  const ownerId = getOwnerId(request) || "anonymous";
  const now = new Date().toISOString();

  // 3. Persist user message to D1 (fire-and-forget or awaited quickly)
  if (env.DB && queryText && ownerId !== "anonymous") {
    try {
      const title = queryText.slice(0, 60).trim();
      await env.DB.prepare(
        `INSERT INTO conversations (id, owner_id, title, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET updated_at = ?`
      ).bind(sessionId, ownerId, title, now, now, now).run();

      const userMsgId = crypto.randomUUID();
      await env.DB.prepare(
        "INSERT INTO messages (id, conversation_id, role, content, intent, escalated, created_at) VALUES (?, ?, 'user', ?, NULL, 0, ?)"
      ).bind(userMsgId, sessionId, queryText, now).run();
    } catch (d1Err) {
      console.warn("Could not persist user message to D1:", d1Err.message);
    }
  }

  // 4. Forward to FastAPI backend
  const BACKEND_URL = env.BACKEND_API_URL || "http://127.0.0.1:8000";
  const targetUrl = new URL(url.pathname + url.search, BACKEND_URL);

  const newHeaders = new Headers(request.headers);
  const clientIp = request.headers.get("CF-Connecting-IP") || "127.0.0.1";
  const requestId = request.headers.get("X-Request-ID") || crypto.randomUUID().slice(0, 8);

  newHeaders.set("X-Forwarded-For", clientIp);
  newHeaders.set("X-Request-ID", requestId);
  newHeaders.set("Content-Type", "application/json");
  if (env.EDGE_SHARED_SECRET) {
    newHeaders.set("X-Hiver-Edge-Auth", env.EDGE_SHARED_SECRET);
  }

  let backendResponse;
  try {
    backendResponse = await fetch(targetUrl.toString(), {
      method: "POST",
      headers: newHeaders,
      body: rawBodyText,
      redirect: "follow",
    });
  } catch (err) {
    return jsonResponse({
      error: "Edge Gateway: Unable to connect to backend support service",
      request_id: requestId,
    }, 502, request, env);
  }

  const isStreaming = url.pathname === "/api/v1/chat/stream";

  // Construct response headers with CORS
  const responseHeaders = new Headers(backendResponse.headers);
  const corsHeaders = getCorsHeaders(request, env);
  for (const [key, value] of Object.entries(corsHeaders)) {
    responseHeaders.set(key, value);
  }
  responseHeaders.set("X-Edge-Request-ID", requestId);
  responseHeaders.set("X-Content-Type-Options", "nosniff");
  responseHeaders.set("X-Frame-Options", "DENY");

  // Non-streaming response: save assistant message and return
  if (!isStreaming) {
    if (backendResponse.ok && env.DB && ownerId !== "anonymous") {
      try {
        const resJson = await backendResponse.clone().json();
        if (resJson && resJson.reply) {
          const assistantMsgId = crypto.randomUUID();
          const assistantNow = new Date().toISOString();
          await env.DB.prepare(
            "INSERT INTO messages (id, conversation_id, role, content, intent, escalated, created_at) VALUES (?, ?, 'assistant', ?, ?, ?, ?)"
          ).bind(
            assistantMsgId,
            sessionId,
            resJson.reply,
            resJson.intent || null,
            resJson.escalate ? 1 : 0,
            assistantNow
          ).run();
        }
      } catch (d1Err) {
        console.warn("Could not persist non-streaming assistant reply:", d1Err.message);
      }
    }
    return new Response(backendResponse.body, {
      status: backendResponse.status,
      statusText: backendResponse.statusText,
      headers: responseHeaders,
    });
  }

  // Streaming SSE response: stream tokens unbuffered and record assistant message at completion
  const textDecoder = new TextDecoder();
  let accumulatedAssistantText = "";
  let detectedIntent = null;
  let detectedEscalate = 0;
  let streamHadError = false;

  const { readable, writable } = new TransformStream({
    transform(chunk, controller) {
      controller.enqueue(chunk);

      // Parse chunk in background without blocking
      const chunkStr = textDecoder.decode(chunk, { stream: true });
      const lines = chunkStr.split("\n");
      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith("data:")) {
          const dataStr = trimmed.slice(5).trim();
          try {
            const parsed = JSON.parse(dataStr);
            if (parsed.text) {
              accumulatedAssistantText += parsed.text;
            }
            if (parsed.intent) {
              detectedIntent = parsed.intent;
            }
            if (parsed.escalate !== undefined) {
              detectedEscalate = parsed.escalate ? 1 : 0;
            }
            if (parsed.code || parsed.error) {
              streamHadError = true;
            }
          } catch {}
        }
      }
    },
    async flush() {
      // Stream completed cleanly: write one message to D1
      if (env.DB && ownerId !== "anonymous" && accumulatedAssistantText.trim() && !streamHadError) {
        try {
          const assistantMsgId = crypto.randomUUID();
          const assistantNow = new Date().toISOString();
          await env.DB.prepare(
            "INSERT INTO messages (id, conversation_id, role, content, intent, escalated, created_at) VALUES (?, ?, 'assistant', ?, ?, ?, ?)"
          ).bind(
            assistantMsgId,
            sessionId,
            accumulatedAssistantText.trim(),
            detectedIntent,
            detectedEscalate,
            assistantNow
          ).run();
        } catch (d1Err) {
          console.warn("Could not persist streamed assistant reply to D1:", d1Err.message);
        }
      }
    }
  });

  // Pipe backend stream through our TransformStream to browser
  ctx.waitUntil(backendResponse.body.pipeTo(writable).catch(err => {
    console.warn("Stream pipe error:", err.message);
  }));

  return new Response(readable, {
    status: backendResponse.status,
    statusText: backendResponse.statusText,
    headers: responseHeaders,
  });
}

// =========================================================================
// Proxy Helper
// =========================================================================

async function proxyToBackend(request, env, url) {
  const BACKEND_URL = env.BACKEND_API_URL || "http://127.0.0.1:8000";
  const targetUrl = new URL(url.pathname + url.search, BACKEND_URL);

  const newHeaders = new Headers(request.headers);
  const clientIp = request.headers.get("CF-Connecting-IP") || "127.0.0.1";
  const requestId = request.headers.get("X-Request-ID") || crypto.randomUUID().slice(0, 8);

  newHeaders.set("X-Forwarded-For", clientIp);
  newHeaders.set("X-Request-ID", requestId);

  try {
    const backendResponse = await fetch(targetUrl.toString(), {
      method: request.method,
      headers: newHeaders,
      body: request.method !== "GET" && request.method !== "HEAD" ? request.body : null,
      redirect: "follow",
    });

    const responseHeaders = new Headers(backendResponse.headers);
    const corsHeaders = getCorsHeaders(request, env);
    for (const [key, value] of Object.entries(corsHeaders)) {
      responseHeaders.set(key, value);
    }
    responseHeaders.set("X-Edge-Request-ID", requestId);

    return new Response(backendResponse.body, {
      status: backendResponse.status,
      statusText: backendResponse.statusText,
      headers: responseHeaders,
    });
  } catch (err) {
    return jsonResponse({
      error: "Edge Gateway: Unable to connect to backend support service",
      request_id: requestId,
    }, 502, request, env);
  }
}

// =========================================================================
// CORS & Utility Helpers
// =========================================================================

function getCorsHeaders(request, env) {
  const origin = request.headers.get("Origin") || "";
  const configured = (env.ALLOWED_ORIGINS || "http://localhost:3000,http://127.0.0.1:3000").split(",").map(v => v.trim());
  const allowedOrigin = configured.includes(origin) ? origin : configured[0];
  return {
    "Access-Control-Allow-Origin": allowedOrigin,
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Request-ID, X-Hiver-Client-ID, x-hiver-client-id, CF-Turnstile-Token, x-turnstile-token",
    "Access-Control-Allow-Credentials": "true",
    "Access-Control-Max-Age": "86400",
  };
}

function handleOptions(request, env) {
  return new Response(null, {
    status: 204,
    headers: getCorsHeaders(request, env),
  });
}

function jsonResponse(data, status, request, env) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json",
      ...getCorsHeaders(request, env),
    },
  });
}
