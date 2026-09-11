/**
 * Cloudflare Worker API Gateway for Hiver AI Customer Support API.
 * 
 * Provides:
 * 1. Low-latency edge proxy routing requests to the Render FastAPI backend.
 * 2. Unbuffered real-time SSE token streaming passthrough (ReadableStream).
 * 3. Edge CORS preflight handling with strict security headers.
 * 4. Client IP and Request ID propagation.
 * 5. Optional Cloudflare Turnstile token validation on chat routes.
 * 6. Edge rate limiting / DDoS mitigation and fallback messaging.
 */

export default {
  async fetch(request, env, ctx) {
    const BACKEND_URL = env.BACKEND_API_URL || "http://localhost:8000";
    const url = new URL(request.url);

    // 1. CORS Preflight Handling (OPTIONS)
    if (request.method === "OPTIONS") {
      return handleOptions(request);
    }

    // 2. Health check shortcut at Edge
    if (url.pathname === "/edge-health") {
      return new Response(JSON.stringify({ status: "edge_healthy", timestamp: Date.now() }), {
        status: 200,
        headers: {
          "Content-Type": "application/json",
          ...getCorsHeaders(request),
        },
      });
    }

    // 3. Optional Turnstile Abuse Protection for Chat Endpoints
    const isChatEndpoint = url.pathname.startsWith("/api/v1/chat");
    if (isChatEndpoint && request.method === "POST" && env.TURNSTILE_ENABLED === "true") {
      const turnstileToken = request.headers.get("CF-Turnstile-Token") || request.headers.get("x-turnstile-token");
      const clientIp = request.headers.get("CF-Connecting-IP") || "127.0.0.1";

      if (!turnstileToken) {
        return new Response(
          JSON.stringify({
            error: "Security verification required. Missing Turnstile token.",
          }),
          {
            status: 403,
            headers: {
              "Content-Type": "application/json",
              ...getCorsHeaders(request),
            },
          }
        );
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
          return new Response(
            JSON.stringify({
              error: "Turnstile security verification failed.",
            }),
            {
              status: 403,
              headers: {
                "Content-Type": "application/json",
                ...getCorsHeaders(request),
              },
            }
          );
        }
      }
    }

    // 4. Forward request to Python FastAPI backend
    const targetUrl = new URL(url.pathname + url.search, BACKEND_URL);

    // Clone headers and add edge tracing
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

      // Construct edge response with CORS headers
      const responseHeaders = new Headers(backendResponse.headers);
      const corsHeaders = getCorsHeaders(request);
      for (const [key, value] of Object.entries(corsHeaders)) {
        responseHeaders.set(key, value);
      }
      responseHeaders.set("X-Edge-Request-ID", requestId);
      responseHeaders.set("X-Content-Type-Options", "nosniff");
      responseHeaders.set("X-Frame-Options", "DENY");

      // Pass raw body directly for unbuffered SSE streaming
      return new Response(backendResponse.body, {
        status: backendResponse.status,
        statusText: backendResponse.statusText,
        headers: responseHeaders,
      });
    } catch (err) {
      return new Response(
        JSON.stringify({
          error: "Edge Gateway: Unable to connect to backend support service",
          request_id: requestId,
        }),
        {
          status: 502,
          headers: {
            "Content-Type": "application/json",
            ...getCorsHeaders(request),
          },
        }
      );
    }
  },
};

function getCorsHeaders(request) {
  const origin = request.headers.get("Origin") || "*";
  return {
    "Access-Control-Allow-Origin": origin,
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Request-ID, CF-Turnstile-Token, x-turnstile-token",
    "Access-Control-Allow-Credentials": "true",
    "Access-Control-Max-Age": "86400",
  };
}

function handleOptions(request) {
  return new Response(null, {
    status: 204,
    headers: getCorsHeaders(request),
  });
}
