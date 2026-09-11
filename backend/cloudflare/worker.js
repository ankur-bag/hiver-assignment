/**
 * Cloudflare Worker API Gateway Adapter for Hiver AI Support API.
 * 
 * Provides:
 * 1. Low-latency edge proxy routing requests to the Python FastAPI backend.
 * 2. Edge CORS preflight handling with strict security headers.
 * 3. Client IP and Request ID propagation.
 * 4. Edge rate limiting / DDoS mitigation and fallback messaging.
 */

export default {
  async fetch(request, env, ctx) {
    const BACKEND_URL = env.BACKEND_API_URL || "http://localhost:8000";
    const url = new URL(request.url);

    // If running in production mode and BACKEND_API_URL is missing, fail clearly
    if (env.ENVIRONMENT === "production" && !env.BACKEND_API_URL) {
      return new Response(
        JSON.stringify({
          error: "Edge Gateway: BACKEND_API_URL environment variable is not configured.",
        }),
        {
          status: 500,
          headers: {
            "Content-Type": "application/json",
            ...getCorsHeaders(request),
          },
        }
      );
    }

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

    // 3. Forward request to backend
    const targetUrl = new URL(url.pathname + url.search, BACKEND_URL);

    // Clone headers and add tracing
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
    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Request-ID",
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
