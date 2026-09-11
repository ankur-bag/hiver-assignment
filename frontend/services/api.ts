/**
 * API Client Service for Hiver AI Support.
 * Communicates with FastAPI backend or Cloudflare Worker edge proxy.
 */

import { ChatResponsePayload, ServiceHealth } from '../types/chat';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function checkBackendHealth(): Promise<ServiceHealth> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/health`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
      cache: 'no-store',
    });

    if (!res.ok) {
      return { status: 'degraded' };
    }

    return await res.json();
  } catch (err) {
    console.error('Health check failed:', err);
    return { status: 'offline' };
  }
}

export async function sendChatMessage(
  text: string,
  sessionId?: string,
  language?: string
): Promise<ChatResponsePayload> {
  const res = await fetch(`${API_BASE_URL}/api/v1/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify({
      text,
      session_id: sessionId,
      language: language || undefined,
    }),
  });

  if (!res.ok) {
    let errorDetail = 'Support service temporarily unavailable';
    try {
      const errJson = await res.json();
      if (errJson.error) {
        errorDetail = errJson.error;
      }
    } catch {
      // fallback to status text
      errorDetail = `Server error (${res.status}: ${res.statusText})`;
    }
    throw new Error(errorDetail);
  }

  return await res.json();
}
