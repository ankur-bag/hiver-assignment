import {
  ChatResponsePayload,
  ServiceHealth,
  StreamMetadataPayload,
  StreamCompletePayload,
} from '../types/chat';

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
      if (errJson.error || errJson.detail) {
        errorDetail = errJson.error || errJson.detail;
      }
    } catch {
      errorDetail = `Server error (${res.status}: ${res.statusText})`;
    }
    throw new Error(errorDetail);
  }

  return await res.json();
}

export interface StreamCallbacks {
  onMetadata?: (data: StreamMetadataPayload) => void;
  onToken?: (text: string) => void;
  onComplete?: (data: StreamCompletePayload) => void;
  onError?: (error: Error) => void;
}

/**
 * Connects to POST /api/v1/chat/stream using Fetch API and ReadableStream.
 * Emits real-time tokens to progressively grow the response text in the UI.
 */
export async function streamChatMessage(
  text: string,
  sessionId?: string,
  language?: string,
  callbacks?: StreamCallbacks,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/v1/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'text/event-stream',
    },
    body: JSON.stringify({
      text,
      session_id: sessionId,
      language: language || undefined,
    }),
    signal,
  });

  if (!res.ok) {
    let errorDetail = 'Support service temporarily unavailable';
    try {
      const errJson = await res.json();
      if (errJson.error || errJson.detail) {
        errorDetail = errJson.error || errJson.detail;
      }
    } catch {
      errorDetail = `Server error (${res.status}: ${res.statusText})`;
    }
    const err = new Error(errorDetail);
    callbacks?.onError?.(err);
    throw err;
  }

  if (!res.body) {
    const err = new Error('No response body returned from streaming server');
    callbacks?.onError?.(err);
    throw err;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE chunks are separated by double newlines (\n\n)
      const parts = buffer.split('\n\n');
      buffer = parts.pop() || '';

      for (const part of parts) {
        if (!part.trim()) continue;

        let eventType = 'message';
        let dataStr = '';

        const lines = part.split('\n');
        for (const line of lines) {
          const trimmedLine = line.trim();
          if (trimmedLine.startsWith('event:')) {
            eventType = trimmedLine.replace('event:', '').trim();
          } else if (trimmedLine.startsWith('data:')) {
            dataStr += trimmedLine.replace('data:', '').trim();
          }
        }

        if (!dataStr) continue;

        try {
          const parsed = JSON.parse(dataStr);
          if (eventType === 'metadata') {
            callbacks?.onMetadata?.(parsed);
          } else if (eventType === 'token') {
            if (typeof parsed.text === 'string') {
              callbacks?.onToken?.(parsed.text);
            }
          } else if (eventType === 'complete') {
            callbacks?.onComplete?.(parsed);
          } else if (eventType === 'error') {
            const err = new Error(parsed.error || 'Support service temporarily unavailable');
            callbacks?.onError?.(err);
          }
        } catch (parseErr) {
          console.warn('Failed to parse SSE event data chunk:', dataStr, parseErr);
        }
      }
    }
  } catch (err: any) {
    if (err.name === 'AbortError') {
      console.log('Stream fetch aborted by user.');
    } else {
      callbacks?.onError?.(err);
      throw err;
    }
  } finally {
    reader.releaseLock();
  }
}

