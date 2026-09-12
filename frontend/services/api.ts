import {
  ChatResponsePayload,
  ServiceHealth,
  StreamMetadataPayload,
  StreamCompletePayload,
} from '../types/chat';

const DEFAULT_API_URL = process.env.NEXT_PUBLIC_API_URL || 'https://hiver-assignment-yd38.onrender.com';

export async function checkBackendHealth(): Promise<ServiceHealth> {
  const urlsToTry = [DEFAULT_API_URL];
  if (DEFAULT_API_URL.includes('localhost')) {
    urlsToTry.push(DEFAULT_API_URL.replace('localhost', '127.0.0.1'));
  }

  for (const baseUrl of urlsToTry) {
    try {
      const res = await fetch(`${baseUrl}/api/v1/health`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        },
        cache: 'no-store',
      });

      if (res.ok) {
        return await res.json();
      }
    } catch {
      // Continue to fallback url
    }
  }

  return { status: 'offline' };
}

export async function sendChatMessage(
  text: string,
  sessionId?: string,
  language?: string
): Promise<ChatResponsePayload> {
  const res = await fetch(`${DEFAULT_API_URL}/api/v1/chat`, {
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
  onStatus?: (stage: string) => void;
  onMetadata?: (data: StreamMetadataPayload) => void;
  onGrounding?: (data: { retrieved_context_available?: boolean; retrieved_context_count?: number; grounding_metadata?: any[] }) => void;
  onEscalation?: (data: { escalate: boolean; reason?: string | null }) => void;
  onToken?: (text: string) => void;
  onComplete?: (data: StreamCompletePayload) => void;
  onError?: (error: Error) => void;
}

/**
 * Connects to POST /api/v1/chat/stream using Fetch API and ReadableStream.
 * Emits real-time tokens and progressive SSE updates to the UI.
 */
export async function streamChatMessage(
  text: string,
  sessionId?: string,
  language?: string,
  callbacks?: StreamCallbacks,
  signal?: AbortSignal
): Promise<void> {
  let response: Response | null = null;
  const urlsToTry = [DEFAULT_API_URL];
  if (DEFAULT_API_URL.includes('localhost')) {
    urlsToTry.push(DEFAULT_API_URL.replace('localhost', '127.0.0.1'));
  }

  let lastError: Error | null = null;

  for (const baseUrl of urlsToTry) {
    try {
      response = await fetch(`${baseUrl}/api/v1/chat/stream`, {
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

      if (response && response.ok) {
        break;
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        throw err;
      }
      lastError = err;
    }
  }

  if (!response || !response.ok) {
    let errorDetail = 'Support service temporarily unavailable';
    let errorCode: string | undefined;
    if (response) {
      try {
        const errJson = await response.json();
        if (errJson.error) {
          if (typeof errJson.error === 'object') {
            errorDetail = errJson.error.message || errJson.error.code || errorDetail;
            errorCode = errJson.error.code;
          } else if (typeof errJson.error === 'string') {
            errorDetail = errJson.error;
          }
        } else if (errJson.detail) {
          errorDetail = typeof errJson.detail === 'string' ? errJson.detail : (errJson.detail.message || JSON.stringify(errJson.detail));
          errorCode = typeof errJson.detail === 'object' ? errJson.detail.code : undefined;
        }
      } catch {
        errorDetail = `Server error (${response.status}: ${response.statusText})`;
      }
    } else if (lastError) {
      errorDetail = lastError.message;
    }
    const err = new Error(errorDetail);
    if (errorCode) {
      (err as any).code = errorCode;
    }
    callbacks?.onError?.(err);
    throw err;
  }

  if (!response.body) {
    const err = new Error('No response body returned from streaming server');
    callbacks?.onError?.(err);
    throw err;
  }

  const reader = response.body.getReader();
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
          if (eventType === 'status') {
            if (parsed.stage) {
              callbacks?.onStatus?.(parsed.stage);
            }
          } else if (eventType === 'metadata') {
            callbacks?.onMetadata?.(parsed);
          } else if (eventType === 'grounding') {
            callbacks?.onGrounding?.(parsed);
          } else if (eventType === 'escalation') {
            callbacks?.onEscalation?.(parsed);
          } else if (eventType === 'token') {
            if (typeof parsed.text === 'string') {
              callbacks?.onToken?.(parsed.text);
            }
          } else if (eventType === 'complete') {
            callbacks?.onComplete?.(parsed);
          } else if (eventType === 'error') {
            const errorMsg = parsed.message || parsed.error || 'Support service temporarily unavailable';
            const err = new Error(errorMsg);
            if (parsed.code) {
              (err as any).code = parsed.code;
            }
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
