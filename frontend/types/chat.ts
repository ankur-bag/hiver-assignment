/**
 * Core Type Definitions for Hiver AI Support Chatbot & Analysis Telemetry.
 */

export interface TelemetryData {
  request_id?: string;
  query_sanitized_length?: number;
  language_detected?: string;
  inference_time_ms?: number;
  retrieval_time_ms?: number;
  generation_time_ms?: number;
  total_time_ms?: number;
  [key: string]: any;
}

export interface ChatResponsePayload {
  reply: string;
  intent: string;
  confidence: number;
  retrieved_cases: number;
  escalate: boolean;
  escalation_reason?: string | null;
  language: string;
  session_id: string;
  request_id: string;
  telemetry: TelemetryData;
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'agent' | 'system';
  text: string;
  timestamp: string;
  intent?: string;
  confidence?: number;
  retrieved_cases?: number;
  escalate?: boolean;
  escalation_reason?: string | null;
  language?: string;
  request_id?: string;
  telemetry?: TelemetryData;
  error?: boolean;
}

export interface ServiceHealth {
  status: 'healthy' | 'degraded' | 'offline';
  services?: {
    intent_model: string;
    pinecone: string;
    gemini: string;
  };
}
