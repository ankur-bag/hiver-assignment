export interface TelemetryData {
  request_id?: string;
  analysis_time_ms?: number;
  generation_time_ms?: number;
  total_time_ms?: number;
  [key: string]: unknown;
}

export interface AnalysisFields {
  intent: string;
  retrieved_context?: number | 'available' | 'unavailable' | null;
  retrieved_cases?: number | null;
  escalate?: boolean | null;
  escalation_reason?: string | null;
  language: string;
}

export interface ChatResponsePayload extends AnalysisFields {
  reply: string;
  session_id: string;
  request_id: string;
  telemetry: TelemetryData;
}

export interface ChatMessage extends Partial<AnalysisFields> {
  id: string;
  sender: 'user' | 'agent' | 'system';
  text: string;
  timestamp: string;
  request_id?: string;
  telemetry?: TelemetryData;
  error?: boolean;
  isStreaming?: boolean;
}

export interface StreamMetadataPayload extends AnalysisFields {
  request_id: string;
  session_id?: string;
}

export interface StreamCompletePayload extends AnalysisFields {
  fallback: boolean;
  session_id?: string;
  telemetry: TelemetryData;
}

export interface ServiceHealth {
  status: 'healthy' | 'degraded' | 'offline';
  services?: { application: string; file_search_store: string; gemini: string };
}
