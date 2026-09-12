export interface TelemetryData {
  request_id?: string;
  ttft_ms?: number;
  analysis_time_ms?: number;
  generation_time_ms?: number;
  total_time_ms?: number;
  [key: string]: unknown;
}

export type IntentStatus = 'idle' | 'analyzing' | 'resolved' | 'unavailable';
export type RetrievalStatus = 'idle' | 'searching' | 'resolved' | 'unavailable';
export type EscalationStatus = 'idle' | 'checking' | 'resolved' | 'unavailable';

export interface AnalysisFields {
  intent?: string | null;
  retrieved_context?: number | string | null;
  retrieved_cases?: number | null;
  escalate?: boolean | null;
  escalation_reason?: string | null;
  language?: string;
}

export interface ProgressiveAnalysisState {
  intent: string | null;
  intentStatus: IntentStatus;
  retrievedContext: string | null;
  retrievedCases: number | null;
  retrievedStatus: RetrievalStatus;
  escalate: boolean | null;
  escalationReason: string | null;
  escalationStatus: EscalationStatus;
  language: string;
  requestId: string;
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
  stage?: 'analyzing' | 'retrieving' | 'generating' | string;
}

export interface StreamMetadataPayload extends Partial<AnalysisFields> {
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
