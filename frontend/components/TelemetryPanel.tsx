import React from 'react';
import { ChatResponsePayload, ServiceHealth } from '../types/chat';
import { IntentCard } from './IntentCard';
import { ConfidenceBadge } from './ConfidenceBadge';
import { EscalationAlert } from './EscalationAlert';

interface TelemetryPanelProps {
  analysis: ChatResponsePayload | null;
  health: ServiceHealth;
  sessionId: string;
}

export const TelemetryPanel: React.FC<TelemetryPanelProps> = ({
  analysis,
  health,
  sessionId,
}) => {
  const telemetry = analysis?.telemetry || {};

  return (
    <div className="flex flex-col h-full bg-slate-950/60 border-l border-slate-800/80 p-5 overflow-y-auto">
      <div className="flex items-center justify-between pb-4 border-b border-slate-800/80">
        <div>
          <h3 className="text-sm font-semibold text-slate-100 uppercase tracking-wider">
            AI Inference Telemetry
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Real-time pipeline diagnostics & guardrails
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              health.status === 'healthy' ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'
            }`}
          />
          <span className="text-xs text-slate-300 font-mono capitalize">
            {health.status}
          </span>
        </div>
      </div>

      <div className="space-y-4 my-4">
        {/* Session and Request Metadata */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-3.5 space-y-2 text-xs">
          <div className="flex justify-between items-center text-slate-400">
            <span>Session ID:</span>
            <span className="font-mono text-slate-200">{sessionId || 'N/A'}</span>
          </div>
          <div className="flex justify-between items-center text-slate-400">
            <span>Request ID:</span>
            <span className="font-mono text-slate-200">
              {analysis?.request_id || 'Waiting for query...'}
            </span>
          </div>
        </div>

        {/* Active Analysis Section */}
        {analysis ? (
          <>
            {/* Escalation banner if triggered */}
            {analysis.escalate && (
              <EscalationAlert reason={analysis.escalation_reason} />
            )}

            {/* Detected Intent Card */}
            <IntentCard
              intent={analysis.intent}
              confidence={analysis.confidence}
              retrievedCases={analysis.retrieved_cases}
              language={analysis.language}
            />

            {/* Confidence Metrics */}
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-slate-400 font-semibold uppercase">
                  Intent Confidence Score
                </span>
                <ConfidenceBadge confidence={analysis.confidence} />
              </div>
              <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden mt-2">
                <div
                  className={`h-full transition-all duration-500 ${
                    analysis.confidence >= 0.75
                      ? 'bg-emerald-400'
                      : analysis.confidence >= 0.50
                      ? 'bg-amber-400'
                      : 'bg-rose-400'
                  }`}
                  style={{ width: `${Math.round(analysis.confidence * 100)}%` }}
                />
              </div>
            </div>

            {/* Latency Breakdown */}
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
              <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
                Pipeline Latency Breakdown
              </h4>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between text-slate-300">
                  <span className="flex items-center gap-1.5">
                    <span className="text-sky-400 font-mono">1.</span> Intent Inference:
                  </span>
                  <span className="font-mono text-slate-200">
                    {telemetry.intent_latency_ms ?? telemetry.inference_time_ms ?? '0'} ms
                  </span>
                </div>
                <div className="flex justify-between text-slate-300">
                  <span className="flex items-center gap-1.5">
                    <span className="text-indigo-400 font-mono">2.</span> Pinecone Vector Search:
                  </span>
                  <span className="font-mono text-slate-200">
                    {telemetry.retrieval_latency_ms ?? telemetry.retrieval_time_ms ?? '0'} ms
                  </span>
                </div>
                <div className="flex justify-between text-slate-300">
                  <span className="flex items-center gap-1.5">
                    <span className="text-purple-400 font-mono">3.</span> Gemini Synthesis:
                  </span>
                  <span className="font-mono text-slate-200">
                    {telemetry.gemini_latency_ms ?? telemetry.generation_time_ms ?? '0'} ms
                  </span>
                </div>
                <div className="pt-2 border-t border-slate-800 flex justify-between font-semibold text-slate-200">
                  <span>Total Response Latency:</span>
                  <span className="font-mono text-emerald-400">
                    {telemetry.total_latency_ms ?? telemetry.total_time_ms ?? '0'} ms
                  </span>
                </div>
              </div>
            </div>
          </>
        ) : (
          <div className="flex flex-col items-center justify-center p-8 border border-dashed border-slate-800 rounded-xl text-center">
            <span className="text-3xl mb-2">⚡</span>
            <p className="text-xs text-slate-400 font-medium">
              Awaiting your first customer query.
            </p>
            <p className="text-[11px] text-slate-500 mt-1 max-w-xs">
              Type a message to see real-time intent classification, vector retrieval, and RAG telemetry.
            </p>
          </div>
        )}

        {/* Subsystems Readiness */}
        <div className="bg-slate-900/40 border border-slate-800/80 rounded-xl p-3.5 space-y-1.5 text-xs text-slate-400">
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1">
            System Subsystems
          </div>
          <div className="flex justify-between">
            <span>Intent Classifier (MiniLM-L12):</span>
            <span className="text-emerald-400 font-mono">Ready</span>
          </div>
          <div className="flex justify-between">
            <span>Pinecone Vector Store (52,124 docs):</span>
            <span className="text-emerald-400 font-mono">Connected</span>
          </div>
          <div className="flex justify-between">
            <span>Gemini LLM Synthesis:</span>
            <span className="text-emerald-400 font-mono">Available</span>
          </div>
        </div>
      </div>
    </div>
  );
};
