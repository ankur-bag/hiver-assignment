import React from 'react';
import { ChatResponsePayload, ServiceHealth } from '../types/chat';
import { IntentCard } from './IntentCard';
import { ConfidenceBadge } from './ConfidenceBadge';
import { EscalationAlert } from './EscalationAlert';
import { MetricRow } from './ui/MetricRow';

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
    <div className="flex flex-col h-full bg-[#111116] border-l border-[rgba(255,255,255,0.08)] p-5 overflow-y-auto text-xs">
      {/* Panel Header */}
      <div className="pb-4 border-b border-[rgba(255,255,255,0.08)]">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-[#F5F5F5] font-mono">
          AI Inference Telemetry
        </h3>
        <p className="text-[11px] text-[#71717A] mt-0.5">
          Real-time pipeline diagnostics
        </p>
      </div>

      <div className="space-y-5 my-5">
        {/* System Subsystems Status */}
        <div className="space-y-1">
          <span className="text-[10px] font-mono uppercase tracking-wider text-[#71717A]">
            System Status
          </span>
          <div className="pt-1.5 space-y-1">
            <MetricRow
              label="Intent Model"
              value="READY"
              badge="MiniLM"
              badgeColor="emerald"
            />
            <MetricRow
              label="Pinecone Vector Store"
              value="52,124"
              unit="docs"
              badge="Ready"
              badgeColor="emerald"
            />
            <MetricRow
              label="Gemini Generation"
              value="AVAILABLE"
              badge="Flash Lite"
              badgeColor="emerald"
            />
          </div>
        </div>

        {/* Query Reasoning Diagnostics */}
        {analysis ? (
          <div className="space-y-4 pt-3 border-t border-[rgba(255,255,255,0.08)]">
            <span className="text-[10px] font-mono uppercase tracking-wider text-[#71717A]">
              Inference Diagnostics
            </span>

            {analysis.escalate && (
              <EscalationAlert reason={analysis.escalation_reason} />
            )}

            <IntentCard
              intent={analysis.intent}
              confidence={analysis.confidence}
              retrievedCases={analysis.retrieved_cases}
              language={analysis.language}
            />

            <div className="py-2 border-b border-[rgba(255,255,255,0.06)] flex items-center justify-between">
              <span className="text-[#A1A1AA]">Confidence</span>
              <ConfidenceBadge confidence={analysis.confidence} />
            </div>

            {/* Monospace Latency Breakdown */}
            <div className="space-y-1.5 pt-1">
              <span className="text-[10px] font-mono uppercase tracking-wider text-[#71717A]">
                Pipeline Latencies
              </span>
              <div className="pt-1 space-y-1 font-mono">
                <MetricRow
                  label="Intent"
                  value={telemetry.intent_latency_ms ?? telemetry.inference_time_ms ?? 0}
                  unit="ms"
                />
                <MetricRow
                  label="Retrieval"
                  value={telemetry.retrieval_latency_ms ?? telemetry.retrieval_time_ms ?? 0}
                  unit="ms"
                />
                <MetricRow
                  label="Generation"
                  value={telemetry.gemini_latency_ms ?? telemetry.generation_time_ms ?? 0}
                  unit="ms"
                />
                <div className="pt-1.5 border-t border-[rgba(255,255,255,0.06)]">
                  <MetricRow
                    label="Total E2E"
                    value={telemetry.total_latency_ms ?? telemetry.total_time_ms ?? 0}
                    unit="ms"
                    highlight
                  />
                </div>
              </div>
            </div>

            {/* Session identifiers */}
            <div className="pt-3 border-t border-[rgba(255,255,255,0.06)] space-y-1 text-[11px] font-mono text-[#71717A]">
              <div className="flex justify-between">
                <span>req_id</span>
                <span className="text-[#A1A1AA]">{analysis.request_id}</span>
              </div>
              <div className="flex justify-between">
                <span>session_id</span>
                <span className="text-[#A1A1AA]">{sessionId}</span>
              </div>
            </div>
          </div>
        ) : (
          <div className="pt-6 pb-4 text-center border-t border-[rgba(255,255,255,0.08)]">
            <p className="text-[11px] font-mono text-[#71717A]">
              Awaiting query execution...
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
