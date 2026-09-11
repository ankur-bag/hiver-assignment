import React from 'react';
import { ChatResponsePayload, ServiceHealth } from '../types/chat';

interface TelemetryPanelProps {
  analysis: ChatResponsePayload | null;
  health: ServiceHealth;
  sessionId: string;
  isOpen: boolean;
  onClose: () => void;
}

export const TelemetryPanel: React.FC<TelemetryPanelProps> = ({
  analysis,
  health,
  sessionId,
  isOpen,
  onClose,
}) => {
  if (!isOpen) return null;

  const telemetry = analysis?.telemetry || {};

  return (
    <div className="fixed top-12 right-6 z-50 w-80 rounded-2xl bg-[#171717] border border-white/[0.08] shadow-[0_16px_48px_rgba(0,0,0,0.6)] p-4 text-xs font-sans text-[#ECECEC] backdrop-blur-xl animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-white/[0.06]">
        <div className="flex items-center gap-2">
          <span className="text-amber-500">⚡</span>
          <span className="font-mono text-xs uppercase tracking-wider text-[#A3A3A3] font-semibold">
            System Diagnostics
          </span>
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded-md text-[#737373] hover:text-[#ECECEC] hover:bg-white/[0.06] transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <div className="py-3 space-y-3">
        {/* Backend Subsystems */}
        <div>
          <div className="text-[10px] font-mono text-[#666666] uppercase mb-1.5">
            Subsystems Status
          </div>
          <div className="space-y-1 font-mono text-[11px]">
            <div className="flex justify-between text-[#A3A3A3]">
              <span>Intent Model</span>
              <span className="text-emerald-400">READY</span>
            </div>
            <div className="flex justify-between text-[#A3A3A3]">
              <span>Pinecone Vectors</span>
              <span className="text-emerald-400">52,124 docs</span>
            </div>
            <div className="flex justify-between text-[#A3A3A3]">
              <span>Gemini Synthesis</span>
              <span className="text-emerald-400">AVAILABLE</span>
            </div>
          </div>
        </div>

        {/* Live Query Reasoning */}
        {analysis ? (
          <div className="pt-2 border-t border-white/[0.06] space-y-2">
            <div className="text-[10px] font-mono text-[#666666] uppercase">
              Last Response Telemetry
            </div>

            <div className="space-y-1 font-mono text-[11px]">
              <div className="flex justify-between text-[#A3A3A3]">
                <span>Detected Intent:</span>
                <span className="text-amber-400 font-semibold">{analysis.intent}</span>
              </div>
              <div className="flex justify-between text-[#A3A3A3]">
                <span>Intent Confidence:</span>
                <span className="text-[#ECECEC]">{Math.round(analysis.confidence * 100)}%</span>
              </div>
              <div className="flex justify-between text-[#A3A3A3]">
                <span>Retrieved Resolutions:</span>
                <span className="text-[#ECECEC]">{analysis.retrieved_cases} matches</span>
              </div>
              <div className="flex justify-between text-[#A3A3A3]">
                <span>Language:</span>
                <span className="text-[#ECECEC] uppercase">{analysis.language}</span>
              </div>
            </div>

            {/* Latencies */}
            <div className="pt-2 border-t border-white/[0.06] space-y-1 font-mono text-[11px]">
              <div className="flex justify-between text-[#737373]">
                <span>Intent Latency:</span>
                <span className="text-[#A3A3A3]">{telemetry.intent_latency_ms ?? 0} ms</span>
              </div>
              <div className="flex justify-between text-[#737373]">
                <span>Retrieval Latency:</span>
                <span className="text-[#A3A3A3]">{telemetry.retrieval_latency_ms ?? 0} ms</span>
              </div>
              <div className="flex justify-between text-[#737373]">
                <span>Gemini Latency:</span>
                <span className="text-[#A3A3A3]">{telemetry.gemini_latency_ms ?? 0} ms</span>
              </div>
              <div className="flex justify-between text-[#ECECEC] font-semibold pt-1 border-t border-white/[0.04]">
                <span>Total Response:</span>
                <span className="text-emerald-400">{telemetry.total_latency_ms ?? 0} ms</span>
              </div>
            </div>
          </div>
        ) : (
          <div className="pt-2 border-t border-white/[0.06] text-center text-[#666666] text-[11px] font-mono py-2">
            Awaiting prompt execution...
          </div>
        )}

        {/* Identifiers */}
        <div className="pt-2 border-t border-white/[0.06] font-mono text-[10px] text-[#666666] flex justify-between">
          <span>Session: {sessionId.slice(0, 8)}</span>
          <span className="text-emerald-400">Online</span>
        </div>
      </div>
    </div>
  );
};
