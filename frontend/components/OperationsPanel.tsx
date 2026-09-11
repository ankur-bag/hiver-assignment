import React from 'react';
import { ChatResponsePayload, ServiceHealth } from '../types/chat';

interface OperationsPanelProps {
  analysis: ChatResponsePayload | null;
  health: ServiceHealth;
  sessionId: string;
}

export const OperationsPanel: React.FC<OperationsPanelProps> = ({
  analysis,
  health,
  sessionId,
}) => {
  const telemetry = analysis?.telemetry || {};

  const genLatencySec = telemetry.generation_time_ms
    ? `${(telemetry.generation_time_ms / 1000).toFixed(1)}s`
    : telemetry.gemini_latency_ms
    ? `${(telemetry.gemini_latency_ms / 1000).toFixed(1)}s`
    : null;

  const totalLatencySec = telemetry.total_time_ms
    ? `${(telemetry.total_time_ms / 1000).toFixed(1)}s`
    : telemetry.total_latency_ms
    ? `${(telemetry.total_latency_ms / 1000).toFixed(1)}s`
    : null;

  return (
    <aside className="h-full flex flex-col bg-[#111116] border-l border-[rgba(255,255,255,0.07)] p-5 overflow-y-auto text-xs font-sans">
      {/* Panel Top Header */}
      <div className="pb-4 border-b border-[rgba(255,255,255,0.07)] flex items-center justify-between">
        <div>
          <h2 className="text-[11px] font-mono uppercase tracking-wider text-[#F3F4F6] font-semibold">
            AI Analysis & Telemetry
          </h2>
          <p className="text-[11px] text-[#6B7280] mt-0.5">
            Real-time inference & safety telemetry
          </p>
        </div>
        <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 font-mono text-[10px]">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-beacon" />
          <span>Active</span>
        </div>
      </div>

      <div className="space-y-6 my-5 font-mono">
        {/* SECTION 1: INTENT ANALYSIS */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-[10px] text-[#6B7280] tracking-wider uppercase">
            <span>Intent Analysis</span>
            <span className="text-amber-500/80">MiniLM-L12</span>
          </div>

          <div className="bg-[#15151C] border border-[rgba(255,255,255,0.06)] rounded-xl p-3 space-y-2 text-[11px]">
            <div className="flex justify-between items-center">
              <span className="text-[#9CA3AF]">Intent:</span>
              <span className="text-[#F3F4F6] font-semibold px-1.5 py-0.5 rounded bg-white/[0.04]">
                {analysis?.intent || 'AWAITING_INPUT'}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-[#9CA3AF]">Confidence:</span>
              <div className="flex items-center gap-2">
                <div className="w-12 h-1 bg-white/[0.08] rounded-full overflow-hidden">
                  <div
                    className="h-full bg-emerald-400 transition-all duration-300"
                    style={{
                      width: `${analysis ? Math.round(analysis.confidence * 100) : 0}%`,
                    }}
                  />
                </div>
                <span className="text-emerald-400 font-semibold">
                  {analysis ? `${Math.round(analysis.confidence * 100)}%` : '—'}
                </span>
              </div>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-[#9CA3AF]">Language:</span>
              <span className="text-[#D1D5DB] uppercase">
                {analysis?.language === 'hi-Latn'
                  ? 'Hinglish'
                  : analysis?.language === 'hi'
                  ? 'Hindi'
                  : analysis?.language === 'de'
                  ? 'German'
                  : analysis?.language === 'es'
                  ? 'Spanish'
                  : analysis?.language === 'fr'
                  ? 'French'
                  : 'English'}
              </span>
            </div>
          </div>
        </div>

        {/* SECTION 2: RETRIEVAL ANALYSIS */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-[10px] text-[#6B7280] tracking-wider uppercase">
            <span>Retrieval Analysis</span>
            <span className="text-sky-400/80">Vector DB</span>
          </div>

          <div className="bg-[#15151C] border border-[rgba(255,255,255,0.06)] rounded-xl p-3 space-y-2 text-[11px]">
            <div className="flex justify-between items-center">
              <span className="text-[#9CA3AF]">Pinecone:</span>
              <span className="text-emerald-400 font-semibold flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                Connected
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-[#9CA3AF]">Retrieved Cases:</span>
              <span className="text-amber-400 font-semibold">
                {analysis?.retrieved_cases !== undefined ? `${analysis.retrieved_cases} cases` : '—'}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-[#9CA3AF]">Retrieval Latency:</span>
              <span className="text-[#D1D5DB]">
                {telemetry.retrieval_time_ms !== undefined
                  ? `${telemetry.retrieval_time_ms} ms`
                  : telemetry.retrieval_latency_ms !== undefined
                  ? `${telemetry.retrieval_latency_ms} ms`
                  : '—'}
              </span>
            </div>
          </div>
        </div>

        {/* SECTION 3: GENERATION */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-[10px] text-[#6B7280] tracking-wider uppercase">
            <span>Generation</span>
            <span className="text-purple-400/80">Gemini</span>
          </div>

          <div className="bg-[#15151C] border border-[rgba(255,255,255,0.06)] rounded-xl p-3 space-y-2 text-[11px]">
            <div className="flex justify-between items-center">
              <span className="text-[#9CA3AF]">Gemini:</span>
              <span className="text-emerald-400 font-semibold flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                Available
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-[#9CA3AF]">Generation latency:</span>
              <span className="text-[#F3F4F6]">
                {genLatencySec || (analysis ? 'Streaming...' : '—')}
              </span>
            </div>

            {totalLatencySec && (
              <div className="flex justify-between items-center pt-1.5 border-t border-white/[0.04]">
                <span className="text-[#9CA3AF]">Total response time:</span>
                <span className="text-emerald-400 font-semibold">
                  {totalLatencySec}
                </span>
              </div>
            )}
          </div>
        </div>

        {/* SECTION 4: SAFETY */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-[10px] text-[#6B7280] tracking-wider uppercase">
            <span>Safety & Escalation</span>
            <span className="text-amber-400/80">Guardrails</span>
          </div>

          <div className="bg-[#15151C] border border-[rgba(255,255,255,0.06)] rounded-xl p-3 space-y-2 text-[11px]">
            <div className="flex justify-between items-center">
              <span className="text-[#9CA3AF]">Escalation:</span>
              <span
                className={`font-semibold ${
                  analysis?.escalate ? 'text-amber-400' : 'text-[#9CA3AF]'
                }`}
              >
                {analysis?.escalate ? 'True' : 'False'}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-[#9CA3AF]">Fallback:</span>
              <span className="text-[#9CA3AF] font-semibold">
                {analysis?.escalate || (analysis && analysis.retrieved_cases === 0)
                  ? 'True'
                  : 'False'}
              </span>
            </div>

            {analysis?.escalate && analysis.escalation_reason && (
              <div className="pt-2 border-t border-amber-500/20 text-[10px] text-amber-300 font-sans leading-tight">
                Reason: {analysis.escalation_reason}
              </div>
            )}
          </div>
        </div>

        {/* Console Session ID Footer */}
        <div className="pt-2 border-t border-[rgba(255,255,255,0.06)] text-[10px] text-[#4B5563] flex justify-between">
          <span>SESSION: {sessionId.slice(0, 8)}</span>
          <span>CLUSTER: US-EAST-1</span>
        </div>
      </div>
    </aside>
  );
};
