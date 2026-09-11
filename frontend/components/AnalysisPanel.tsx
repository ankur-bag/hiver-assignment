'use client';

import React, { useState } from 'react';
import { FiChevronDown, FiChevronRight, FiCheckCircle, FiAlertCircle } from 'react-icons/fi';
import { ChatResponsePayload, ServiceHealth } from '../types/chat';

interface AnalysisPanelProps {
  analysis: ChatResponsePayload | null;
  health: ServiceHealth;
  sessionId: string;
}

export const AnalysisPanel: React.FC<AnalysisPanelProps> = ({
  analysis,
  health,
  sessionId,
}) => {
  const [isTechDetailsOpen, setIsTechDetailsOpen] = useState(false);

  const hasAnalysis = Boolean(analysis && analysis.intent);
  const telemetry = analysis?.telemetry || {};

  const confidencePct =
    analysis && typeof analysis.confidence === 'number'
      ? Math.round(analysis.confidence * 100)
      : null;

  const genLatencySec = telemetry.generation_time_ms
    ? `${(telemetry.generation_time_ms / 1000).toFixed(2)}s`
    : telemetry.gemini_latency_ms
    ? `${(telemetry.gemini_latency_ms / 1000).toFixed(2)}s`
    : null;

  const totalLatencySec = telemetry.total_time_ms
    ? `${(telemetry.total_time_ms / 1000).toFixed(2)}s`
    : telemetry.total_latency_ms
    ? `${(telemetry.total_latency_ms / 1000).toFixed(2)}s`
    : null;

  const retrievalLatencyMs =
    telemetry.retrieval_time_ms !== undefined
      ? `${telemetry.retrieval_time_ms} ms`
      : telemetry.retrieval_latency_ms !== undefined
      ? `${telemetry.retrieval_latency_ms} ms`
      : null;

  const intentLatencyMs =
    telemetry.intent_latency_ms !== undefined
      ? `${telemetry.intent_latency_ms} ms`
      : null;

  const formatLanguage = (lang?: string) => {
    if (!lang) return 'Auto';
    if (lang === 'hi-Latn') return 'Hinglish';
    if (lang === 'hi') return 'Hindi';
    if (lang === 'en') return 'English';
    if (lang === 'es') return 'Spanish';
    if (lang === 'fr') return 'French';
    if (lang === 'de') return 'German';
    return lang;
  };

  return (
    <aside className="h-full flex flex-col bg-[#171615] border-l border-white/[0.06] p-5 overflow-y-auto text-xs select-none">
      {/* Panel Header */}
      <div className="pb-4 border-b border-white/[0.06] flex items-center justify-between">
        <h2 className="text-[13px] font-medium text-[#EDEDEC]">
          Analysis
        </h2>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 py-5 space-y-6">
        {!hasAnalysis ? (
          <div className="py-8 text-center text-[#8C8B88] text-[12.5px] leading-relaxed">
            Send a message to see intent and retrieval details.
          </div>
        ) : (
          <div className="space-y-5 animate-fade-in">
            {/* 1. Intent */}
            <div className="space-y-1.5">
              <div className="text-[11.5px] text-[#8C8B88]">Intent</div>
              <div className="text-[13.5px] font-medium text-[#EDEDEC] capitalize">
                {analysis?.intent ? analysis.intent.toLowerCase().replace(/_/g, ' ') : 'Analyzing...'}
              </div>
            </div>

            {/* 2. Confidence */}
            <div className="space-y-1.5">
              <div className="text-[11.5px] text-[#8C8B88]">Confidence</div>
              <div className="flex items-center gap-3">
                <div className="flex-1 h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
                  <div
                    className="h-full bg-amber-500 transition-all duration-300 rounded-full"
                    style={{ width: `${confidencePct || 0}%` }}
                  />
                </div>
                <span className="text-[13px] font-medium text-[#EDEDEC]">
                  {confidencePct !== null ? `${confidencePct}%` : '—'}
                </span>
              </div>
            </div>

            {/* 3. Retrieved context */}
            <div className="space-y-1.5">
              <div className="text-[11.5px] text-[#8C8B88]">Retrieved context</div>
              <div className="text-[13px] text-[#EDEDEC]">
                {analysis?.retrieved_cases !== undefined
                  ? `${analysis.retrieved_cases} similar case${analysis.retrieved_cases === 1 ? '' : 's'}`
                  : '—'}
              </div>
            </div>

            {/* 4. Escalation */}
            <div className="space-y-1.5">
              <div className="text-[11.5px] text-[#8C8B88]">Escalation</div>
              <div className="flex items-center gap-1.5 text-[13px]">
                {analysis?.escalate ? (
                  <>
                    <FiAlertCircle className="w-3.5 h-3.5 text-rose-400" />
                    <span className="text-rose-300 font-medium">Escalated</span>
                  </>
                ) : (
                  <>
                    <FiCheckCircle className="w-3.5 h-3.5 text-emerald-400" />
                    <span className="text-[#EDEDEC]">Not required</span>
                  </>
                )}
              </div>
              {analysis?.escalate && analysis.escalation_reason && (
                <p className="text-[11px] text-rose-300/80 pt-1 leading-normal">
                  {analysis.escalation_reason}
                </p>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Collapsible Technical Details (Collapsed by default) */}
      <div className="border-t border-white/[0.06] pt-3">
        <button
          onClick={() => setIsTechDetailsOpen(!isTechDetailsOpen)}
          className="w-full flex items-center justify-between py-2 text-[12px] text-[#8C8B88] hover:text-[#EDEDEC] transition-colors cursor-pointer"
        >
          <span>Technical details</span>
          {isTechDetailsOpen ? (
            <FiChevronDown className="w-3.5 h-3.5" />
          ) : (
            <FiChevronRight className="w-3.5 h-3.5" />
          )}
        </button>

        {isTechDetailsOpen && (
          <div className="pt-2 pb-1 space-y-2 text-[11.5px] text-[#8C8B88] font-sans animate-fade-in">
            <div className="flex justify-between py-1 border-b border-white/[0.04]">
              <span>Model</span>
              <span className="text-[#EDEDEC]">MiniLM</span>
            </div>
            <div className="flex justify-between py-1 border-b border-white/[0.04]">
              <span>Retrieval</span>
              <span className="text-[#EDEDEC]">Pinecone</span>
            </div>
            <div className="flex justify-between py-1 border-b border-white/[0.04]">
              <span>Generation</span>
              <span className="text-[#EDEDEC]">Gemini</span>
            </div>
            {analysis?.language && (
              <div className="flex justify-between py-1 border-b border-white/[0.04]">
                <span>Detected language</span>
                <span className="text-[#EDEDEC]">{formatLanguage(analysis.language)}</span>
              </div>
            )}
            {intentLatencyMs && (
              <div className="flex justify-between py-1 border-b border-white/[0.04]">
                <span>Intent latency</span>
                <span className="text-[#EDEDEC]">{intentLatencyMs}</span>
              </div>
            )}
            {retrievalLatencyMs && (
              <div className="flex justify-between py-1 border-b border-white/[0.04]">
                <span>Retrieval latency</span>
                <span className="text-[#EDEDEC]">{retrievalLatencyMs}</span>
              </div>
            )}
            {genLatencySec && (
              <div className="flex justify-between py-1 border-b border-white/[0.04]">
                <span>Generation latency</span>
                <span className="text-[#EDEDEC]">{genLatencySec}</span>
              </div>
            )}
            {totalLatencySec && (
              <div className="flex justify-between py-1 border-b border-white/[0.04]">
                <span>Total latency</span>
                <span className="text-[#EDEDEC]">{totalLatencySec}</span>
              </div>
            )}
            {analysis?.request_id && (
              <div className="flex flex-col py-1 border-b border-white/[0.04]">
                <span>Request ID</span>
                <span className="text-[#EDEDEC] font-mono text-[10px] break-all pt-0.5">
                  {analysis.request_id}
                </span>
              </div>
            )}
            {sessionId && (
              <div className="flex flex-col py-1">
                <span>Session ID</span>
                <span className="text-[#EDEDEC] font-mono text-[10px] break-all pt-0.5">
                  {sessionId}
                </span>
              </div>
            )}
          </div>
        )}
      </div>
    </aside>
  );
};
