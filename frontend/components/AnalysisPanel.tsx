'use client';

import React from 'react';
import { FiCheckCircle, FiAlertCircle } from 'react-icons/fi';
import { ProgressiveAnalysisState } from '../types/chat';

interface Props {
  analysis: ProgressiveAnalysisState | null;
}

export const AnalysisPanel: React.FC<Props> = ({ analysis }) => {
  const isIdle = !analysis || (analysis.intentStatus === 'idle' && analysis.retrievedStatus === 'idle' && analysis.escalationStatus === 'idle');

  return (
    <aside className="h-full flex flex-col bg-[#171615] border-l border-white/[0.06] p-5 overflow-y-auto text-xs select-none">
      {/* Header */}
      <div className="pb-4 border-b border-white/[0.06]">
        <h2 className="text-[13px] font-medium text-[#EDEDEC]">Analysis</h2>
      </div>

      <div className="flex-1 py-5 space-y-6">
        {/* 1. INTENT */}
        <div className="space-y-1.5">
          <div className="text-[11.5px] text-[#8C8B88]">Intent</div>
          <div className="text-[13px] font-medium text-[#EDEDEC] flex items-center gap-2">
            {isIdle ? (
              <span className="text-[#8C8B88] font-normal">—</span>
            ) : analysis?.intentStatus === 'analyzing' ? (
              <span className="flex items-center gap-2 text-[#8C8B88] font-normal">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
                <span>Analyzing...</span>
              </span>
            ) : analysis?.intentStatus === 'unavailable' ? (
              <span className="text-[#8C8B88] font-normal">Unavailable</span>
            ) : (
              <span>{analysis?.intent || '—'}</span>
            )}
          </div>
        </div>

        {/* 2. RETRIEVED CONTEXT */}
        <div className="space-y-1.5">
          <div className="text-[11.5px] text-[#8C8B88]">Retrieved context</div>
          <div className="text-[13px] text-[#EDEDEC] flex items-center gap-2">
            {isIdle ? (
              <span className="text-[#8C8B88]">—</span>
            ) : analysis?.retrievedStatus === 'searching' ? (
              <span className="flex items-center gap-2 text-[#8C8B88]">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
                <span>Searching support history...</span>
              </span>
            ) : analysis?.retrievedStatus === 'unavailable' ? (
              <span className="text-[#8C8B88]">Unavailable</span>
            ) : (
              <span>{analysis?.retrievedContext || '—'}</span>
            )}
          </div>
        </div>

        {/* 3. ESCALATION */}
        <div className="space-y-1.5">
          <div className="text-[11.5px] text-[#8C8B88]">Escalation</div>
          <div className="flex items-center gap-1.5 text-[13px]">
            {isIdle ? (
              <span className="text-[#8C8B88]">—</span>
            ) : analysis?.escalationStatus === 'checking' ? (
              <span className="flex items-center gap-2 text-[#8C8B88]">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
                <span>Checking...</span>
              </span>
            ) : analysis?.escalationStatus === 'unavailable' ? (
              <span className="text-[#8C8B88]">Unavailable</span>
            ) : analysis?.escalate === true ? (
              <span className="flex items-center gap-1.5 text-rose-300 font-medium">
                <FiAlertCircle className="w-3.5 h-3.5 text-rose-400" />
                <span>Required</span>
              </span>
            ) : analysis?.escalate === false ? (
              <span className="flex items-center gap-1.5 text-[#EDEDEC]">
                <FiCheckCircle className="w-3.5 h-3.5 text-emerald-400" />
                <span>Not required</span>
              </span>
            ) : (
              <span className="text-[#8C8B88]">—</span>
            )}
          </div>
          {analysis?.escalationStatus === 'resolved' && analysis?.escalate && analysis?.escalationReason && (
            <p className="text-[11px] text-rose-300/80 pt-1 leading-relaxed">
              {analysis.escalationReason}
            </p>
          )}
        </div>
      </div>
    </aside>
  );
};
