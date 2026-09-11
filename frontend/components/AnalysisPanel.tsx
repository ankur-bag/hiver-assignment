'use client';

import React from 'react';
import { FiCheckCircle, FiAlertCircle } from 'react-icons/fi';
import { ChatResponsePayload, ServiceHealth } from '../types/chat';

interface Props { analysis: ChatResponsePayload | null; health: ServiceHealth; sessionId: string }

export const AnalysisPanel: React.FC<Props> = ({ analysis }) => {
  const intent = analysis?.intent && analysis.intent !== 'Analyzing...' ? analysis.intent : null;
  const context = analysis?.retrieved_context;
  return (
    <aside className="h-full flex flex-col bg-[#171615] border-l border-white/[0.06] p-5 overflow-y-auto text-xs select-none">
      <div className="pb-4 border-b border-white/[0.06]">
        <h2 className="text-[13px] font-medium text-[#EDEDEC]">Analysis</h2>
      </div>
      <div className="flex-1 py-5 space-y-6">
        <div className="space-y-1.5">
          <div className="text-[11.5px] text-[#8C8B88]">Intent</div>
          <div className="text-[13.5px] font-medium text-[#EDEDEC]">
            {intent ? intent.toLowerCase().replace(/_/g, ' ') : '—'}
          </div>
        </div>
        <div className="space-y-1.5">
          <div className="text-[11.5px] text-[#8C8B88]">Retrieved context</div>
          <div className="text-[13px] text-[#EDEDEC]">
            {context === null || context === undefined ? '—' : typeof context === 'number' ? `${context} source${context === 1 ? '' : 's'}` : context === 'available' ? 'Available' : 'Unavailable'}
          </div>
        </div>
        <div className="space-y-1.5">
          <div className="text-[11.5px] text-[#8C8B88]">Escalation</div>
          <div className="flex items-center gap-1.5 text-[13px]">
            {analysis?.escalate === true ? <><FiAlertCircle className="w-3.5 h-3.5 text-rose-400"/><span className="text-rose-300 font-medium">Required</span></> :
             analysis?.escalate === false ? <><FiCheckCircle className="w-3.5 h-3.5 text-emerald-400"/><span className="text-[#EDEDEC]">Not required</span></> : <span className="text-[#8C8B88]">—</span>}
          </div>
          {analysis?.escalate && analysis.escalation_reason && <p className="text-[11px] text-rose-300/80 pt-1">{analysis.escalation_reason}</p>}
        </div>
      </div>
    </aside>
  );
};
