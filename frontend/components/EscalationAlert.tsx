import React from 'react';

interface EscalationAlertProps {
  reason?: string | null;
}

export const EscalationAlert: React.FC<EscalationAlertProps> = ({ reason }) => {
  return (
    <div className="rounded-xl bg-amber-500/[0.07] border border-amber-500/20 p-3 text-xs animate-fade-in">
      <div className="flex items-center gap-2 mb-1.5">
        <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
        <span className="font-medium text-amber-300 tracking-wide uppercase text-[10px] font-mono">
          Escalation Triggered
        </span>
      </div>
      <p className="text-[#A1A1AA] leading-relaxed text-[12px]">
        {reason || 'Transferred to a human customer support supervisor.'}
      </p>
    </div>
  );
};
