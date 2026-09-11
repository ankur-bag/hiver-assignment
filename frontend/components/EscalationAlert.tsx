import React from 'react';

interface EscalationAlertProps {
  reason?: string | null;
}

export const EscalationAlert: React.FC<EscalationAlertProps> = ({ reason }) => {
  return (
    <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 text-amber-200">
      <div className="flex items-start gap-3">
        <span className="text-xl">⚠️</span>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <h4 className="text-sm font-semibold text-amber-300">
              Human Escalation Triggered
            </h4>
            <span className="px-2 py-0.2 rounded-full text-[10px] bg-amber-500/20 font-mono text-amber-300 uppercase">
              Priority
            </span>
          </div>
          <p className="text-xs text-amber-200/90 mt-1 leading-relaxed">
            {reason || 'This conversation has been automatically escalated to a human specialist according to enterprise safety and compliance policies.'}
          </p>
        </div>
      </div>
    </div>
  );
};
