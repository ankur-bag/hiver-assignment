import React from 'react';

interface ConfidenceBadgeProps {
  confidence: number;
}

export const ConfidenceBadge: React.FC<ConfidenceBadgeProps> = ({ confidence }) => {
  const pct = Math.round(confidence * 100);

  let textColor = 'text-emerald-400';
  let barColor = 'bg-emerald-400';
  let badgeBg = 'bg-emerald-500/10 border-emerald-500/20';

  if (confidence < 0.5) {
    textColor = 'text-rose-400';
    barColor = 'bg-rose-400';
    badgeBg = 'bg-rose-500/10 border-rose-500/20';
  } else if (confidence < 0.75) {
    textColor = 'text-amber-400';
    barColor = 'bg-amber-400';
    badgeBg = 'bg-amber-500/10 border-amber-500/20';
  }

  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-1 bg-white/[0.06] rounded-full overflow-hidden">
        <div
          className={`h-full ${barColor} transition-all duration-300`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span
        className={`px-1.5 py-0.5 rounded text-[11px] font-mono border ${badgeBg} ${textColor}`}
      >
        {pct}%
      </span>
    </div>
  );
};
