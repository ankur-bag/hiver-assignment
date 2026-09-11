import React from 'react';

interface MetricRowProps {
  label: string;
  value: string | number;
  unit?: string;
  badge?: string;
  badgeColor?: 'emerald' | 'amber' | 'neutral';
  highlight?: boolean;
}

export const MetricRow: React.FC<MetricRowProps> = ({
  label,
  value,
  unit,
  badge,
  badgeColor = 'neutral',
  highlight = false,
}) => {
  return (
    <div className="flex items-center justify-between py-1.5 text-xs">
      <span className="text-[#A1A1AA] font-normal">{label}</span>
      <div className="flex items-center gap-1.5">
        <span
          className={`font-mono text-xs ${
            highlight ? 'text-[#F5F5F5] font-medium' : 'text-[#A1A1AA]'
          }`}
        >
          {value}
          {unit && <span className="text-[#71717A] ml-0.5">{unit}</span>}
        </span>
        {badge && (
          <span
            className={`px-1.5 py-0.5 rounded text-[10px] font-mono uppercase tracking-wider ${
              badgeColor === 'emerald'
                ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                : badgeColor === 'amber'
                ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                : 'bg-[#15151C] text-[#A1A1AA] border border-[rgba(255,255,255,0.08)]'
            }`}
          >
            {badge}
          </span>
        )}
      </div>
    </div>
  );
};
