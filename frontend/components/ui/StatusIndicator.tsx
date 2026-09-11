import React from 'react';

interface StatusIndicatorProps {
  status?: 'healthy' | 'degraded' | 'offline';
  label?: string;
}

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  status = 'healthy',
  label = 'AI Systems Online',
}) => {
  const isHealthy = status === 'healthy';
  const isDegraded = status === 'degraded';

  const dotColor = isHealthy
    ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.5)]'
    : isDegraded
    ? 'bg-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.5)]'
    : 'bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.5)]';

  return (
    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#15151C] border border-[rgba(255,255,255,0.08)]">
      <span className={`w-2 h-2 rounded-full ${dotColor} animate-pulse-glow`} />
      <span className="text-xs font-medium text-[#A1A1AA] tracking-wide">
        {label}
      </span>
    </div>
  );
};
