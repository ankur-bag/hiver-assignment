import React from 'react';

interface IntentCardProps {
  intent: string;
  confidence?: number;
  retrievedCases?: number;
  language?: string;
}

export const IntentCard: React.FC<IntentCardProps> = ({
  intent,
  retrievedCases,
  language = 'en',
}) => {
  return (
    <div className="py-2.5 border-b border-[rgba(255,255,255,0.06)] space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs text-[#A1A1AA]">Intent</span>
        <span className="text-xs font-mono text-[#F5F5F5] font-medium bg-white/[0.04] px-2 py-0.5 rounded border border-[rgba(255,255,255,0.06)]">
          {intent}
        </span>
      </div>

      <div className="flex items-center justify-between text-xs">
        <span className="text-[#A1A1AA]">Language</span>
        <span className="text-xs font-mono text-[#A1A1AA] uppercase">
          {language}
        </span>
      </div>

      <div className="flex items-center justify-between text-xs">
        <span className="text-[#A1A1AA]">Retrieved Cases</span>
        <span className="text-xs font-mono text-amber-400 font-medium">
          {retrievedCases ?? 0}
        </span>
      </div>
    </div>
  );
};
