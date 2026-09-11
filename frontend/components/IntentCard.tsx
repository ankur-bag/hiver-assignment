import React from 'react';

interface IntentCardProps {
  intent: string;
  confidence?: number;
  retrievedCases?: number;
  language?: string;
}

const INTENT_ICONS: Record<string, string> = {
  DELIVERY_DELAY: '🚚',
  PACKAGE_NOT_RECEIVED: '📦',
  ORDER_STATUS: '📋',
  REFUND_PENDING: '💳',
  ACCOUNT_ACCESS: '🔐',
  RETURNS_EXCHANGE: '🔄',
  SUBSCRIPTION_INQUIRY: '🔁',
  CUSTOMER_SERVICE_CONTACT: '💬',
  ESCALATION: '🚨',
};

export const IntentCard: React.FC<IntentCardProps> = ({
  intent,
  confidence,
  retrievedCases,
  language = 'en',
}) => {
  const icon = INTENT_ICONS[intent] || '⚡';
  const cleanLabel = intent.replace(/_/g, ' ');

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 shadow-sm backdrop-blur-sm">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs uppercase tracking-wider font-semibold text-slate-400">
          Detected Intent
        </span>
        <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">
          Lang: {language.toUpperCase()}
        </span>
      </div>

      <div className="flex items-center gap-3">
        <span className="text-2xl p-2 rounded-lg bg-slate-800/80 border border-slate-700/50">
          {icon}
        </span>
        <div className="min-w-0 flex-1">
          <h4 className="text-sm font-semibold text-slate-100 truncate capitalize">
            {cleanLabel.toLowerCase()}
          </h4>
          <p className="text-xs text-slate-400 font-mono mt-0.5">
            {intent}
          </p>
        </div>
      </div>

      <div className="mt-3 pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
        <span>Pinecone Retrieved Cases:</span>
        <span className="font-semibold text-sky-400 font-mono">
          {retrievedCases ?? 0} matches
        </span>
      </div>
    </div>
  );
};
