'use client';

import React, { useRef, useEffect } from 'react';
import { FiSend, FiSquare } from 'react-icons/fi';

interface ChatComposerProps {
  value: string;
  onChange: (val: string) => void;
  onSubmit: (e: React.FormEvent) => void;
  loading: boolean;
  isStreaming?: boolean;
  onStop?: () => void;
}

export const ChatComposer: React.FC<ChatComposerProps> = ({
  value,
  onChange,
  onSubmit,
  loading,
  isStreaming,
  onStop,
}) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`;
    }
  }, [value]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (value.trim() && !loading) {
        onSubmit(e);
      }
    }
  };

  return (
    <form onSubmit={onSubmit} className="w-full">
      <div className="relative flex items-end rounded-2xl bg-[#1E1D1C] border border-white/[0.08] hover:border-white/[0.12] focus-within:border-white/[0.2] transition-colors shadow-lg p-2">
        {/* Textarea Input */}
        <textarea
          ref={textareaRef}
          rows={1}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about orders, deliveries, refunds, returns, or account issues..."
          disabled={loading && !isStreaming}
          className="flex-1 bg-transparent px-3 py-2 text-[14px] text-[#EDEDEC] placeholder-[#6E6D6A] resize-none focus:outline-none leading-relaxed max-h-[160px]"
        />

        {/* Action Button: Send or Stop */}
        <div className="shrink-0 pl-2 pb-1">
          {isStreaming ? (
            <button
              type="button"
              onClick={onStop}
              aria-label="Stop response generation"
              className="h-8 px-3 rounded-xl text-[12px] font-medium flex items-center gap-1.5 bg-rose-500/15 hover:bg-rose-500/25 text-rose-300 border border-rose-500/20 transition-colors cursor-pointer"
            >
              <FiSquare className="w-3 h-3 fill-current" />
              <span>Stop</span>
            </button>
          ) : (
            <button
              type="submit"
              disabled={!value.trim() || loading}
              aria-label="Send message"
              className={`h-8 w-8 rounded-xl flex items-center justify-center transition-colors ${
                value.trim() && !loading
                  ? 'bg-amber-500 hover:bg-amber-400 text-[#171615] cursor-pointer'
                  : 'bg-white/[0.04] text-[#63625F] cursor-not-allowed'
              }`}
            >
              <FiSend className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>
    </form>
  );
};
