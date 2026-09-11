'use client';

import React, { useRef, useEffect } from 'react';

interface ChatComposerProps {
  value: string;
  onChange: (val: string) => void;
  onSubmit: (e: React.FormEvent) => void;
  loading: boolean;
  placeholder?: string;
}

export const ChatComposer: React.FC<ChatComposerProps> = ({
  value,
  onChange,
  onSubmit,
  loading,
  placeholder = 'Ask about orders, refunds, delivery issues...',
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
    <form onSubmit={onSubmit} className="relative w-full">
      <div className="relative rounded-2xl bg-[#15151C]/90 border border-[rgba(255,255,255,0.08)] focus-within:border-[rgba(255,255,255,0.18)] shadow-[0_12px_40px_rgba(0,0,0,0.45)] backdrop-blur-xl transition-all duration-200">
        <textarea
          ref={textareaRef}
          rows={1}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={loading}
          className="w-full bg-transparent px-5 pt-4 pb-14 text-[15px] text-[#F5F5F5] placeholder-[#71717A] resize-none focus:outline-none leading-relaxed"
        />

        <div className="absolute left-4 bottom-3.5 flex items-center gap-2 text-[#71717A]">
          <button
            type="button"
            title="Attach file (Enterprise feature)"
            className="p-1.5 rounded-lg hover:text-[#A1A1AA] hover:bg-white/[0.04] transition-colors"
          >
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.75}
                d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13"
              />
            </svg>
          </button>
          <span className="text-[11px] font-mono tracking-tight hidden sm:inline">
            Return to send • Shift+Return for new line
          </span>
        </div>

        <div className="absolute right-4 bottom-3.5 flex items-center gap-2">
          <button
            type="submit"
            disabled={!value.trim() || loading}
            aria-label="Send message"
            className="w-8 h-8 rounded-xl bg-[#F5F5F5] text-[#0B0B0F] disabled:opacity-20 disabled:hover:bg-[#F5F5F5] hover:bg-amber-400 hover:text-black transition-all flex items-center justify-center font-medium shadow-sm"
          >
            {loading ? (
              <svg
                className="w-4 h-4 animate-spin"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="3"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8v8H4z"
                />
              </svg>
            ) : (
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M5 12h14M12 5l7 7-7 7"
                />
              </svg>
            )}
          </button>
        </div>
      </div>
    </form>
  );
};
