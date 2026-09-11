'use client';

import React, { useRef, useEffect } from 'react';

interface ClaudeComposerProps {
  value: string;
  onChange: (val: string) => void;
  onSubmit: (e: React.FormEvent) => void;
  loading: boolean;
  selectedLanguage?: string;
  onLanguageChange?: (lang: string) => void;
}

export const ClaudeComposer: React.FC<ClaudeComposerProps> = ({
  value,
  onChange,
  onSubmit,
  loading,
  selectedLanguage = '',
  onLanguageChange,
}) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
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
      <div className="relative rounded-2xl bg-[#262626] border border-[rgba(255,255,255,0.08)] hover:border-[rgba(255,255,255,0.12)] focus-within:border-[rgba(255,255,255,0.18)] shadow-[0_4px_24px_rgba(0,0,0,0.35)] transition-all">
        {/* Text Input */}
        <textarea
          ref={textareaRef}
          rows={1}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="How can I help you today?"
          disabled={loading}
          className="w-full bg-transparent px-4 pt-3.5 pb-12 text-[15px] text-[#ECECEC] placeholder-[#737373] resize-none focus:outline-none leading-relaxed min-h-[52px]"
        />

        {/* Bottom Bar inside Composer */}
        <div className="absolute left-3 bottom-2.5 right-3 flex items-center justify-between pointer-events-auto">
          {/* Left: Mode / Action pills */}
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              title="Add attachment"
              className="p-1.5 rounded-lg text-[#8C8C8C] hover:text-[#ECECEC] hover:bg-white/[0.06] transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
            </button>

            {/* Chat Pill */}
            <div className="flex items-center gap-1 bg-[#323232] rounded-lg px-2.5 py-1 text-xs text-[#ECECEC] font-normal border border-white/[0.04]">
              <span>Chat</span>
            </div>

            {/* Language Selection Pill */}
            {onLanguageChange && (
              <select
                value={selectedLanguage}
                onChange={(e) => onLanguageChange(e.target.value)}
                aria-label="Language selector"
                className="bg-[#2E2E2E] border border-white/[0.06] text-[11px] rounded-lg px-2 py-1 text-[#A3A3A3] hover:text-[#ECECEC] focus:outline-none cursor-pointer"
              >
                <option value="">Auto-Detect</option>
                <option value="en">English (en)</option>
                <option value="hi">Hindi (hi)</option>
                <option value="hi-Latn">Hinglish (hi-Latn)</option>
                <option value="de">German (de)</option>
                <option value="es">Spanish (es)</option>
                <option value="fr">French (fr)</option>
              </select>
            )}
          </div>

          {/* Right: Model Label & Send / Microphone Action */}
          <div className="flex items-center gap-2">
            <div className="hidden sm:flex items-center gap-1.5 text-xs text-[#8C8C8C] pr-1">
              <span>Sonnet 3.5</span>
              <span className="text-[10px] bg-white/[0.06] px-1.5 py-0.5 rounded text-[#A3A3A3]">
                RAG
              </span>
            </div>

            <button
              type="button"
              title="Voice input"
              className="p-1.5 rounded-lg text-[#8C8C8C] hover:text-[#ECECEC] hover:bg-white/[0.06] transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 003-3V5a3 3 0 10-6 0v6a3 3 0 003 3z" />
              </svg>
            </button>

            {/* Send Button */}
            <button
              type="submit"
              disabled={!value.trim() || loading}
              aria-label="Send message"
              className={`w-7 h-7 rounded-lg flex items-center justify-center transition-all ${
                value.trim() && !loading
                  ? 'bg-[#DA7756] text-white hover:opacity-90 shadow-sm'
                  : 'bg-white/[0.06] text-[#666666] cursor-not-allowed'
              }`}
            >
              {loading ? (
                <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
              ) : (
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.2} d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>
    </form>
  );
};
