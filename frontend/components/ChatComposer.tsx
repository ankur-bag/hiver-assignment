'use client';

import React, { useRef, useEffect } from 'react';

interface ChatComposerProps {
  value: string;
  onChange: (val: string) => void;
  onSubmit: (e: React.FormEvent) => void;
  loading: boolean;
  isStreaming?: boolean;
  onStop?: () => void;
  selectedLanguage: string;
  onLanguageChange: (lang: string) => void;
}

export const ChatComposer: React.FC<ChatComposerProps> = ({
  value,
  onChange,
  onSubmit,
  loading,
  isStreaming,
  onStop,
  selectedLanguage,
  onLanguageChange,
}) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 180)}px`;
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
      <div className="relative rounded-2xl bg-[#15151C] border border-[rgba(255,255,255,0.08)] hover:border-[rgba(255,255,255,0.13)] focus-within:border-amber-500/50 shadow-[0_8px_32px_rgba(0,0,0,0.4)] transition-all">
        {/* Textarea */}
        <textarea
          ref={textareaRef}
          rows={1}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about orders, refunds, delivery, account issues..."
          disabled={loading}
          className="w-full bg-transparent px-4 pt-3.5 pb-11 text-[14px] text-[#F3F4F6] placeholder-[#6B7280] resize-none focus:outline-none leading-relaxed min-h-[50px]"
        />

        {/* Action Toolbar Inside Bottom of Input */}
        <div className="absolute left-3 bottom-2.5 right-3 flex items-center justify-between pointer-events-auto">
          {/* Left toolbar items */}
          <div className="flex items-center gap-2">
            {/* Language Selector Pill */}
            <div className="flex items-center gap-1.5 bg-[#1C1C26] border border-white/[0.06] rounded-lg px-2 py-0.5 text-[11px] text-[#9CA3AF]">
              <span className="text-amber-400 text-[10px]">🌐</span>
              <select
                value={selectedLanguage}
                onChange={(e) => onLanguageChange(e.target.value)}
                aria-label="Target language selector"
                disabled={loading}
                className="bg-transparent text-[#D1D5DB] hover:text-white focus:outline-none cursor-pointer pr-1"
              >
                <option value="" className="bg-[#15151C]">Auto-Detect</option>
                <option value="en" className="bg-[#15151C]">English (en)</option>
                <option value="hi" className="bg-[#15151C]">Hindi (hi)</option>
                <option value="hi-Latn" className="bg-[#15151C]">Hinglish (hi-Latn)</option>
                <option value="de" className="bg-[#15151C]">German (de)</option>
                <option value="es" className="bg-[#15151C]">Spanish (es)</option>
                <option value="fr" className="bg-[#15151C]">French (fr)</option>
              </select>
            </div>

            <span className="text-[11px] text-[#4B5563] hidden md:inline font-mono">
              ↵ to send · shift+↵ for new line
            </span>
          </div>

          {/* Right action button (Send or Stop) */}
          {isStreaming ? (
            <button
              type="button"
              onClick={onStop}
              aria-label="Stop generating response"
              className="h-7 px-3 rounded-lg text-xs font-medium flex items-center gap-1.5 bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 border border-rose-500/30 transition-all font-mono"
            >
              <span className="w-2 h-2 rounded-xs bg-rose-400" />
              <span>Stop</span>
            </button>
          ) : (
            <button
              type="submit"
              disabled={!value.trim() || loading}
              aria-label="Send message"
              className={`h-7 px-3 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-all ${
                value.trim() && !loading
                  ? 'bg-amber-500 hover:bg-amber-400 text-black shadow-sm font-semibold'
                  : 'bg-white/[0.06] text-[#6B7280] cursor-not-allowed'
              }`}
            >
              {loading ? (
                <>
                  <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                  </svg>
                  <span>Connecting</span>
                </>
              ) : (
                <>
                  <span>Send</span>
                  <span className="text-[10px]">→</span>
                </>
              )}
            </button>
          )}
        </div>
      </div>
    </form>
  );
};

