import React from 'react';
import { ChatMessage } from '../types/chat';

interface MessageBubbleProps {
  message: ChatMessage;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const isUser = message.sender === 'user';
  const isSystem = message.sender === 'system';

  if (isSystem) {
    return (
      <div className="flex justify-center my-4 animate-fade-in">
        <div className="px-3 py-1.5 rounded-lg bg-rose-500/10 border border-rose-500/20 text-xs text-rose-300 font-mono text-center">
          {message.text}
        </div>
      </div>
    );
  }

  return (
    <div
      className={`w-full flex my-6 animate-fade-in ${
        isUser ? 'justify-end' : 'justify-start'
      }`}
    >
      <div
        className={`max-w-[720px] w-full ${
          isUser ? 'flex flex-col items-end pl-12' : 'flex flex-col items-start pr-8'
        }`}
      >
        {/* User Message: Clean dark rounded container */}
        {isUser ? (
          <div className="rounded-2xl bg-[#282828] text-[#ECECEC] px-4 py-3 text-[15px] leading-relaxed shadow-xs max-w-full">
            <p className="whitespace-pre-wrap">{message.text}</p>
          </div>
        ) : (
          /* Claude AI Response: Typographic clean prose directly on canvas */
          <div className="w-full space-y-3">
            {/* Claude Avatar & Name Header */}
            <div className="flex items-center gap-2 text-xs text-[#9E9E9E] mb-1">
              <span className="text-[#DA7756] text-sm">✳</span>
              <span className="font-medium text-[#ECECEC]">Claude</span>
              <span className="text-[11px] text-[#666666] font-mono">{message.timestamp}</span>
            </div>

            {/* AI Text Prose */}
            <div className="text-[15px] leading-[1.65] text-[#ECECEC] font-normal pl-0.5">
              <p className="whitespace-pre-wrap">{message.text}</p>
            </div>

            {/* Subtle Diagnostic Pills below response */}
            {message.intent && (
              <div className="flex items-center gap-2 pt-1 text-[11px] text-[#8C8C8C] font-mono">
                <span className="px-2 py-0.5 rounded bg-white/[0.04] border border-white/[0.06] text-[#A3A3A3]">
                  {message.intent}
                </span>
                {message.confidence !== undefined && (
                  <span className="text-[#666666]">
                    {Math.round(message.confidence * 100)}% match
                  </span>
                )}
                {message.retrieved_cases !== undefined && (
                  <span className="text-[#666666]">
                    • {message.retrieved_cases} cases retrieved
                  </span>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
