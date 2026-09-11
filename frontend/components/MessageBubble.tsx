'use client';

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
        <div className="px-3 py-1.5 rounded-lg bg-rose-500/10 border border-rose-500/15 text-[12px] text-rose-300 text-center">
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
        className={`max-w-[680px] w-full ${
          isUser ? 'flex flex-col items-end pl-8' : 'flex flex-col items-start pr-8'
        }`}
      >
        {/* Header: Sender & Timestamp */}
        <div className="flex items-center gap-2 mb-1.5 text-[11.5px] text-[#8C8B88]">
          <span className="font-medium text-[#EDEDEC]">
            {isUser ? 'You' : 'Hiver Support'}
          </span>
          <span>·</span>
          <span suppressHydrationWarning>{message.timestamp}</span>
        </div>

        {/* Message Content */}
        {isUser ? (
          <div className="rounded-2xl bg-[#1E1D1B] border border-white/[0.06] px-4 py-3 text-[14px] text-[#EDEDEC] leading-relaxed">
            <p className="whitespace-pre-wrap">{message.text}</p>
          </div>
        ) : (
          <div className="w-full text-[14.5px] leading-[1.68] text-[#D6D5D4] font-normal pt-0.5">
            {message.isStreaming && !message.text ? (
              <div className="flex items-center gap-2 py-1 text-[13px] text-[#8C8B88]">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
                <span>Searching support history...</span>
              </div>
            ) : (
              <div className="whitespace-pre-wrap">
                {message.text}
                {message.isStreaming && (
                  <span className="inline-block w-1.5 h-4 ml-1 bg-amber-400/90 rounded-xs animate-caret align-middle" />
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
