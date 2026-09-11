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
        <div className="px-3.5 py-1.5 rounded-lg bg-rose-500/10 border border-rose-500/20 text-xs text-rose-300 font-mono text-center">
          {message.text}
        </div>
      </div>
    );
  }

  return (
    <div
      className={`w-full flex my-5 animate-fade-in ${
        isUser ? 'justify-end' : 'justify-start'
      }`}
    >
      <div
        className={`max-w-[680px] w-full ${
          isUser ? 'flex flex-col items-end pl-10' : 'flex flex-col items-start pr-6'
        }`}
      >
        {/* Header: Sender & Timestamp */}
        <div className="flex items-center gap-2 mb-1 text-[11px] text-[#6B7280]">
          <span className="font-medium text-[#9CA3AF]">
            {isUser ? 'Customer' : 'Hiver Support AI'}
          </span>
          <span>•</span>
          <span className="font-mono">{message.timestamp}</span>
        </div>

        {/* Message Content */}
        {isUser ? (
          <div className="rounded-2xl bg-[#181824] border border-[rgba(255,255,255,0.08)] px-4 py-3 text-[14px] text-[#F3F4F6] leading-relaxed shadow-sm">
            <p className="whitespace-pre-wrap">{message.text}</p>
          </div>
        ) : (
          <div className="w-full text-[14.5px] leading-[1.65] text-[#E5E7EB] font-normal pl-0.5">
            {message.isStreaming && !message.text ? (
              <div className="flex items-center gap-2 py-1 text-xs text-[#9CA3AF]">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-beacon" />
                <span className="text-[13px] text-[#9CA3AF]">Connecting to support intelligence...</span>
              </div>
            ) : (
              <p className="whitespace-pre-wrap inline">
                {message.text}
                {message.isStreaming && (
                  <span className="inline-block w-1.5 h-4 ml-1 bg-amber-400/80 animate-pulse align-middle rounded-xs" />
                )}
              </p>
            )}
          </div>
        )}

        {/* Subtle Intent Tag on AI replies */}
        {!isUser && message.intent && !message.isStreaming && (
          <div className="flex items-center gap-2 mt-2 pl-0.5 text-[11px] font-mono text-[#6B7280]">
            <span className="px-2 py-0.5 rounded bg-[#15151C] border border-white/[0.06] text-[#9CA3AF]">
              {message.intent}
            </span>
            {message.confidence !== undefined && (
              <span>{Math.round(message.confidence * 100)}% match</span>
            )}
            {message.retrieved_cases !== undefined && (
              <span>• {message.retrieved_cases} cases retrieved</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
