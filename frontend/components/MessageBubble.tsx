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
      <div className="flex justify-center my-6 animate-fade-in">
        <div className="px-3.5 py-1.5 rounded-full bg-rose-500/[0.08] border border-rose-500/20 text-xs text-rose-300 font-mono text-center">
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
        className={`max-w-[760px] w-full ${
          isUser
            ? 'flex flex-col items-end pl-12'
            : 'flex flex-col items-start pr-12'
        }`}
      >
        {/* Author Label & Timestamp */}
        <div className="flex items-center gap-2 mb-1.5 text-xs text-[#71717A]">
          <span className="font-medium text-[#A1A1AA]">
            {isUser ? 'You' : 'Hiver AI'}
          </span>
          <span>•</span>
          <span className="text-[11px] font-mono">{message.timestamp}</span>
        </div>

        {/* Message Content */}
        {isUser ? (
          <div className="rounded-2xl bg-[#15151C] border border-[rgba(255,255,255,0.08)] px-5 py-3.5 text-[15px] sm:text-[16px] text-[#F5F5F5] leading-relaxed shadow-sm">
            <p className="whitespace-pre-wrap">{message.text}</p>
          </div>
        ) : (
          <div className="w-full text-[15px] sm:text-[16px] text-[#F5F5F5] leading-relaxed pl-1">
            <p className="whitespace-pre-wrap">{message.text}</p>
          </div>
        )}

        {/* Subtle AI Metadata Tag */}
        {!isUser && message.intent && (
          <div className="flex items-center gap-2 mt-2 pl-1 text-[11px] font-mono text-[#71717A]">
            <span>Intent: {message.intent}</span>
            {message.confidence !== undefined && (
              <>
                <span>•</span>
                <span>{Math.round(message.confidence * 100)}% confidence</span>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
