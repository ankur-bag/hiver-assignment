import React from 'react';
import { ChatMessage } from '../types/chat';
import { ConfidenceBadge } from './ConfidenceBadge';

interface MessageBubbleProps {
  message: ChatMessage;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const isUser = message.sender === 'user';
  const isSystem = message.sender === 'system';

  if (isSystem) {
    return (
      <div className="flex justify-center my-3">
        <div className="px-4 py-2 rounded-lg bg-rose-500/10 border border-rose-500/20 text-xs text-rose-300 font-mono max-w-lg text-center">
          {message.text}
        </div>
      </div>
    );
  }

  return (
    <div className={`flex flex-col my-3 ${isUser ? 'items-end' : 'items-start'}`}>
      <div className="flex items-center gap-2 mb-1 px-1">
        <span className="text-[11px] font-semibold text-slate-400">
          {isUser ? 'Customer' : 'Amazon Support AI'}
        </span>
        <span className="text-[10px] text-slate-500">{message.timestamp}</span>
      </div>

      <div
        className={`max-w-[85%] sm:max-w-xl rounded-2xl px-4 py-3 shadow-md text-sm leading-relaxed ${
          isUser
            ? 'bg-blue-600 text-white rounded-br-sm'
            : 'bg-slate-900 border border-slate-800 text-slate-200 rounded-bl-sm'
        }`}
      >
        <p className="whitespace-pre-wrap">{message.text}</p>
      </div>

      {!isUser && message.confidence !== undefined && (
        <div className="flex items-center gap-2 mt-1.5 px-1">
          <ConfidenceBadge confidence={message.confidence} />
          {message.intent && (
            <span className="text-[10px] font-mono text-slate-400 bg-slate-800/80 px-2 py-0.5 rounded">
              {message.intent}
            </span>
          )}
        </div>
      )}
    </div>
  );
};
