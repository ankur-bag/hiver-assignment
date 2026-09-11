'use client';

import React, { useState, useRef, useEffect } from 'react';
import { useChat } from '../hooks/useChat';
import { MessageBubble } from './MessageBubble';
import { TelemetryPanel } from './TelemetryPanel';

const QUICK_PROMPTS = [
  'Where is my package? It was supposed to arrive yesterday.',
  'I was charged twice for order #402-998812 and need a refund.',
  'How do I return a damaged pair of shoes I received?',
  'Someone hacked into my account and placed unauthorized orders!',
  'Mein Paket ist verspätet, wo ist meine Bestellung?',
  'Mera order delay ho gaya hai, kab deliver hoga?',
];

export const ChatWindow: React.FC = () => {
  const {
    messages,
    loading,
    error,
    sessionId,
    health,
    activeAnalysis,
    sendMessage,
    retryLastMessage,
    clearChat,
  } = useChat();

  const [inputQuery, setInputQuery] = useState('');
  const [selectedLanguage, setSelectedLanguage] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputQuery.trim() || loading) return;
    sendMessage(inputQuery, selectedLanguage || undefined);
    setInputQuery('');
  };

  return (
    <div className="flex h-screen w-full bg-slate-950 text-slate-100 overflow-hidden font-sans">
      {/* Left / Main Section: Customer Support Chat Experience */}
      <div className="flex-1 flex flex-col h-full relative">
        {/* Chat Top Bar */}
        <header className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900/60 backdrop-blur-md z-10">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-amber-500 to-orange-400 flex items-center justify-center font-bold text-slate-950 shadow-md">
              H
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-bold text-slate-100">
                  Hiver AI Support Agent
                </h1>
                <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-amber-500/10 text-amber-400 border border-amber-500/30">
                  AmazonHelp RAG
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Hybrid Intent-Vector Retrieval with Gemini Response Synthesis
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Language Selector Override */}
            <select
              value={selectedLanguage}
              onChange={(e) => setSelectedLanguage(e.target.value)}
              className="bg-slate-800 border border-slate-700 text-xs rounded-lg px-2.5 py-1.5 text-slate-300 focus:outline-none focus:ring-1 focus:ring-amber-500"
            >
              <option value="">Auto-Detect Language</option>
              <option value="en">English (en)</option>
              <option value="hi">Hindi (hi)</option>
              <option value="hi-Latn">Hinglish (hi-Latn)</option>
              <option value="de">German (de)</option>
              <option value="es">Spanish (es)</option>
              <option value="fr">French (fr)</option>
            </select>

            <button
              onClick={clearChat}
              title="Reset conversation"
              className="px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-800/80 hover:bg-slate-700 text-xs text-slate-300 transition-colors"
            >
              New Chat
            </button>
          </div>
        </header>

        {/* Message Stream */}
        <div className="flex-1 overflow-y-auto p-6 space-y-2">
          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}

          {loading && (
            <div className="flex items-center gap-2 text-xs text-slate-400 my-3 px-2">
              <div className="w-2 h-2 rounded-full bg-amber-400 animate-bounce" />
              <div className="w-2 h-2 rounded-full bg-amber-400 animate-bounce [animation-delay:0.2s]" />
              <div className="w-2 h-2 rounded-full bg-amber-400 animate-bounce [animation-delay:0.4s]" />
              <span className="font-mono ml-2">Classifying intent & searching past resolutions...</span>
            </div>
          )}

          {error && (
            <div className="flex items-center justify-between p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-xs text-rose-300">
              <span>{error}</span>
              <button
                onClick={retryLastMessage}
                className="px-2.5 py-1 bg-rose-500 text-white rounded font-medium hover:bg-rose-600 transition"
              >
                Retry
              </button>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Quick Suggestion Prompts */}
        <div className="px-6 py-2 border-t border-slate-800/50 bg-slate-950/40">
          <div className="flex items-center gap-2 overflow-x-auto no-scrollbar py-1">
            <span className="text-[11px] font-semibold text-slate-500 uppercase shrink-0">
              Try:
            </span>
            {QUICK_PROMPTS.map((prompt, idx) => (
              <button
                key={idx}
                onClick={() => sendMessage(prompt, selectedLanguage || undefined)}
                disabled={loading}
                className="text-xs shrink-0 px-3 py-1 rounded-full border border-slate-800 bg-slate-900/60 hover:bg-slate-800 hover:border-slate-700 text-slate-400 hover:text-slate-200 transition-colors"
              >
                {prompt}
              </button>
            ))}
          </div>
        </div>

        {/* Input Bar */}
        <form
          onSubmit={handleSubmit}
          className="p-4 border-t border-slate-800 bg-slate-900/40 backdrop-blur-sm"
        >
          <div className="flex items-center gap-3 max-w-4xl mx-auto">
            <input
              type="text"
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              placeholder="Ask about orders, delivery delays, refunds, returns..."
              disabled={loading}
              className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-amber-500/80 focus:ring-1 focus:ring-amber-500/80 transition-all"
            />
            <button
              type="submit"
              disabled={loading || !inputQuery.trim()}
              className="px-5 py-3 rounded-xl bg-amber-500 hover:bg-amber-400 disabled:opacity-50 disabled:hover:bg-amber-500 text-slate-950 font-semibold text-sm transition-all shadow-md flex items-center gap-1.5"
            >
              <span>Send</span>
              <span>→</span>
            </button>
          </div>
        </form>
      </div>

      {/* Right Section: Real-time ML Inference, Retrieval & Telemetry Panel */}
      <aside className="w-80 lg:w-96 hidden md:block shrink-0 h-full">
        <TelemetryPanel
          analysis={activeAnalysis}
          health={health}
          sessionId={sessionId}
        />
      </aside>
    </div>
  );
};
