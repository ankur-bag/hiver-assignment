'use client';

import React, { useState, useRef, useEffect } from 'react';
import { useChat } from '../hooks/useChat';
import { MessageBubble } from './MessageBubble';
import { TelemetryPanel } from './TelemetryPanel';
import { StatusIndicator } from './ui/StatusIndicator';
import { ChatComposer } from './ui/ChatComposer';

const STARTER_PROMPTS = [
  'My package is delayed',
  'Where is my refund?',
  'Mera order deliver nahi hua',
  'I need to return a damaged item',
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
  const [showTelemetry, setShowTelemetry] = useState(true);
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

  const hasMessages = messages.length > 1;

  return (
    <div className="flex h-screen w-full bg-[#0B0B0F] text-[#F5F5F5] overflow-hidden font-sans">
      {/* Main Conversation Workspace */}
      <div className="flex-1 flex flex-col h-full relative">
        {/* Top Minimalist Navigation Bar */}
        <header className="flex items-center justify-between px-6 py-3.5 border-b border-[rgba(255,255,255,0.08)] bg-[#0B0B0F]/90 backdrop-blur-md z-20">
          {/* Left: Brand Identity */}
          <div className="flex flex-col">
            <h1 className="text-[17px] font-semibold text-[#F5F5F5] tracking-tight">
              Hiver AI Support Agent
            </h1>
            <p className="text-[12px] text-[#A1A1AA]">
              Hybrid Intent + Vector Retrieval + Gemini RAG
            </p>
          </div>

          {/* Center: Status Indicator */}
          <div className="hidden sm:flex items-center">
            <StatusIndicator status={health.status} label="AI Systems Online" />
          </div>

          {/* Right: Controls */}
          <div className="flex items-center gap-3">
            {/* Language Selector */}
            <select
              value={selectedLanguage}
              onChange={(e) => setSelectedLanguage(e.target.value)}
              aria-label="Language selector"
              className="bg-[#15151C] border border-[rgba(255,255,255,0.08)] text-xs rounded-xl px-3 py-1.5 text-[#A1A1AA] hover:text-[#F5F5F5] focus:outline-none focus:border-[rgba(255,255,255,0.2)] transition-colors cursor-pointer"
            >
              <option value="">Auto-Detect</option>
              <option value="en">English (en)</option>
              <option value="hi">Hindi (hi)</option>
              <option value="hi-Latn">Hinglish (hi-Latn)</option>
              <option value="de">German (de)</option>
              <option value="es">Spanish (es)</option>
              <option value="fr">French (fr)</option>
            </select>

            {/* New Conversation Button */}
            <button
              onClick={clearChat}
              title="Start a new conversation"
              className="px-3.5 py-1.5 rounded-xl border border-[rgba(255,255,255,0.08)] bg-[#15151C] hover:bg-[#1A1A22] text-xs font-medium text-[#F5F5F5] transition-all"
            >
              New Chat
            </button>

            {/* Toggle Telemetry Panel on smaller screens */}
            <button
              onClick={() => setShowTelemetry(!showTelemetry)}
              title="Toggle AI Diagnostics Panel"
              className="lg:hidden p-1.5 rounded-xl border border-[rgba(255,255,255,0.08)] bg-[#15151C] text-[#A1A1AA] hover:text-[#F5F5F5]"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M4 6h16M4 12h16m-7 6h7" />
              </svg>
            </button>
          </div>
        </header>

        {/* Message Stream Area */}
        <div className="flex-1 overflow-y-auto px-4 sm:px-8 pt-6 pb-36">
          <div className="max-w-[760px] mx-auto w-full">
            {/* Empty Canvas Landing State */}
            {!hasMessages && (
              <div className="flex flex-col items-center justify-center pt-24 pb-12 text-center animate-fade-in">
                <div className="w-12 h-12 rounded-2xl bg-gradient-to-tr from-amber-500/20 to-orange-500/10 border border-amber-500/30 flex items-center justify-center text-xl font-bold text-amber-400 mb-5 shadow-[0_0_24px_rgba(245,158,11,0.15)]">
                  H
                </div>
                <h2 className="text-2xl sm:text-3xl font-semibold tracking-tight text-[#F5F5F5]">
                  Hiver AI Support Agent
                </h2>
                <p className="text-[14px] text-[#A1A1AA] mt-2 max-w-md leading-relaxed">
                  Ask about orders, refunds, delivery, or account issues.
                </p>

                {/* Example Quick Prompts */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 mt-8 w-full max-w-lg">
                  {STARTER_PROMPTS.map((prompt, idx) => (
                    <button
                      key={idx}
                      onClick={() => sendMessage(prompt, selectedLanguage || undefined)}
                      disabled={loading}
                      className="text-left px-4 py-3 rounded-xl bg-[#111116] hover:bg-[#15151C] border border-[rgba(255,255,255,0.06)] hover:border-[rgba(255,255,255,0.14)] text-xs text-[#A1A1AA] hover:text-[#F5F5F5] transition-all duration-200"
                    >
                      <span>"{prompt}"</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Render Conversation Messages */}
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}

            {/* Typing Indicator */}
            {loading && (
              <div className="flex items-center gap-3 my-6 animate-fade-in text-xs text-[#A1A1AA]">
                <div className="flex items-center gap-1.5 py-2 px-3 rounded-full bg-[#15151C] border border-[rgba(255,255,255,0.08)]">
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse-glow" />
                  <span className="font-mono text-[11px] text-[#A1A1AA]">
                    Reasoning & searching resolutions...
                  </span>
                </div>
              </div>
            )}

            {/* Error Message with Retry */}
            {error && (
              <div className="flex items-center justify-between p-3.5 my-4 rounded-xl bg-rose-500/[0.08] border border-rose-500/20 text-xs text-rose-300">
                <span>{error}</span>
                <button
                  onClick={retryLastMessage}
                  className="px-3 py-1 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 border border-rose-500/30 font-medium transition-colors"
                >
                  Retry
                </button>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Floating Bottom Composer */}
        <div className="fixed bottom-6 left-0 right-0 lg:right-80 z-20 pointer-events-none px-4 flex justify-center">
          <div className="w-full max-w-[760px] pointer-events-auto">
            <ChatComposer
              value={inputQuery}
              onChange={setInputQuery}
              onSubmit={handleSubmit}
              loading={loading}
              placeholder="Ask about orders, refunds, delivery issues..."
            />
          </div>
        </div>
      </div>

      {/* Developer AI Reasoning & Telemetry Side Panel */}
      <aside
        className={`${
          showTelemetry ? 'block' : 'hidden'
        } lg:block w-80 shrink-0 h-full z-10`}
      >
        <TelemetryPanel
          analysis={activeAnalysis}
          health={health}
          sessionId={sessionId}
        />
      </aside>
    </div>
  );
};
