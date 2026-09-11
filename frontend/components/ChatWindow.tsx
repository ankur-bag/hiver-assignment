'use client';

import React, { useState, useRef, useEffect } from 'react';
import { useChat } from '../hooks/useChat';
import { ClaudeSidebar } from './ClaudeSidebar';
import { ClaudeComposer } from './ClaudeComposer';
import { MessageBubble } from './MessageBubble';
import { TelemetryPanel } from './TelemetryPanel';

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
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [showTelemetry, setShowTelemetry] = useState(false);
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

  const hasStarted = messages.length > 1;

  return (
    <div className="flex h-screen w-full bg-[#1E1E1E] text-[#ECECEC] overflow-hidden font-sans">
      {/* 1. Claude Exact Left Sidebar */}
      <ClaudeSidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        onNewChat={clearChat}
        activeSessionId={sessionId}
      />

      {/* 2. Main Large Canvas Area */}
      <div className="flex-1 flex flex-col h-full relative overflow-hidden bg-[#1E1E1E]">
        {/* Top Minimalist Header */}
        <header className="h-12 border-b border-[rgba(255,255,255,0.06)] px-4 flex items-center justify-between text-xs text-[#9E9E9E]">
          {/* Left: Mobile sidebar toggle + Model Title */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              className="lg:hidden p-1.5 rounded-md hover:bg-white/[0.06] text-[#9E9E9E] hover:text-[#ECECEC]"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            <div className="flex items-center gap-2">
              <span className="font-serif-claude text-sm text-[#ECECEC]">Claude 3.5 Sonnet</span>
              <span className="text-[11px] text-[#666666] hidden sm:inline">• Enterprise Support</span>
            </div>
          </div>

          {/* Right: Plan, Diagnostics & Profile */}
          <div className="flex items-center gap-3">
            <span className="text-[#8C8C8C] text-[11px] hidden sm:inline">
              Free plan · <span className="text-[#DA7756] hover:underline cursor-pointer">Upgrade</span>
            </span>

            {/* Telemetry Button */}
            <button
              onClick={() => setShowTelemetry(!showTelemetry)}
              className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-[#282828] hover:bg-[#323232] border border-white/[0.06] text-[11px] font-mono text-[#ECECEC] transition-colors"
              title="Toggle System Diagnostics"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              <span className="hidden sm:inline">Diagnostics</span>
            </button>

            {/* Profile Avatar */}
            <div className="w-6 h-6 rounded-full bg-[#323232] border border-white/[0.08] flex items-center justify-center text-[10px] text-[#ECECEC] font-mono cursor-pointer">
              H
            </div>
          </div>
        </header>

        {/* Telemetry Popover Panel */}
        <TelemetryPanel
          analysis={activeAnalysis}
          health={health}
          sessionId={sessionId}
          isOpen={showTelemetry}
          onClose={() => setShowTelemetry(false)}
        />

        {/* Conversation Stream OR Claude Greeting State */}
        <div className="flex-1 overflow-y-auto px-4 sm:px-6">
          <div className="max-w-[720px] mx-auto min-h-full flex flex-col justify-between pt-8 pb-32">
            {!hasStarted ? (
              /* Claude Exact Greeting Center */
              <div className="my-auto flex flex-col items-center text-center animate-fade-in py-12">
                <div className="flex items-center gap-3 text-2xl sm:text-3xl text-[#ECECEC] font-serif-claude mb-2 tracking-tight">
                  <span className="text-[#DA7756] text-3xl">✳</span>
                  <span>Back at it, how can I help?</span>
                </div>
                <p className="text-xs text-[#8C8C8C] mt-1 font-sans">
                  Amazon Customer Support AI with Pinecone Vector Retrieval & Gemini RAG
                </p>
              </div>
            ) : (
              /* Conversation Messages */
              <div className="space-y-2">
                {messages.map((msg) => (
                  <MessageBubble key={msg.id} message={msg} />
                ))}

                {/* Claude Thinking indicator */}
                {loading && (
                  <div className="flex items-center gap-2 text-xs text-[#9E9E9E] my-4 animate-fade-in pl-1">
                    <span className="text-[#DA7756] animate-spin text-sm">✳</span>
                    <span className="font-serif-claude text-sm text-[#CCCCCC]">Claude is thinking...</span>
                  </div>
                )}

                {/* Error with retry */}
                {error && (
                  <div className="flex items-center justify-between p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-xs text-rose-300 my-4">
                    <span>{error}</span>
                    <button
                      onClick={retryLastMessage}
                      className="px-2.5 py-1 rounded bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 font-medium transition-colors"
                    >
                      Retry
                    </button>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
        </div>

        {/* 3. Floating Bottom Center Claude Composer */}
        <div className="absolute bottom-6 left-0 right-0 z-30 px-4 pointer-events-none flex justify-center">
          <div className="w-full max-w-[720px] pointer-events-auto">
            <ClaudeComposer
              value={inputQuery}
              onChange={setInputQuery}
              onSubmit={handleSubmit}
              loading={loading}
              selectedLanguage={selectedLanguage}
              onLanguageChange={setSelectedLanguage}
            />
          </div>
        </div>
      </div>
    </div>
  );
};
