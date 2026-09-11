'use client';

import React, { useState, useRef, useEffect } from 'react';
import { useChat } from '../hooks/useChat';
import { MessageBubble } from './MessageBubble';
import { OperationsPanel } from './OperationsPanel';
import { ChatComposer } from './ChatComposer';

const SAMPLE_PROMPTS = [
  'My package has not arrived yet',
  'I need a refund for cancelled order #108-9812',
  'Mera order deliver nahi hua abhi tak',
  'Someone compromised my account with unauthorized orders',
  'How do I return a damaged product?',
];

export const ChatWindow: React.FC = () => {
  const {
    messages,
    loading,
    isStreaming,
    streamingStatus,
    error,
    sessionId,
    health,
    activeAnalysis,
    sendMessage,
    stopGeneration,
    retryLastMessage,
    clearChat,
  } = useChat();

  const [inputQuery, setInputQuery] = useState('');
  const [selectedLanguage, setSelectedLanguage] = useState('');
  const [showMobileOperations, setShowMobileOperations] = useState(false);
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading, isStreaming]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputQuery.trim() || loading) return;
    sendMessage(inputQuery, selectedLanguage || undefined);
    setInputQuery('');
  };

  const hasMessages = messages.length > 1;

  return (
    <div className="flex flex-col h-screen w-full bg-[#0B0B0F] text-[#F3F4F6] overflow-hidden font-sans">
      {/* -------------------------------------------------
          TOP HEADER
          ------------------------------------------------- */}
      <header className="h-14 border-b border-[rgba(255,255,255,0.07)] px-6 bg-[#0E0E14]/90 backdrop-blur-md flex items-center justify-between z-20 shrink-0">
        {/* Left: Brand & Subtitle */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-amber-500 to-amber-600 flex items-center justify-center font-bold text-black text-sm shadow-[0_0_16px_rgba(245,158,11,0.2)]">
            H
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-sm font-semibold tracking-tight text-[#F3F4F6]">
                Hiver AI Support Agent
              </h1>
              <span className="hidden sm:inline-block px-1.5 py-0.2 rounded text-[10px] font-mono bg-amber-500/10 text-amber-400 border border-amber-500/20">
                PROD-v1.0
              </span>
            </div>
            <p className="text-[11px] text-[#9CA3AF] hidden md:block">
              Hybrid Intent Classification + Pinecone Vector Retrieval + Gemini RAG
            </p>
          </div>
        </div>

        {/* Right Actions */}
        <div className="flex items-center gap-3 text-xs">
          {/* System Status Beacon */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[#15151C] border border-[rgba(255,255,255,0.08)]">
            <span
              className={`w-2 h-2 rounded-full ${
                health.status === 'healthy'
                  ? 'bg-emerald-400 animate-beacon'
                  : 'bg-amber-400'
              }`}
            />
            <span className="text-[11px] font-mono text-[#D1D5DB]">
              AI Systems Online
            </span>
          </div>

          {/* New Conversation Button */}
          <button
            onClick={clearChat}
            className="px-3 py-1.5 rounded-lg border border-[rgba(255,255,255,0.08)] bg-[#15151C] hover:bg-[#1C1C26] text-xs font-medium text-[#F3F4F6] transition-colors flex items-center gap-1.5"
            title="Start new support conversation"
          >
            <span className="text-sm leading-none text-amber-400">+</span>
            <span>New Conversation</span>
          </button>

          {/* Settings Button */}
          <button
            onClick={() => setShowSettingsModal(true)}
            className="p-1.5 rounded-lg border border-[rgba(255,255,255,0.08)] bg-[#15151C] hover:bg-[#1C1C26] text-[#9CA3AF] hover:text-[#F3F4F6] transition-colors"
            title="Console Settings"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </button>

          {/* Toggle Mobile Operations Drawer */}
          <button
            onClick={() => setShowMobileOperations(!showMobileOperations)}
            className="lg:hidden p-1.5 rounded-lg border border-[rgba(255,255,255,0.08)] bg-[#15151C] text-amber-400"
            title="Toggle Operations Panel"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </button>
        </div>
      </header>

      {/* -------------------------------------------------
          MAIN AREA (70% LEFT CONVERSATION, 30% RIGHT AI OPERATIONS)
          ------------------------------------------------- */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* LEFT 70%: Customer Conversation Window */}
        <section className="flex-1 lg:w-[70%] flex flex-col h-full relative bg-[#0B0B0F]">
          {/* Scrollable Conversation History */}
          <div className="flex-1 overflow-y-auto px-4 sm:px-8 pt-6 pb-36">
            <div className="max-w-[760px] mx-auto w-full">
              {/* Sample Prompts on Initial Blank Canvas */}
              {!hasMessages && (
                <div className="pt-16 pb-8 text-center animate-fade-in">
                  <h2 className="text-xl font-semibold tracking-tight text-[#F3F4F6]">
                    Amazon Customer Support AI
                  </h2>
                  <p className="text-xs text-[#9CA3AF] mt-1.5 max-w-md mx-auto leading-relaxed">
                    Test the agent with order status, late deliveries, refund inquiries, or account security incidents.
                  </p>

                  <div className="flex flex-wrap justify-center gap-2 mt-6 max-w-xl mx-auto">
                    {SAMPLE_PROMPTS.map((prompt, idx) => (
                      <button
                        key={idx}
                        onClick={() => sendMessage(prompt, selectedLanguage || undefined)}
                        disabled={loading}
                        className="text-left px-3.5 py-2 rounded-xl bg-[#111116] hover:bg-[#15151C] border border-[rgba(255,255,255,0.06)] hover:border-[rgba(255,255,255,0.14)] text-xs text-[#9CA3AF] hover:text-[#F3F4F6] transition-all duration-150"
                      >
                        <span>"{prompt}"</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Message List */}
              {messages.map((msg) => (
                <MessageBubble key={msg.id} message={msg} />
              ))}

              {/* Real-time Typing / Pipeline Animation */}
              {loading && !messages.some((m) => m.isStreaming && m.text) && (
                <div className="flex items-center gap-2 my-5 text-xs text-[#9CA3AF] animate-fade-in pl-1">
                  <div className="flex items-center gap-1.5 py-1.5 px-3 rounded-full bg-[#15151C] border border-white/[0.06]">
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-beacon" />
                    <span className="font-mono text-[11px] text-[#9CA3AF]">
                      Predicting intent & retrieving verified cases...
                    </span>
                  </div>
                </div>
              )}

              {/* Error Box with Retry */}
              {error && (
                <div className="flex items-center justify-between p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/20 text-xs text-rose-300 my-4">
                  <span>{error}</span>
                  <button
                    onClick={retryLastMessage}
                    className="px-3 py-1 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 font-medium transition-colors"
                  >
                    Retry
                  </button>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* -------------------------------------------------
              BOTTOM: Floating Input Composer
              ------------------------------------------------- */}
          <div className="absolute bottom-5 left-0 right-0 z-20 px-4 pointer-events-none flex justify-center">
            <div className="w-full max-w-[760px] pointer-events-auto">
              <ChatComposer
                value={inputQuery}
                onChange={setInputQuery}
                onSubmit={handleSubmit}
                loading={loading}
                isStreaming={isStreaming}
                onStop={stopGeneration}
                selectedLanguage={selectedLanguage}
                onLanguageChange={setSelectedLanguage}
              />
            </div>
          </div>
        </section>

        {/* RIGHT 30%: AI Operations Panel */}
        <section
          className={`fixed lg:static top-14 bottom-0 right-0 z-30 w-80 sm:w-96 lg:w-[30%] shrink-0 transition-transform duration-200 ease-in-out ${
            showMobileOperations ? 'translate-x-0' : 'translate-x-full lg:translate-x-0'
          }`}
        >
          <OperationsPanel
            analysis={activeAnalysis}
            health={health}
            sessionId={sessionId}
          />
        </section>
      </div>

      {/* Settings Modal Dialog */}
      {showSettingsModal && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-xs flex items-center justify-center p-4 animate-fade-in">
          <div className="w-full max-w-md bg-[#15151C] border border-white/[0.08] rounded-2xl p-5 shadow-2xl space-y-4 font-sans text-xs">
            <div className="flex items-center justify-between pb-3 border-b border-white/[0.06]">
              <h3 className="text-sm font-semibold text-[#F3F4F6]">
                Console Settings & Runtime Config
              </h3>
              <button
                onClick={() => setShowSettingsModal(false)}
                className="text-[#9CA3AF] hover:text-white"
              >
                ✕
              </button>
            </div>

            <div className="space-y-3 font-mono text-[11px] text-[#9CA3AF]">
              <div className="flex justify-between py-1 border-b border-white/[0.04]">
                <span>Backend API:</span>
                <span className="text-[#D1D5DB]">
                  {process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}
                </span>
              </div>
              <div className="flex justify-between py-1 border-b border-white/[0.04]">
                <span>Vector Namespace:</span>
                <span className="text-[#D1D5DB]">amazon (52,124 vectors)</span>
              </div>
              <div className="flex justify-between py-1 border-b border-white/[0.04]">
                <span>Classifier Model:</span>
                <span className="text-[#D1D5DB]">paraphrase-multilingual-MiniLM-L12-v2</span>
              </div>
              <div className="flex justify-between py-1 border-b border-white/[0.04]">
                <span>LLM Engine:</span>
                <span className="text-[#D1D5DB]">gemini-3.5-flash-lite</span>
              </div>
              <div className="flex justify-between py-1 border-b border-white/[0.04]">
                <span>Active Session:</span>
                <span className="text-amber-400">{sessionId}</span>
              </div>
            </div>

            <div className="pt-2 flex justify-end">
              <button
                onClick={() => setShowSettingsModal(false)}
                className="px-4 py-1.5 rounded-lg bg-white/[0.08] hover:bg-white/[0.14] text-[#F3F4F6] font-medium"
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
