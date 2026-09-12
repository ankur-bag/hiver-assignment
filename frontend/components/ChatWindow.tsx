'use client';

import React, { useState, useRef, useEffect } from 'react';
import { FiMenu, FiX, FiLayers } from 'react-icons/fi';
import { useChat } from '../hooks/useChat';
import { Sidebar } from './Sidebar';
import { AnalysisPanel } from './AnalysisPanel';
import { MessageBubble } from './MessageBubble';
import { ChatComposer } from './ChatComposer';

const SAMPLE_PROMPTS = [
  "My package hasn't arrived yet",
  "Where is my refund?",
  "I can't access my account",
  "Mera order deliver nahi hua",
];

export const ChatWindow: React.FC = () => {
  const {
    messages,
    loading,
    isStreaming,
    error,
    sessionId,
    health,
    analysisState,
    sendMessage,
    stopGeneration,
    retryLastMessage,
    clearChat,
  } = useChat();

  const [inputQuery, setInputQuery] = useState('');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isAnalysisOpen, setIsAnalysisOpen] = useState(true);
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
    sendMessage(inputQuery);
    setInputQuery('');
  };

  const hasCustomerMessage = messages.some((m) => m.sender === 'user');

  return (
    <div className="flex h-screen w-full bg-[#171615] text-[#D6D5D4] overflow-hidden">
      {/* -------------------------------------------------
          1. LEFT SIDEBAR (Collapsible, ~230px)
          ------------------------------------------------- */}
      <Sidebar
        isOpen={isSidebarOpen}
        onToggle={() => setIsSidebarOpen(!isSidebarOpen)}
        onNewChat={clearChat}
        onOpenSettings={() => setShowSettingsModal(true)}
        health={health}
      />

      {/* -------------------------------------------------
          2. CENTER CONVERSATION AREA
          ------------------------------------------------- */}
      <div className="flex-1 flex flex-col h-full min-w-0 bg-[#171615] relative">
        {/* Top Minimal Bar */}
        <header className="h-14 px-4 sm:px-6 border-b border-white/[0.06] flex items-center justify-between shrink-0 bg-[#171615]">
          <div className="flex items-center gap-3">
            {!isSidebarOpen && (
              <button
                onClick={() => setIsSidebarOpen(true)}
                className="p-1.5 rounded-lg hover:bg-white/[0.06] text-[#8C8B88] hover:text-[#EDEDEC] transition-colors cursor-pointer"
                title="Open sidebar"
                aria-label="Open sidebar"
              >
                <FiMenu className="w-4 h-4" />
              </button>
            )}
            <div>
              <h1 className="text-[13.5px] font-medium text-[#EDEDEC]">
                Hiver AI Support
              </h1>
              <p className="text-[11px] text-[#8C8B88] hidden sm:block">
                Amazon customer support assistant
              </p>
            </div>
          </div>

          {/* Right Action Icons (Analysis toggle for responsive screens) */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsAnalysisOpen(!isAnalysisOpen)}
              className={`p-1.5 rounded-lg border border-white/[0.06] text-xs transition-colors lg:hidden cursor-pointer ${
                isAnalysisOpen
                  ? 'bg-white/[0.08] text-amber-400'
                  : 'bg-white/[0.03] text-[#8C8B88] hover:text-[#EDEDEC]'
              }`}
              title="Toggle Analysis"
              aria-label="Toggle Analysis"
            >
              <FiLayers className="w-4 h-4" />
            </button>
          </div>
        </header>

        {/* Scrollable Conversation Container */}
        <div className="flex-1 overflow-y-auto px-4 sm:px-8 pt-4 pb-48">
          <div className="max-w-[780px] mx-auto w-full">
            {/* Message Stream List */}
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}

            {/* Error Message banner with retry */}
            {error && (
              <div className="flex items-center justify-between p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/20 text-[12.5px] text-rose-300 my-4 animate-fade-in">
                <span>{error}</span>
                <button
                  onClick={retryLastMessage}
                  className="px-3 py-1 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 text-xs font-medium transition-colors cursor-pointer"
                >
                  Retry
                </button>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Floating Input Composer & Recommended Questions Area */}
        <div className="absolute bottom-4 left-0 right-0 z-20 px-4 flex flex-col items-center pointer-events-none">
          <div className="w-full max-w-[780px] pointer-events-auto space-y-3">
            {/* Recommended Questions: Placed neatly just above the input box */}
            {!hasCustomerMessage && (
              <div className="space-y-2 animate-fade-in">
                <div className="flex items-center justify-between px-1 text-[11.5px] text-[#8C8B88]">
                  <span>Suggested questions</span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {SAMPLE_PROMPTS.map((prompt, idx) => (
                    <button
                      key={idx}
                      onClick={() => sendMessage(prompt)}
                      disabled={loading}
                      className="text-left px-3.5 py-2.5 rounded-xl bg-[#1E1D1B] hover:bg-[#252422] border border-white/[0.06] hover:border-white/[0.12] text-[12.5px] text-[#D6D5D4] hover:text-[#EDEDEC] transition-all duration-150 cursor-pointer shadow-sm"
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Input Composer */}
            <ChatComposer
              value={inputQuery}
              onChange={setInputQuery}
              onSubmit={handleSubmit}
              loading={loading}
              isStreaming={isStreaming}
              onStop={stopGeneration}
            />
          </div>
        </div>
      </div>

      {/* -------------------------------------------------
          3. RIGHT ANALYSIS PANEL (280px-300px)
          ------------------------------------------------- */}
      <div
        className={`fixed lg:static inset-y-0 right-0 z-30 w-[290px] shrink-0 transition-transform duration-200 ease-in-out ${
          isAnalysisOpen ? 'translate-x-0' : 'translate-x-full lg:hidden'
        }`}
      >
        <AnalysisPanel analysis={analysisState} />
      </div>

      {/* Settings Modal Dialog */}
      {showSettingsModal && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-xs flex items-center justify-center p-4 animate-fade-in">
          <div className="w-full max-w-sm bg-[#1E1D1B] border border-white/[0.08] rounded-2xl p-5 shadow-2xl space-y-4 text-xs">
            <div className="flex items-center justify-between pb-3 border-b border-white/[0.06]">
              <h3 className="text-[13px] font-medium text-[#EDEDEC]">
                System Information
              </h3>
              <button
                onClick={() => setShowSettingsModal(false)}
                className="p-1 rounded-md text-[#8C8B88] hover:text-[#EDEDEC] transition-colors cursor-pointer"
              >
                <FiX className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-2.5 text-[12px] text-[#8C8B88]">
              <div className="flex justify-between py-1 border-b border-white/[0.04]">
                <span>Backend service</span>
                <span className="text-[#EDEDEC] truncate max-w-[200px]">
                  {process.env.NEXT_PUBLIC_API_URL || 'https://hiver-assignment-yd38.onrender.com'}
                </span>
              </div>
              <div className="flex justify-between py-1 border-b border-white/[0.04]">
                <span>Session ID</span>
                <span className="text-amber-400 font-mono text-[10.5px]">
                  {sessionId || '—'}
                </span>
              </div>
              <div className="flex justify-between py-1 border-b border-white/[0.04]">
                <span>Status</span>
                <span className="text-emerald-400 capitalize">
                  {health.status}
                </span>
              </div>
            </div>

            <div className="pt-2 flex justify-end">
              <button
                onClick={() => setShowSettingsModal(false)}
                className="px-4 py-1.5 rounded-lg bg-white/[0.06] hover:bg-white/[0.1] text-[#EDEDEC] text-xs font-medium transition-colors cursor-pointer"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
