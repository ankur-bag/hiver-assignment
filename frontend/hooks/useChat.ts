'use client';

import { useState, useCallback, useEffect } from 'react';
import { ChatMessage, ChatResponsePayload, ServiceHealth } from '../types/chat';
import { sendChatMessage, checkBackendHealth } from '../services/api';

/**
 * Custom React Hook for Chat State Management (Addon 6).
 * Encapsulates messages, loading, errors, retry logic, and backend health status.
 */
export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'init-msg-1',
      sender: 'agent',
      text: 'Hello! I am your Amazon Support Assistant. How can I assist you with your order, delivery, return, or account today?',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string>('');
  const [health, setHealth] = useState<ServiceHealth>({ status: 'healthy' });
  const [activeAnalysis, setActiveAnalysis] = useState<ChatResponsePayload | null>(null);

  // Initialize session ID on mount
  useEffect(() => {
    const storedSession = typeof window !== 'undefined' ? localStorage.getItem('hiver_chat_session') : null;
    const initialSession = storedSession || Math.random().toString(36).substring(2, 10);
    setSessionId(initialSession);
    if (typeof window !== 'undefined') {
      localStorage.setItem('hiver_chat_session', initialSession);
    }

    // Check backend health
    checkBackendHealth().then((h) => setHealth(h));
  }, []);

  const sendMessage = useCallback(
    async (text: string, manualLang?: string) => {
      const trimmed = text.trim();
      if (!trimmed || loading) return;

      const userMsgId = 'msg-' + Date.now();
      const userMessage: ChatMessage = {
        id: userMsgId,
        sender: 'user',
        text: trimmed,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages((prev) => [...prev, userMessage]);
      setLoading(true);
      setError(null);

      try {
        const responseData = await sendChatMessage(trimmed, sessionId, manualLang);

        // Update active session ID if server returned one
        if (responseData.session_id && responseData.session_id !== sessionId) {
          setSessionId(responseData.session_id);
          if (typeof window !== 'undefined') {
            localStorage.setItem('hiver_chat_session', responseData.session_id);
          }
        }

        // Store active analysis telemetry for side inspection panel
        setActiveAnalysis(responseData);

        const agentMessage: ChatMessage = {
          id: 'agent-' + Date.now(),
          sender: 'agent',
          text: responseData.reply,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          intent: responseData.intent,
          confidence: responseData.confidence,
          retrieved_cases: responseData.retrieved_cases,
          escalate: responseData.escalate,
          escalation_reason: responseData.escalation_reason,
          language: responseData.language,
          request_id: responseData.request_id,
          telemetry: responseData.telemetry,
        };

        setMessages((prev) => [...prev, agentMessage]);
      } catch (err: any) {
        const errorMsg = err.message || 'Unable to connect to AI Support Service';
        setError(errorMsg);

        const systemErrMsg: ChatMessage = {
          id: 'err-' + Date.now(),
          sender: 'system',
          text: `⚠️ ${errorMsg}`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          error: true,
        };

        setMessages((prev) => [...prev, systemErrMsg]);
      } finally {
        setLoading(false);
      }
    },
    [loading, sessionId]
  );

  const retryLastMessage = useCallback(() => {
    // Find last user message
    const lastUserMessage = [...messages].reverse().find((m) => m.sender === 'user');
    if (lastUserMessage) {
      sendMessage(lastUserMessage.text);
    }
  }, [messages, sendMessage]);

  const clearChat = useCallback(() => {
    const newSession = Math.random().toString(36).substring(2, 10);
    setSessionId(newSession);
    if (typeof window !== 'undefined') {
      localStorage.setItem('hiver_chat_session', newSession);
    }
    setMessages([
      {
        id: 'init-msg-1',
        sender: 'agent',
        text: 'Hello! I am your Amazon Support Assistant. How can I assist you with your order, delivery, return, or account today?',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      },
    ]);
    setActiveAnalysis(null);
    setError(null);
  }, []);

  return {
    messages,
    loading,
    error,
    sessionId,
    health,
    activeAnalysis,
    sendMessage,
    retryLastMessage,
    clearChat,
  };
}
