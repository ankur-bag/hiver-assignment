'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import {
  ChatMessage,
  ChatResponsePayload,
  ServiceHealth,
  StreamMetadataPayload,
  StreamCompletePayload,
} from '../types/chat';
import { streamChatMessage, checkBackendHealth } from '../services/api';

export type StreamingStatus = 'idle' | 'connecting' | 'streaming' | 'completed' | 'error' | 'aborted';

const getFormattedTime = () => {
  if (typeof window === 'undefined') return 'Just now';
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

/**
 * Custom React Hook for Chat State Management.
 * Encapsulates real HTTP token streaming, AbortController cancellation,
 * progressive message building, telemetry inspection, and backend health status.
 */
export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'init-msg-1',
      sender: 'agent',
      text: 'Hello! I am your Amazon Support Assistant. How can I assist you with your order, delivery, return, or account today?',
      timestamp: 'Just now',
    },
  ]);
  const [loading, setLoading] = useState<boolean>(false);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [streamingStatus, setStreamingStatus] = useState<StreamingStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string>('');
  const [health, setHealth] = useState<ServiceHealth>({ status: 'healthy' });
  const [activeAnalysis, setActiveAnalysis] = useState<ChatResponsePayload | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);

  // Initialize session ID on mount & run health polling
  useEffect(() => {
    const storedSession = typeof window !== 'undefined' ? localStorage.getItem('hiver_chat_session') : null;
    const initialSession = storedSession || Math.random().toString(36).substring(2, 10);
    setSessionId(initialSession);
    if (typeof window !== 'undefined') {
      localStorage.setItem('hiver_chat_session', initialSession);
    }

    const pollHealth = () => {
      checkBackendHealth().then((h) => {
        if (h && h.status) {
          setHealth(h);
        }
      });
    };

    // Immediate check
    pollHealth();

    // Periodic check every 10 seconds
    const interval = setInterval(pollHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const stopGeneration = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setLoading(false);
    setIsStreaming(false);
    setStreamingStatus('aborted');

    // Mark any active streaming message as non-streaming
    setMessages((prev) =>
      prev.map((msg) => (msg.isStreaming ? { ...msg, isStreaming: false } : msg))
    );
  }, []);

  const sendMessage = useCallback(
    async (text: string, manualLang?: string) => {
      const trimmed = text.trim();
      if (!trimmed || loading) return;

      // Abort any existing stream
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      const controller = new AbortController();
      abortControllerRef.current = controller;

      const currentTime = getFormattedTime();
      const userMsgId = 'msg-' + Date.now();
      const userMessage: ChatMessage = {
        id: userMsgId,
        sender: 'user',
        text: trimmed,
        timestamp: currentTime,
      };

      const agentMsgId = 'agent-' + (Date.now() + 1);
      const placeholderAgentMessage: ChatMessage = {
        id: agentMsgId,
        sender: 'agent',
        text: '',
        timestamp: currentTime,
        isStreaming: true,
      };

      setMessages((prev) => [...prev, userMessage, placeholderAgentMessage]);
      setLoading(true);
      setIsStreaming(true);
      setStreamingStatus('connecting');
      setError(null);

      // Pre-populate analysis panel with awaiting state
      setActiveAnalysis({
        reply: '',
        intent: 'Analyzing...',
        confidence: 0,
        retrieved_cases: 0,
        escalate: false,
        language: manualLang || 'auto',
        session_id: sessionId,
        request_id: '',
        telemetry: {},
      });

      let accumulatedText = '';

      try {
        await streamChatMessage(
          trimmed,
          sessionId,
          manualLang,
          {
            onMetadata: (metadata: StreamMetadataPayload) => {
              setStreamingStatus('streaming');
              setHealth({ status: 'healthy' });

              // Immediately populate AI Analysis panel with classified intent & retrieval count
              setActiveAnalysis((prev) => ({
                reply: accumulatedText,
                intent: metadata.intent,
                confidence: metadata.confidence,
                retrieved_cases: metadata.retrieved_cases,
                escalate: prev?.escalate || false,
                language: metadata.language,
                session_id: metadata.session_id || sessionId,
                request_id: metadata.request_id,
                telemetry: { request_id: metadata.request_id },
              }));

              if (metadata.session_id && metadata.session_id !== sessionId) {
                setSessionId(metadata.session_id);
                if (typeof window !== 'undefined') {
                  localStorage.setItem('hiver_chat_session', metadata.session_id);
                }
              }

              // Update agent placeholder with initial metadata
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === agentMsgId
                    ? {
                        ...msg,
                        intent: metadata.intent,
                        confidence: metadata.confidence,
                        retrieved_cases: metadata.retrieved_cases,
                        language: metadata.language,
                        request_id: metadata.request_id,
                      }
                    : msg
                )
              );
            },
            onToken: (tokenText: string) => {
              accumulatedText += tokenText;
              setStreamingStatus('streaming');

              // Efficient token chunk appending
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === agentMsgId
                    ? { ...msg, text: msg.text + tokenText }
                    : msg
                )
              );
            },
            onComplete: (completeData: StreamCompletePayload) => {
              setStreamingStatus('completed');
              setIsStreaming(false);
              setHealth({ status: 'healthy' });

              // Update final authoritative message details
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === agentMsgId
                    ? {
                        ...msg,
                        isStreaming: false,
                        intent: completeData.intent,
                        confidence: completeData.confidence,
                        retrieved_cases: completeData.retrieved_cases,
                        escalate: completeData.escalate,
                        escalation_reason: completeData.escalation_reason,
                        language: completeData.language,
                        telemetry: completeData.telemetry,
                      }
                    : msg
                )
              );

              // Finalize side inspection telemetry
              setActiveAnalysis({
                reply: accumulatedText,
                intent: completeData.intent,
                confidence: completeData.confidence,
                retrieved_cases: completeData.retrieved_cases,
                escalate: completeData.escalate,
                escalation_reason: completeData.escalation_reason,
                language: completeData.language,
                session_id: completeData.session_id || sessionId,
                request_id: completeData.telemetry?.request_id || '',
                telemetry: completeData.telemetry,
              });
            },
            onError: (err: Error) => {
              setStreamingStatus('error');
              setIsStreaming(false);
              const errorMsg = err.message || 'Support service temporarily unavailable';
              setError(errorMsg);

              // If assistant message was empty, remove it and add system error
              setMessages((prev) => {
                const target = prev.find((m) => m.id === agentMsgId);
                if (target && !target.text.trim()) {
                  return [
                    ...prev.filter((m) => m.id !== agentMsgId),
                    {
                      id: 'err-' + Date.now(),
                      sender: 'system',
                      text: errorMsg,
                      timestamp: getFormattedTime(),
                      error: true,
                    },
                  ];
                }
                return prev.map((m) => (m.id === agentMsgId ? { ...m, isStreaming: false } : m));
              });
            },
          },
          controller.signal
        );
      } catch (err: any) {
        if (err.name === 'AbortError') {
          setStreamingStatus('aborted');
        } else {
          setStreamingStatus('error');
          const errorMsg = err.message || 'Unable to connect to AI Support Service';
          setError(errorMsg);
        }
      } finally {
        setLoading(false);
        setIsStreaming(false);
        abortControllerRef.current = null;
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
    stopGeneration();
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
        timestamp: 'Just now',
      },
    ]);
    setActiveAnalysis(null);
    setError(null);
    setStreamingStatus('idle');
  }, [stopGeneration]);

  return {
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
  };
}
