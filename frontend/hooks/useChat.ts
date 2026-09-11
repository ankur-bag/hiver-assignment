'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import {
  ChatMessage,
  ChatResponsePayload,
  ConversationItem,
  ServiceHealth,
  StreamMetadataPayload,
  StreamCompletePayload,
} from '../types/chat';
import {
  streamChatMessage,
  checkBackendHealth,
  listConversations,
  getConversation,
  deleteConversation,
} from '../services/api';

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
  const [conversations, setConversations] = useState<ConversationItem[]>([]);

  const abortControllerRef = useRef<AbortController | null>(null);

  const refreshConversations = useCallback(async () => {
    try {
      const list = await listConversations();
      setConversations(list);
    } catch (err) {
      console.warn('Could not refresh conversations:', err);
    }
  }, []);

  // Initialize session ID on mount & run health and conversation polling
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
    refreshConversations();

    // Periodic check every 10 seconds
    const interval = setInterval(pollHealth, 10000);
    return () => clearInterval(interval);
  }, [refreshConversations]);

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

      // Pre-populate analysis panel with awaiting state.
      setActiveAnalysis({
        reply: '',
        intent: 'Analyzing...',
        retrieved_context: null,
        retrieved_cases: null,
        escalate: null,
        language: manualLang || 'auto',
        session_id: sessionId,
        request_id: '',
        telemetry: {},
      });

      let accumulatedText = '';

      const handleFailure = (err: any) => {
        setStreamingStatus('error');
        setIsStreaming(false);
        setLoading(false);

        let rawError = err?.message || 'Support AI is temporarily busy. Please try again.';
        const isQuota =
          err?.code === 'PROVIDER_QUOTA_EXHAUSTED' ||
          err?.code === 'EMBEDDING_DAILY_QUOTA_EXHAUSTED' ||
          rawError.toLowerCase().includes('quota') ||
          rawError.includes('RESOURCE_EXHAUSTED');

        let errorMsg = 'Support AI is temporarily busy. Please try again.';
        if (isQuota) {
          errorMsg = 'Support AI daily quota reached. Please try again later.';
        }

        setError(errorMsg);

        // On failure: set Intent to Unavailable and all metrics to null (displaying '—')
        setActiveAnalysis({
          reply: '',
          intent: 'Unavailable',
          retrieved_context: null,
          retrieved_cases: null,
          escalate: null,
          escalation_reason: null,
          language: manualLang || 'auto',
          session_id: sessionId,
          request_id: '',
          telemetry: {},
        });

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
      };

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
                retrieved_context: metadata.retrieved_context,
                retrieved_cases: metadata.retrieved_cases,
                escalate: prev?.escalate ?? null,
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
                        retrieved_context: metadata.retrieved_context,
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
                        retrieved_context: completeData.retrieved_context,
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
                retrieved_context: completeData.retrieved_context,
                retrieved_cases: completeData.retrieved_cases,
                escalate: completeData.escalate,
                escalation_reason: completeData.escalation_reason,
                language: completeData.language,
                session_id: completeData.session_id || sessionId,
                request_id: completeData.telemetry?.request_id || '',
                telemetry: completeData.telemetry,
              });

              // Refresh conversation list after response is fully generated
              refreshConversations();
            },
            onError: (err: Error) => {
              handleFailure(err);
            },
          },
          controller.signal
        );
      } catch (err: any) {
        if (err.name === 'AbortError') {
          setStreamingStatus('aborted');
        } else {
          handleFailure(err);
        }
      } finally {
        setLoading(false);
        setIsStreaming(false);
        abortControllerRef.current = null;
      }
    },
    [loading, sessionId, refreshConversations]
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

  const selectConversation = useCallback(async (id: string) => {
    if (!id || id === sessionId) return;
    stopGeneration();
    try {
      const detail = await getConversation(id);
      if (detail && detail.conversation) {
        setSessionId(detail.conversation.id);
        if (typeof window !== 'undefined') {
          localStorage.setItem('hiver_chat_session', detail.conversation.id);
        }
        if (detail.messages && detail.messages.length > 0) {
          const mapped: ChatMessage[] = detail.messages.map((m: {
            id: string;
            role: 'user' | 'assistant';
            content: string;
            intent?: string | null;
            escalated?: number | boolean | null;
            created_at: string;
          }) => ({
            id: m.id,
            sender: m.role === 'assistant' ? 'agent' : 'user',
            text: m.content,
            timestamp: new Date(m.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            intent: m.intent || undefined,
            escalate: m.escalated === 1 || m.escalated === true,
          }));
          setMessages(mapped);

          // Find last assistant message to populate analysis panel
          const lastAssistant = [...detail.messages].reverse().find((m) => m.role === 'assistant');
          if (lastAssistant) {
            setActiveAnalysis({
              reply: lastAssistant.content,
              intent: lastAssistant.intent || 'CUSTOMER_SERVICE_CONTACT',
              retrieved_context: 'available',
              retrieved_cases: 1,
              escalate: lastAssistant.escalated === 1 || lastAssistant.escalated === true,
              escalation_reason: null,
              language: 'en',
              session_id: detail.conversation.id,
              request_id: '',
              telemetry: {},
            });
          } else {
            setActiveAnalysis(null);
          }
        }
        setError(null);
      }
    } catch (err) {
      console.warn('Could not select conversation:', err);
    }
  }, [sessionId, stopGeneration]);

  const deleteConv = useCallback(async (id: string) => {
    try {
      const ok = await deleteConversation(id);
      if (ok) {
        setConversations((prev) => prev.filter((c) => c.id !== id));
        if (sessionId === id) {
          clearChat();
        }
      }
    } catch (err) {
      console.warn('Could not delete conversation:', err);
    }
  }, [sessionId, clearChat]);

  return {
    messages,
    conversations,
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
    selectConversation,
    deleteConv,
    refreshConversations,
  };
}
