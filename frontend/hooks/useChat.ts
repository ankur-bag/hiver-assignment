'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import {
  ChatMessage,
  ProgressiveAnalysisState,
  ServiceHealth,
  StreamMetadataPayload,
  StreamCompletePayload,
} from '../types/chat';
import {
  streamChatMessage,
  checkBackendHealth,
} from '../services/api';

export type StreamingStatus = 'idle' | 'connecting' | 'streaming' | 'completed' | 'error' | 'aborted';

export function formatIntentName(rawIntent?: string | null): string {
  if (!rawIntent) return '—';
  const clean = rawIntent.trim().toUpperCase();
  const mapping: Record<string, string> = {
    PACKAGE_NOT_RECEIVED: 'Package not received',
    REFUND_PENDING: 'Refund pending',
    PRODUCT_ISSUE: 'Product issue',
    ACCOUNT_ACCESS: 'Account access',
    ACCOUNT_SUPPORT: 'Account support',
    DELIVERY_DELAY: 'Delivery delay',
    ORDER_STATUS: 'Order status',
    CUSTOMER_SERVICE_CONTACT: 'Customer care',
  };
  if (mapping[clean]) return mapping[clean];
  return clean.toLowerCase().replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

const getFormattedTime = () => {
  if (typeof window === 'undefined') return 'Just now';
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

const INITIAL_ANALYSIS_STATE: ProgressiveAnalysisState = {
  intent: null,
  intentStatus: 'idle',
  retrievedContext: null,
  retrievedCases: null,
  retrievedStatus: 'idle',
  escalate: null,
  escalationReason: null,
  escalationStatus: 'idle',
  language: 'en',
  requestId: '',
};

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
  const [analysisState, setAnalysisState] = useState<ProgressiveAnalysisState>(INITIAL_ANALYSIS_STATE);

  const abortControllerRef = useRef<AbortController | null>(null);

  // Initialize in-memory session ID and periodic health check
  useEffect(() => {
    setSessionId(Math.random().toString(36).substring(2, 10));

    const pollHealth = () => {
      checkBackendHealth().then((h) => {
        if (h && h.status) {
          setHealth(h);
        }
      });
    };

    pollHealth();
    const interval = setInterval(pollHealth, 15000);
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

    setMessages((prev) =>
      prev.map((msg) => (msg.isStreaming ? { ...msg, isStreaming: false } : msg))
    );
  }, []);

  const clearChat = useCallback(() => {
    stopGeneration();
    setSessionId(Math.random().toString(36).substring(2, 10));
    setMessages([
      {
        id: 'init-msg-1',
        sender: 'agent',
        text: 'Hello! I am your Amazon Support Assistant. How can I assist you with your order, delivery, return, or account today?',
        timestamp: 'Just now',
      },
    ]);
    setAnalysisState(INITIAL_ANALYSIS_STATE);
    setError(null);
    setStreamingStatus('idle');
  }, [stopGeneration]);

  const sendMessage = useCallback(
    async (text: string, manualLang?: string) => {
      const trimmed = text.trim();
      if (!trimmed || loading) return;

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
        stage: 'analyzing',
      };

      setMessages((prev) => [...prev, userMessage, placeholderAgentMessage]);
      setLoading(true);
      setIsStreaming(true);
      setStreamingStatus('connecting');
      setError(null);

      // Immediately set Analysis Panel to active progressive loading states (never "-" or "Unavailable")
      setAnalysisState({
        intent: 'Analyzing...',
        intentStatus: 'analyzing',
        retrievedContext: 'Searching support history...',
        retrievedCases: null,
        retrievedStatus: 'searching',
        escalate: null,
        escalationReason: null,
        escalationStatus: 'checking',
        language: manualLang || 'auto',
        requestId: '',
      });

      let accumulatedText = '';

      const handleFailure = (err: any) => {
        setStreamingStatus('error');
        setIsStreaming(false);
        setLoading(false);

        let rawError = err?.message || 'Support AI is temporarily unavailable. Please try again.';
        const isQuota =
          err?.code === 'PROVIDER_QUOTA_EXHAUSTED' ||
          err?.code === 'EMBEDDING_DAILY_QUOTA_EXHAUSTED' ||
          rawError.toLowerCase().includes('quota') ||
          rawError.includes('RESOURCE_EXHAUSTED');

        let errorMsg = 'Support AI is temporarily unavailable. Please try again.';
        if (isQuota) {
          errorMsg = 'Support AI daily quota reached. Please try again later.';
        }

        setError(errorMsg);

        // On failure: set all inspection fields to Unavailable and stop loading dots
        setAnalysisState({
          intent: 'Unavailable',
          intentStatus: 'unavailable',
          retrievedContext: 'Unavailable',
          retrievedCases: null,
          retrievedStatus: 'unavailable',
          escalate: null,
          escalationReason: null,
          escalationStatus: 'unavailable',
          language: manualLang || 'auto',
          requestId: '',
        });

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
            onStatus: (stage: string) => {
              setStreamingStatus('streaming');
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === agentMsgId ? { ...msg, stage } : msg
                )
              );
            },
            onMetadata: (metadata: StreamMetadataPayload) => {
              setStreamingStatus('streaming');
              if (metadata.intent && metadata.intent !== 'Analyzing...') {
                setAnalysisState((prev) => ({
                  ...prev,
                  intent: formatIntentName(metadata.intent),
                  intentStatus: 'resolved',
                  language: metadata.language || prev.language,
                  requestId: metadata.request_id || prev.requestId,
                }));
              }
            },
            onGrounding: (groundingData) => {
              const count = groundingData.retrieved_context_count ?? 0;
              const text = count === 1 ? '1 source' : count > 1 ? `${count} sources` : 'No relevant history found';
              setAnalysisState((prev) => ({
                ...prev,
                retrievedContext: text,
                retrievedCases: count,
                retrievedStatus: 'resolved',
              }));
            },
            onEscalation: (escData) => {
              setAnalysisState((prev) => ({
                ...prev,
                escalate: escData.escalate,
                escalationReason: escData.reason || null,
                escalationStatus: 'resolved',
              }));
            },
            onToken: (tokenText: string) => {
              accumulatedText += tokenText;
              setStreamingStatus('streaming');

              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === agentMsgId
                    ? { ...msg, text: msg.text + tokenText, stage: undefined }
                    : msg
                )
              );
            },
            onComplete: (completeData: StreamCompletePayload) => {
              setStreamingStatus('completed');
              setIsStreaming(false);
              setHealth({ status: 'healthy' });

              const casesCount = completeData.retrieved_cases ?? (typeof completeData.retrieved_context === 'number' ? completeData.retrieved_context : 0);
              const contextDisplay = casesCount === 1 ? '1 source' : casesCount > 1 ? `${casesCount} sources` : 'No relevant history found';

              setAnalysisState({
                intent: formatIntentName(completeData.intent),
                intentStatus: 'resolved',
                retrievedContext: contextDisplay,
                retrievedCases: casesCount,
                retrievedStatus: 'resolved',
                escalate: completeData.escalate ?? false,
                escalationReason: completeData.escalation_reason || null,
                escalationStatus: 'resolved',
                language: completeData.language || 'en',
                requestId: completeData.telemetry?.request_id || '',
              });

              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === agentMsgId
                    ? {
                        ...msg,
                        isStreaming: false,
                        stage: undefined,
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
    [loading, sessionId]
  );

  const retryLastMessage = useCallback(() => {
    const lastUserMessage = [...messages].reverse().find((m) => m.sender === 'user');
    if (lastUserMessage) {
      sendMessage(lastUserMessage.text);
    }
  }, [messages, sendMessage]);

  return {
    messages,
    loading,
    isStreaming,
    streamingStatus,
    error,
    sessionId,
    health,
    analysisState,
    sendMessage,
    stopGeneration,
    retryLastMessage,
    clearChat,
  };
}
