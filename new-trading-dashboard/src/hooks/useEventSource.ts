import { useEffect, useRef } from 'react';

/**
 * Hook for Server-Sent Events (SSE) connections
 * Provides real-time updates for decisions and other streaming data
 */
export function useEventSource<T = any>(
  url: string,
  onMessage: (event: T) => void,
  options: {
    onOpen?: () => void;
    onError?: (error: Event) => void;
    onClose?: () => void;
    reconnectInterval?: number;
    enabled?: boolean;
  } = {}
) {
  const {
    onOpen,
    onError,
    onClose,
    reconnectInterval = 5000,
    enabled = true
  } = options;

  const esRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (!enabled) return;

    const connect = () => {
      if (esRef.current) {
        esRef.current.close();
      }

      const es = new EventSource(url);
      esRef.current = es;

      es.onopen = () => {
        console.log(`SSE connected to ${url}`);
        onOpen?.();
      };

      es.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onMessage(data);
        } catch (error) {
          console.warn('Failed to parse SSE message:', event.data, error);
        }
      };

      es.onerror = (error) => {
        console.error(`SSE error for ${url}:`, error);
        onError?.(error);

        // Auto-reconnect after delay
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
        }

        reconnectTimeoutRef.current = setTimeout(() => {
          console.log(`Reconnecting SSE to ${url}...`);
          connect();
        }, reconnectInterval);
      };
    };

    connect();

    return () => {
      if (esRef.current) {
        esRef.current.close();
        esRef.current = null;
      }

      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }

      onClose?.();
    };
  }, [url, enabled]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (esRef.current) {
        esRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, []);
}

/**
 * Hook specifically for decisions SSE updates
 */
export function useDecisionsStream(
  onDecision: (decision: {
    ts: number;
    symbol: string;
    stage: 'proposed' | 'intent' | 'executed';
    strategy_id: string;
    confidence: number;
    costs?: {
      spread_bps: number;
      fees_per_contract: number;
      slippage_bps: number;
    };
  }) => void
) {
  useEventSource(
    '/api/decisions/stream',
    (data) => {
      if (data.type === 'connected') {
        console.log('Connected to decisions stream');
        return;
      }

      // Handle different decision stages
      if (['proposed', 'intent', 'executed'].includes(data.stage)) {
        onDecision(data);
      }
    },
    {
      onError: (error) => {
        console.warn('Decisions stream error:', error);
      },
      enabled: true
    }
  );
}
