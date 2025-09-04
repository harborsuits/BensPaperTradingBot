// src/hooks/useDecisionFeed.ts
import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { DecisionTrace, validateTrace } from '@/types/DecisionTrace';

interface UseDecisionFeedOptions {
  basePath?: string;
  initialFetchLimit?: number;
}

type FeedStatus = 'idle' | 'connecting' | 'live' | 'degraded';

export function useDecisionFeed({
  basePath = '',
  initialFetchLimit = 50,
}: UseDecisionFeedOptions = {}) {
  const [status, setStatus] = useState<FeedStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [decisions, setDecisions] = useState<DecisionTrace[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const backoffRef = useRef<number>(500);
  
  // Add a trace to the decisions list, deduplicating by trace_id
  const addTrace = useCallback((obj: unknown) => {
    const result = validateTrace(obj);
    if (!result.ok) {
      setError(result.error);
      return;
    }
    
    const trace = result.value;
    setDecisions(prev => {
      // Replace existing trace with same ID or add new one
      const exists = prev.findIndex(d => d.trace_id === trace.trace_id);
      if (exists >= 0) {
        const updated = [...prev];
        updated[exists] = trace;
        return updated;
      } else {
        return [trace, ...prev].sort(
          (a, b) => new Date(b.as_of).getTime() - new Date(a.as_of).getTime()
        );
      }
    });
  }, []);

  // Initial fetch fallback
  useEffect(() => {
    (async () => {
      try {
        // Try the FastAPI endpoint first (port 8000)
        let res = await fetch(`${basePath || 'http://localhost:8000'}/api/decision-traces?limit=${initialFetchLimit}`);
        
        // Fallback to the original path if that fails
        if (!res.ok && basePath === '') {
          res = await fetch(`/api/decision-traces?limit=${initialFetchLimit}`);
        }
        
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const arr = await res.json();
        if (Array.isArray(arr)) arr.forEach(addTrace);
      } catch (e: any) {
        setError(`Initial fetch failed: ${e.message}`);
        setStatus('degraded');
      }
    })();
  }, [addTrace, basePath, initialFetchLimit]);

  // WebSocket live stream
  useEffect(() => {
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${proto}://${window.location.host}/ws/decisions`;
    setStatus('connecting');
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus('live');
      backoffRef.current = 500;
    };
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (Array.isArray(msg)) msg.forEach(addTrace);
        else addTrace(msg);
      } catch (e: any) {
        setError(`Bad WS message: ${e.message}`);
        setStatus('degraded');
      }
    };
    ws.onerror = () => setStatus('degraded');
    ws.onclose = () => {
      setStatus('degraded');
      // Reconnect with backoff
      const to = setTimeout(() => {
        backoffRef.current = Math.min(backoffRef.current * 1.5, 8000);
        // trigger new effect by changing a key (hack: setStatus to idle → connecting)
        setStatus('idle');
      }, backoffRef.current);
      return () => clearTimeout(to);
    };

    return () => {
      ws.close();
    };
  }, [addTrace]);

  const invalid = useMemo(() => (error ? 1 : 0), [error]);

  return { status, error, decisions, invalid } as const;
}
