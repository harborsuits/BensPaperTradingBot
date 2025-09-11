import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react';
import { useAuth } from '@/context/AuthContext';
import { wsUrl } from '@/lib/wsUrl';
import { showErrorToast, showInfoToast } from '../utils/toast';
import { useQueryClient } from '@tanstack/react-query';

// Define types for different WebSocket message types
export type WebSocketMessageType =
  | 'market_data'
  | 'trade_executed'
  | 'strategy_update'
  | 'context_update'
  | 'log'
  | 'alert'
  | 'cycle_decision'
  | 'evo_progress'
  | 'evo_complete'
  | 'position_update';

export interface WebSocketMessage {
  type: WebSocketMessageType;
  data: any;
  timestamp: string;
}

interface WebSocketContextType {
  isConnected: boolean;
  lastMessage: WebSocketMessage | null;
  sendMessage: (message: any) => void;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
  reconnect: () => void;
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

export const useWebSocket = (): WebSocketContextType => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};

export { useWebSocket as useWebSocketContext };

export const WebSocketProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAuthenticated } = useAuth();
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef<number>(0);
  const heartbeatIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const pingTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const queryClient = useQueryClient();

  // Initialize WebSocket connection
  const connect = useCallback(() => {
    // Always connect: WS layer is public for heartbeat/telemetry and doesn't require auth

    try {
      // Guard against redundant connections
      if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
        return wsRef.current;
      }
      
      setConnectionStatus('connecting');
      
      // Build WS URL using centralized helper (respects VITE_WS_BASE_URL or proxy)
      const token = localStorage.getItem('auth_token');
      const url = wsUrl('/ws') + (token ? `?token=${encodeURIComponent(token)}` : '');
      
      console.log('WebSocket connecting to:', url);
      wsRef.current = new WebSocket(url);
      
      wsRef.current.onopen = () => {
        setIsConnected(true);
        setConnectionStatus('connected');
        reconnectAttemptsRef.current = 0;
        showInfoToast('WebSocket connected');
        
        // Setup heartbeat to keep connection alive
        startHeartbeat();
      };
      
      wsRef.current.onclose = (event) => {
        setIsConnected(false);
        setConnectionStatus('disconnected');
        
        // Stop heartbeat
        stopHeartbeat();
        
        // Don't attempt to reconnect if it was a clean close (code 1000)
        if (event.code !== 1000) {
          scheduleReconnect();
        }
      };
      
      wsRef.current.onerror = (_error) => {
        // In dev, StrictMode can cause transient socket errors during double-mount; keep logs quiet
        setConnectionStatus('error');
      };
      
      wsRef.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          setLastMessage(message);
          handleWebSocketMessage(message);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
    } catch (error) {
      console.error('Error establishing WebSocket connection:', error);
      setConnectionStatus('error');
      scheduleReconnect();
    }
  }, []);

  // Schedule reconnection with exponential backoff and jitter
  const scheduleReconnect = useCallback(() => {
    // Don't reconnect if tab is hidden - wait for visibility change
    if (document.hidden) {
      console.log('Tab hidden - pausing reconnection until visible');
      return;
    }

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }

    // More conservative backoff: 100ms base, max 5s, with jitter
    const BASE_DELAY = 100; // 100ms base (not 1s)
    const MAX_DELAY = 5000; // 5s max (not 30s)
    const JITTER_FACTOR = 0.3; // ±30% jitter

    // Exponential backoff with cap
    const exponentialDelay = Math.min(
      BASE_DELAY * Math.pow(1.5, Math.min(reconnectAttemptsRef.current, 10)), // Cap exponent
      MAX_DELAY
    );

    // Add jitter to prevent thundering herd
    const jitter = exponentialDelay * JITTER_FACTOR * (Math.random() * 2 - 1);
    const reconnectDelay = Math.max(100, exponentialDelay + jitter); // Min 100ms

    reconnectAttemptsRef.current += 1;

    // Only log first few attempts to reduce console spam
    if (reconnectAttemptsRef.current <= 3) {
      console.log(`WebSocket reconnection scheduled in ${(reconnectDelay / 1000).toFixed(1)}s (attempt ${reconnectAttemptsRef.current})`);
    } else if (reconnectAttemptsRef.current % 5 === 0) {
      console.log(`WebSocket reconnection attempt ${reconnectAttemptsRef.current} in ${(reconnectDelay / 1000).toFixed(1)}s`);
    }

    reconnectTimeoutRef.current = setTimeout(() => {
      connect();
    }, reconnectDelay);
  }, [connect]);

  // Handle different types of WebSocket messages
  const handleWebSocketMessage = useCallback((message: any) => {
    // Reset ping timeout if we received any message (proves connection is alive)
    if (pingTimeoutRef.current) {
      clearTimeout(pingTimeoutRef.current);
      pingTimeoutRef.current = null;
    }

    // Handle pong responses explicitly
    if (message.type === 'pong') {
      // Connection is confirmed alive
      reconnectAttemptsRef.current = 0; // Reset reconnection attempts on successful pong
      return; // Don't process further
    }
    switch (message.type) {
      case 'market_data':
        // Market data updates (e.g., price changes)
        // Invalidate queries that depend on market data
        queryClient.invalidateQueries(['market', 'prices']);
        break;
        
      case 'trade_executed':
        // Trade execution event → refresh portfolio/trades and targeted mode if present
        {
          const mode = (message as any)?.data?.account || (message as any)?.data?.mode || (message as any)?.data?.portfolio?.mode;
          if (mode === 'paper' || mode === 'live') {
            queryClient.invalidateQueries(['portfolio', mode]);
          }
          queryClient.invalidateQueries(['portfolio']);
          queryClient.invalidateQueries(['trades']);
        }
        
        // Show a toast notification for trade executions
        showInfoToast(
          `${message.data.action} ${message.data.quantity} ${message.data.symbol} @ ${message.data.price}`
        );
        break;
        
      case 'strategy_update':
        // Strategy status or priority changed
        queryClient.invalidateQueries(['strategies']);
        queryClient.invalidateQueries(['strategies', 'all']);
        break;
        
      case 'context_update':
        // Market context updates (regime, sentiment, etc.)
        queryClient.invalidateQueries(['context']);
        break;
        
      case 'log':
        // System log entry → append to common log keys
        {
          const payload: any = (message as any).data;
          const updateKey = (level: string) => {
            queryClient.setQueryData(['logs', level], (prev: any) => {
              const arr = Array.isArray(prev) ? prev : (Array.isArray(prev?.items) ? prev.items : []);
              const next = [payload, ...arr].slice(0, 100);
              return Array.isArray(prev) ? next : { ...(prev || {}), items: next };
            });
          };
          updateKey('ALL');
          if (payload?.level) updateKey(String(payload.level).toUpperCase()); else updateKey('INFO');
        }
        break;
        
      case 'alert': {
        const a = (message as any).payload || (message as any).data || message;
        if (a?.message) showErrorToast(a.message);
        break;
      }
        
      case 'cycle_decision':
        // Trade decision cycle completed → update recent list and invalidate broader keys
        {
          const payload = (message as any).data;
          queryClient.setQueryData(['decisions', 'recent', 50], (prev: any) => {
            const items = Array.isArray(prev?.items) ? prev.items : (Array.isArray(prev) ? prev : []);
            const incoming = Array.isArray(payload) ? payload : [payload];
            const next = [...incoming, ...items].slice(0, 50);
            return prev?.items ? { ...prev, items: next } : next;
          });
          queryClient.invalidateQueries(['decisions']);
        }
        break;
        
      case 'evo_progress':
      case 'evo_complete':
        // EvoTester updates - handled by EvoContext or component state
        break;
        
      case 'position_update':
        // Position update (e.g., P&L change due to price movement)
        {
          const mode = (message as any)?.data?.account || (message as any)?.data?.mode || (message as any)?.data?.portfolio?.mode;
          if (mode === 'paper' || mode === 'live') {
            queryClient.invalidateQueries(['portfolio', mode]);
          }
          queryClient.invalidateQueries(['portfolio']);
        }
        break;
        
      default:
        console.log('Unhandled WebSocket message type:', message.type);
    }
  }, [queryClient]);

  // Send a message through the WebSocket
  const sendMessage = useCallback((message: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
      return true;
    }
    return false;
  }, []);

  // Heartbeat to keep connection alive and detect disconnections
  const heartbeatMsActive = 15000; // 15 seconds when tab is active (not too frequent)
  const heartbeatMsHidden = 60000; // 60 seconds when tab is hidden (conservative)
  const PING_TIMEOUT = 10000; // 10 seconds to wait for pong

  // Send ping message over WebSocket
  const sendPing = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      try {
        // Send ping with timestamp for debugging
        const pingData = { type: 'ping', ts: Date.now() };
        wsRef.current.send(JSON.stringify(pingData));

        // Set timeout for pong response
        pingTimeoutRef.current = setTimeout(() => {
          console.warn('WebSocket ping timeout - connection may be dead');
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.close(1001, 'ping timeout'); // Going away
          }
        }, PING_TIMEOUT);
      } catch (error) {
        console.error('Error sending ping:', error);
        // Force reconnection on ping send error
        if (wsRef.current) {
          wsRef.current.close(1001, 'ping send error');
        }
      }
    }
  }, []);
  
  // Start heartbeat with appropriate interval
  const startHeartbeat = useCallback(() => {
    stopHeartbeat(); // Clear any existing timers
    
    // Set interval based on tab visibility
    const interval = document.hidden ? heartbeatMsHidden : heartbeatMsActive;
    heartbeatIntervalRef.current = setInterval(sendPing, interval);
  }, [sendPing]);
  
  // Handle visibility change - resume reconnection and adjust heartbeat
  useEffect(() => {
    const handleVisibilityChange = () => {
      const isHidden = document.hidden;

      if (isHidden) {
        // Tab hidden - pause heartbeat and reconnection
        stopHeartbeat();
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
          reconnectTimeoutRef.current = null;
          console.log('Tab hidden - paused reconnection');
        }
      } else {
        // Tab visible - resume heartbeat and try reconnection if needed
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          // Connection is good, just restart heartbeat
          heartbeatIntervalRef.current = setInterval(sendPing, heartbeatMsActive);
        } else if (wsRef.current?.readyState === WebSocket.CLOSED) {
          // Connection is closed, try to reconnect
          console.log('Tab visible - resuming reconnection');
          reconnectAttemptsRef.current = 0; // Reset attempts for fresh start
          scheduleReconnect();
        }
      }
    };

    // Add event listener
    document.addEventListener('visibilitychange', handleVisibilityChange);

    // Clean up
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [sendPing, scheduleReconnect]);
  
  // Stop heartbeat timers
  const stopHeartbeat = useCallback(() => {
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
      heartbeatIntervalRef.current = null;
    }
    
    if (pingTimeoutRef.current) {
      clearTimeout(pingTimeoutRef.current);
      pingTimeoutRef.current = null;
    }
  }, []);

  // Manual reconnection function
  const reconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    stopHeartbeat();
    connect();
  }, [connect, stopHeartbeat]);

  // Complete cleanup function for all resources
  const cleanup = useCallback(() => {
    stopHeartbeat();
    
    const ws = wsRef.current;
    wsRef.current = null;
    try {
      if (ws) {
        if (ws.readyState === WebSocket.OPEN) {
          ws.close(1000);
        } else {
          // Avoid noisy errors when closing sockets that haven't opened yet (StrictMode mount/unmount)
          try { ws.onopen = null as any; ws.onclose = null as any; ws.onerror = null as any; } catch {}
          try { ws.close(); } catch {}
        }
      }
    } catch {}
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
  }, [stopHeartbeat]);

  // Connect when authenticated — defer slightly in dev to avoid React StrictMode double-connect warning
  useEffect(() => {
    const delayMs = (import.meta as any).env?.DEV ? 150 : 0;
    let timer: any = setTimeout(() => {
      connect();
    }, delayMs);
    return () => { try { clearTimeout(timer); } catch {}; cleanup(); };
  }, [connect, cleanup]);

  // Expose debug helpers in dev mode
  if (import.meta.env.DEV) {
    try {
      // Message injector for testing
      (window as any).__injectWsMessage = (m: any) => handleWebSocketMessage(m);
      
      // WebSocket debug info
      (window as any)._wsDebug = () => console.log({ 
        readyState: wsRef.current?.readyState, 
        connected: isConnected,
        heartbeat: !!heartbeatIntervalRef.current,
        pingTimeout: !!pingTimeoutRef.current,
        reconnectAttempts: reconnectAttemptsRef.current
      });
      
      // HMR cleanup
      if (import.meta.hot) {
        import.meta.hot.dispose(() => {
          // Clean up WebSocket on HMR
          console.log('[HMR] Cleaning up WebSocket connection');
          try { 
            if (wsRef.current) {
              wsRef.current.close(1000, 'HMR dispose'); 
            }
          } catch (e) {
            console.error('[HMR] WebSocket cleanup error:', e);
          }
          
          // Clean up timers
          if (heartbeatIntervalRef.current) {
            clearInterval(heartbeatIntervalRef.current);
            heartbeatIntervalRef.current = null;
          }
          
          if (pingTimeoutRef.current) {
            clearTimeout(pingTimeoutRef.current);
            pingTimeoutRef.current = null;
          }
          
          if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
          }
        });
      }
    } catch {}
  }

  return (
    <WebSocketContext.Provider
      value={{
        isConnected,
        lastMessage,
        sendMessage,
        connectionStatus,
        reconnect
      }}
    >
      {children}
    </WebSocketContext.Provider>
  );
};

export default WebSocketContext;
