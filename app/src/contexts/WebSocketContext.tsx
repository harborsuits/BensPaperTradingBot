import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react';
import { useAuth } from '@/context/AuthContext';
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
    if (!isAuthenticated) {
      return;
    }

    try {
      // Guard against redundant connections
      if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
        return wsRef.current;
      }
      
      setConnectionStatus('connecting');
      
      // Build WS URL; token optional - use Vite proxy
      const token = localStorage.getItem('auth_token');
      const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = location.host; // This will be localhost:3003 in dev
      const wsUrl = `${protocol}//${host}/ws${token ? `?token=${encodeURIComponent(token)}` : ''}`;
      
      console.log('WebSocket connecting to:', wsUrl);
      wsRef.current = new WebSocket(wsUrl);
      
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
      
      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('error');
        // Don't show error toast for expected errors (WebSocket not implemented yet)
        console.log('WebSocket not available - using HTTP polling instead');
      };
      
      wsRef.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as WebSocketMessage;
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
  }, [isAuthenticated]);

  // Schedule reconnection with exponential backoff
  const scheduleReconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    
    const MAX_RECONNECT_DELAY = 30000; // 30 seconds
    const MIN_RECONNECT_DELAY = 1000; // 1 second
    const reconnectDelay = Math.min(
      MIN_RECONNECT_DELAY * Math.pow(1.5, reconnectAttemptsRef.current),
      MAX_RECONNECT_DELAY
    );
    
    reconnectAttemptsRef.current += 1;
    
    console.log(`Scheduling WebSocket reconnection in ${reconnectDelay}ms (attempt ${reconnectAttemptsRef.current})`);
    
    reconnectTimeoutRef.current = setTimeout(() => {
      connect();
    }, reconnectDelay);
  }, [connect]);

  // Handle different types of WebSocket messages
  const handleWebSocketMessage = useCallback((message: WebSocketMessage) => {
    // Reset ping timeout if we received any message
    if (pingTimeoutRef.current) {
      clearTimeout(pingTimeoutRef.current);
      pingTimeoutRef.current = null;
    }
    
    // Handle pong responses
    if (message.type === 'pong') {
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
        
      case 'alert':
        // System alert - potentially show as a toast
        showErrorToast(message.data.message);
        break;
        
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
  const heartbeatMsActive = 10000; // 10 seconds when tab is active
  const heartbeatMsHidden = 30000; // 30 seconds when tab is hidden
  const PING_TIMEOUT = 10000; // 10 seconds
  
  // Send ping message over WebSocket
  const sendPing = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      // Send ping
      wsRef.current.send(JSON.stringify({ type: 'ping' }));
      
      // Set timeout for pong response
      pingTimeoutRef.current = setTimeout(() => {
        console.log('Ping timeout - connection may be dead');
        if (wsRef.current) {
          wsRef.current.close(1001); // Going away
          // This will trigger the onclose handler which will schedule reconnect
        }
      }, PING_TIMEOUT);
    }
  }, []);
  
  // Start heartbeat with appropriate interval
  const startHeartbeat = useCallback(() => {
    stopHeartbeat(); // Clear any existing timers
    
    // Set interval based on tab visibility
    const interval = document.hidden ? heartbeatMsHidden : heartbeatMsActive;
    heartbeatIntervalRef.current = setInterval(sendPing, interval);
  }, [sendPing]);
  
  // Handle visibility change
  useEffect(() => {
    const handleVisibilityChange = () => {
      // Re-arm heartbeat at appropriate cadence based on visibility
      if (!wsRef.current) return;
      
      stopHeartbeat();
      heartbeatIntervalRef.current = setInterval(
        sendPing, 
        document.hidden ? heartbeatMsHidden : heartbeatMsActive
      );
    };
    
    // Add event listener
    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    // Clean up
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [sendPing]);
  
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
    
    if (wsRef.current) {
      wsRef.current.close(1000); // Clean close
      wsRef.current = null;
    }
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
  }, [stopHeartbeat]);

  // Connect when authenticated
  useEffect(() => {
    // Guard against redundant connections
    if (isAuthenticated) {
      connect();
    }
    
    // Return cleanup function
    return cleanup;
  }, [isAuthenticated, connect, cleanup]);

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
