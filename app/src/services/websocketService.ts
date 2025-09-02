import { create } from 'zustand';
import { MarketContextData, MarketRegimeData, NewsItem, SentimentData } from './contextApi';
import { Position, PortfolioSummary, Trade } from '../types/portfolio';
import { Strategy } from '../components/strategy/StrategyRanking';
import { TradeCandidate } from '../components/strategy/TradeCandidates';

// ==========================================================
// WebSocket Message Types
// ==========================================================

// Common message structure with specific payload types
export interface WebSocketMessage<T = any> {
  event: WebSocketEventType;
  timestamp: string;
  payload: T;
}

export type WebSocketEventType = 
  // Portfolio events
  | 'portfolio_update'
  | 'position_update'
  | 'trade_executed'
  | 'account_balance_update'
  // Market context events
  | 'market_regime_change'
  | 'sentiment_update'
  | 'news_alert'
  | 'market_anomaly_detected'
  // Strategy events
  | 'strategy_priority_update'
  | 'trade_candidate_added'
  | 'trade_candidate_update'
  | 'trade_candidate_expired'
  | 'strategy_performance_update'
  // Connection events
  | 'connection_status'
  | 'ping'
  | 'pong'
  | 'error';

// ==========================================================
// Type definitions for specific event payloads
// ==========================================================

export interface PortfolioUpdatePayload {
  account: 'live' | 'paper';
  summary: PortfolioSummary;
  positions: Position[];
}

export interface PositionUpdatePayload {
  account: 'live' | 'paper';
  symbol: string;
  position: Position;
}

export interface TradeExecutedPayload {
  account: 'live' | 'paper';
  trade: Trade;
}

export interface AccountBalanceUpdatePayload {
  account: 'live' | 'paper';
  balance: number;
  previous_balance: number;
  currency: string;
  timestamp: string;
}

export interface MarketRegimeChangePayload {
  previous_regime: MarketRegimeData;
  current_regime: MarketRegimeData;
  change_factors: string[];
}

export interface SentimentUpdatePayload extends SentimentData {}

export interface NewsAlertPayload extends NewsItem {
  priority: 'high' | 'medium' | 'low';
  requires_attention: boolean;
}

export interface MarketAnomalyPayload {
  anomaly_id: string;
  description: string;
  detected_at: string;
  severity: 'high' | 'medium' | 'low';
  affected_assets: string[];
  indicators: { name: string; value: number; threshold: number }[];
  recommended_action?: string;
}

export interface StrategyPriorityUpdatePayload {
  strategies: {
    id: string;
    name: string;
    previous_score: number;
    current_score: number;
    direction: 'long' | 'short' | 'neutral';
  }[];
  reason_for_update: string;
}

export interface TradeCandidateAddedPayload {
  candidate: TradeCandidate;
}

export interface TradeCandidateUpdatePayload {
  candidate_id: string;
  updates: Partial<TradeCandidate>;
  reason_for_update: string;
}

export interface TradeCandidateExpiredPayload {
  candidate_id: string;
  reason: string;
  alternative_candidates?: string[];
}

export interface StrategyPerformanceUpdatePayload {
  strategy_id: string;
  strategy_name: string;
  performance_metrics: {
    timeframe: string;
    win_rate: number;
    profit_factor: number;
    expected_value: number;
    trades_count: number;
  };
}

export interface ConnectionStatusPayload {
  status: 'connected' | 'disconnected' | 'reconnecting';
  reason?: string;
  attempt?: number;
}

export interface ErrorPayload {
  code: string;
  message: string;
  details?: any;
}

// ==========================================================
// WebSocket Connection Manager
// ==========================================================

class WebSocketManager {
  private socket: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private pingInterval: NodeJS.Timeout | null = null;
  private url: string;
  private listeners: Map<WebSocketEventType, Set<(message: any) => void>> = new Map();
  private connectionStatus: ConnectionStatusPayload['status'] = 'disconnected';

  constructor(url: string) {
    this.url = url;
  }

  public connect(authToken?: string) {
    if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
      console.log('WebSocket is already connected or connecting');
      return;
    }

    try {
      // Add authentication token to URL if provided
      const wsUrl = authToken ? `${this.url}?token=${authToken}` : this.url;
      
      this.socket = new WebSocket(wsUrl);
      
      this.socket.onopen = this.handleOpen.bind(this);
      this.socket.onmessage = this.handleMessage.bind(this);
      this.socket.onerror = this.handleError.bind(this);
      this.socket.onclose = this.handleClose.bind(this);
      
      // Update connection status
      this.connectionStatus = 'reconnecting';
      this.notifyListeners('connection_status', {
        status: 'reconnecting',
        attempt: this.reconnectAttempts + 1
      });
    } catch (error) {
      console.error('Error connecting to WebSocket:', error);
      this.attemptReconnect();
    }
  }

  public disconnect() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
    
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
    
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    
    this.connectionStatus = 'disconnected';
    this.notifyListeners('connection_status', {
      status: 'disconnected',
      reason: 'User requested disconnect'
    });
  }

  public subscribe<T>(event: WebSocketEventType, callback: (data: T) => void) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    
    this.listeners.get(event)?.add(callback);
    
    // Return unsubscribe function
    return () => {
      const eventListeners = this.listeners.get(event);
      if (eventListeners) {
        eventListeners.delete(callback);
        if (eventListeners.size === 0) {
          this.listeners.delete(event);
        }
      }
    };
  }

  public send<T>(event: WebSocketEventType, payload: T) {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      console.error('Cannot send message, WebSocket is not connected');
      return false;
    }
    
    try {
      const message: WebSocketMessage<T> = {
        event,
        timestamp: new Date().toISOString(),
        payload
      };
      
      this.socket.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error('Error sending message:', error);
      return false;
    }
  }

  public getConnectionStatus(): ConnectionStatusPayload['status'] {
    return this.connectionStatus;
  }

  private handleOpen(event: Event) {
    console.log('WebSocket connected');
    this.connectionStatus = 'connected';
    this.reconnectAttempts = 0;
    
    // Start ping interval to keep connection alive
    this.startPingInterval();
    
    // Notify listeners of connection
    this.notifyListeners('connection_status', {
      status: 'connected'
    });
  }

  private handleMessage(event: MessageEvent) {
    try {
      const message = JSON.parse(event.data) as WebSocketMessage;
      
      // Handle pings automatically
      if (message.event === 'ping') {
        this.send('pong', { time: new Date().toISOString() });
        return;
      }
      
      // Notify listeners
      this.notifyListeners(message.event, message.payload);
    } catch (error) {
      console.error('Error parsing WebSocket message:', error, event.data);
    }
  }

  private handleError(event: Event) {
    console.error('WebSocket error:', event);
    this.notifyListeners('error', {
      code: 'connection_error',
      message: 'WebSocket connection error'
    });
  }

  private handleClose(event: CloseEvent) {
    console.log(`WebSocket closed with code ${event.code}: ${event.reason}`);
    this.connectionStatus = 'disconnected';
    
    // Clear ping interval
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
    
    // Notify listeners
    this.notifyListeners('connection_status', {
      status: 'disconnected',
      reason: event.reason || 'Connection closed'
    });
    
    // Attempt reconnect if not closed cleanly
    if (event.code !== 1000) {
      this.attemptReconnect();
    }
  }

  private attemptReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log('Maximum reconnect attempts reached');
      return;
    }
    
    this.reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000); // Exponential backoff with max of 30s
    
    console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
    
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }
    
    this.reconnectTimeout = setTimeout(() => {
      this.connect();
    }, delay);
  }

  private startPingInterval() {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
    }
    
    // Send ping every 30 seconds to keep connection alive
    this.pingInterval = setInterval(() => {
      if (this.socket?.readyState === WebSocket.OPEN) {
        this.send('ping', { time: new Date().toISOString() });
      }
    }, 30000);
  }

  private notifyListeners(event: WebSocketEventType, payload: any) {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.forEach(callback => {
        try {
          callback(payload);
        } catch (error) {
          console.error(`Error in listener for event ${event}:`, error);
        }
      });
    }
    
    // Also notify 'all' event listeners
    const allListeners = this.listeners.get('all' as WebSocketEventType);
    if (allListeners) {
      allListeners.forEach(callback => {
        try {
          callback({ event, payload });
        } catch (error) {
          console.error(`Error in 'all' listener for event ${event}:`, error);
        }
      });
    }
  }
}

// ==========================================================
// WebSocket Global Store
// ==========================================================

interface WebSocketStore {
  manager: WebSocketManager | null;
  connectionStatus: ConnectionStatusPayload['status'];
  connect: (url: string, authToken?: string) => void;
  disconnect: () => void;
  subscribe: <T>(event: WebSocketEventType, callback: (data: T) => void) => () => void;
  send: <T>(event: WebSocketEventType, payload: T) => boolean;
}

export const useWebSocket = create<WebSocketStore>((set, get) => ({
  manager: null,
  connectionStatus: 'disconnected',
  
  connect: (url: string, authToken?: string) => {
    let manager = get().manager;
    
    if (!manager) {
      manager = new WebSocketManager(url);
      set({ manager });
    }
    
    // Update connection status on changes
    manager.subscribe('connection_status', (data: ConnectionStatusPayload) => {
      set({ connectionStatus: data.status });
    });
    
    manager.connect(authToken);
  },
  
  disconnect: () => {
    const { manager } = get();
    if (manager) {
      manager.disconnect();
    }
  },
  
  subscribe: <T>(event: WebSocketEventType, callback: (data: T) => void) => {
    const { manager } = get();
    if (!manager) {
      console.error('WebSocket manager not initialized, connect first');
      return () => {};
    }
    
    return manager.subscribe(event, callback);
  },
  
  send: <T>(event: WebSocketEventType, payload: T) => {
    const { manager } = get();
    if (!manager) {
      console.error('WebSocket manager not initialized, connect first');
      return false;
    }
    
    return manager.send(event, payload);
  }
}));

// Custom hook for subscribing to specific events with auto-cleanup
export function useWebSocketEvent<T>(event: WebSocketEventType, callback: (data: T) => void) {
  const subscribe = useWebSocket(state => state.subscribe);
  
  React.useEffect(() => {
    const unsubscribe = subscribe<T>(event, callback);
    return () => unsubscribe();
  }, [event, callback, subscribe]);
}
