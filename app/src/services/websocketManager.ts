import { create } from 'zustand';

// Import types from existing websocketService.ts
import { 
  WebSocketMessage,
  WebSocketEventType,
  PortfolioUpdatePayload,
  PositionUpdatePayload,
  TradeExecutedPayload,
  AccountBalanceUpdatePayload,
  MarketRegimeChangePayload,
  SentimentUpdatePayload,
  NewsAlertPayload,
  MarketAnomalyPayload,
  StrategyPriorityUpdatePayload,
  TradeCandidateAddedPayload,
  TradeCandidateUpdatePayload,
  TradeCandidateExpiredPayload,
  StrategyPerformanceUpdatePayload,
  ConnectionStatusPayload,
  ErrorPayload
} from './websocketService';
import { useEffect, useRef } from 'react';
import { withKeepalive } from '../utils/wsKeepalive';
import { pingApi } from './health';

// Channels for the backend WebSocket
export type WebSocketChannel = 
  | 'market_data'
  | 'trades'
  | 'portfolio'
  | 'strategies'
  | 'market_context'
  | 'alerts'
  | 'logs';

// Channel-to-event mapping to help organize which events belong to which channels
const channelEventMap: Record<WebSocketChannel, WebSocketEventType[]> = {
  market_data: ['market_regime_change', 'market_anomaly_detected'],
  trades: ['trade_executed', 'trade_candidate_added', 'trade_candidate_update', 'trade_candidate_expired'],
  portfolio: ['portfolio_update', 'position_update', 'account_balance_update'],
  strategies: ['strategy_priority_update', 'strategy_performance_update'],
  market_context: ['sentiment_update', 'market_regime_change'],
  alerts: ['news_alert'],
  logs: []
};

/**
 * Central WebSocket Manager to handle connections to the trading backend
 */
export class TradingWebSocketManager {
  private socket: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
  private pingInterval: ReturnType<typeof setTimeout> | null = null;
  private baseUrl: string;
  private keepaliveCleanup: (() => void) | null = null;
  private connecting = false;
  
  // Listeners for events and channels
  private eventListeners: Map<WebSocketEventType, Set<(data: any) => void>> = new Map();
  private channelListeners: Map<WebSocketChannel, Set<(data: any) => void>> = new Map();
  private statusListeners: Set<(status: ConnectionStatusPayload) => void> = new Set();
  
  // Active subscriptions to resubscribe on reconnect
  private activeChannels: Set<WebSocketChannel> = new Set();
  
  constructor(baseUrl: string = (typeof location !== 'undefined' ? location.origin.replace(/^http/, 'ws') + '/ws' : 'ws://localhost:8000/ws')) {
    this.baseUrl = baseUrl;
  }
  
  /**
   * Connect to the WebSocket server
   * @param token Optional authentication token
   */
  public connect(token?: string): void {
    // Prevent duplicate connections
    if (this.connecting || 
        (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING))) {
      // Don't log to avoid console spam
      return;
    }
    
    this.connecting = true;
    
    let url = this.baseUrl;
    if (token) {
      url += `?token=${token}`;
    }
    
    try {
      this.socket = new WebSocket(url);
      
      this.socket.onopen = this.handleOpen.bind(this);
      this.socket.onmessage = this.handleMessage.bind(this);
      this.socket.onerror = this.handleError.bind(this);
      this.socket.onclose = this.handleClose.bind(this);
      
      // Notify listeners of connecting status
      this.notifyStatusChange({
        status: 'reconnecting',
        attempt: this.reconnectAttempts
      });
    } catch (error) {
      console.error('Error creating WebSocket connection:', error);
      this.connecting = false;
      this.attemptReconnect();
    }
  }
  
  /**
   * Disconnect from the WebSocket server
   */
  public disconnect(): void {
    if (!this.socket) return;
    
    this.clearIntervals();
    
    if (this.socket.readyState === WebSocket.OPEN) {
      this.socket.close(1000, 'Client initiated disconnect');
    }
    
    this.socket = null;
    this.reconnectAttempts = 0;
    
    this.notifyStatusChange({
      status: 'disconnected',
      reason: 'User initiated disconnect'
    });
  }
  
  /**
   * Subscribe to a specific channel
   * @param channel The channel to subscribe to
   */
  public subscribeToChannel(channel: WebSocketChannel): void {
    if (this.activeChannels.has(channel)) {
      return; // Already subscribed
    }
    
    this.activeChannels.add(channel);
    
    if (this.isConnected()) {
      this.sendSubscription(channel, 'subscribe');
    }
  }
  
  /**
   * Unsubscribe from a specific channel
   * @param channel The channel to unsubscribe from
   */
  public unsubscribeFromChannel(channel: WebSocketChannel): void {
    if (!this.activeChannels.has(channel)) {
      return; // Not subscribed
    }
    
    this.activeChannels.delete(channel);
    
    if (this.isConnected()) {
      this.sendSubscription(channel, 'unsubscribe');
    }
  }
  
  /**
   * Register a listener for a specific event type
   * @param eventType Event type to listen for
   * @param callback Callback function to execute when event occurs
   * @returns Function to unregister the listener
   */
  public addEventTypeListener<T>(eventType: WebSocketEventType, callback: (data: T) => void): () => void {
    if (!this.eventListeners.has(eventType)) {
      this.eventListeners.set(eventType, new Set());
    }
    
    const listeners = this.eventListeners.get(eventType)!;
    listeners.add(callback as any);
    
    return () => {
      listeners.delete(callback as any);
      if (listeners.size === 0) {
        this.eventListeners.delete(eventType);
      }
    };
  }
  
  /**
   * Register a listener for all messages on a specific channel
   * @param channel Channel to listen on
   * @param callback Callback function to execute when any message arrives on the channel
   * @returns Function to unregister the listener
   */
  public addChannelListener<T>(channel: WebSocketChannel, callback: (data: WebSocketMessage<T>) => void): () => void {
    if (!this.channelListeners.has(channel)) {
      this.channelListeners.set(channel, new Set());
    }
    
    const listeners = this.channelListeners.get(channel)!;
    listeners.add(callback as any);
    
    // Subscribe to the channel if we're not already subscribed
    this.subscribeToChannel(channel);
    
    return () => {
      listeners.delete(callback as any);
      if (listeners.size === 0) {
        this.channelListeners.delete(channel);
        // Optional: unsubscribe from channel when no listeners remain
        // this.unsubscribeFromChannel(channel);
      }
    };
  }
  
  /**
   * Register a listener for connection status changes
   * @param callback Callback function to execute when connection status changes
   * @returns Function to unregister the listener
   */
  public addStatusListener(callback: (status: ConnectionStatusPayload) => void): () => void {
    this.statusListeners.add(callback);
    
    // Immediately notify with current status
    if (this.socket) {
      const currentStatus: ConnectionStatusPayload = {
        status: this.getConnectionStatus()
      };
      callback(currentStatus);
    }
    
    return () => {
      this.statusListeners.delete(callback);
    };
  }
  
  /**
   * Send a message to the server
   * @param event Event type
   * @param data Message data
   */
  public sendMessage<T>(event: WebSocketEventType, data: T): boolean {
    if (!this.isConnected()) {
      console.warn('Cannot send message, WebSocket not connected');
      return false;
    }
    
    try {
      const message: WebSocketMessage<T> = {
        event,
        timestamp: new Date().toISOString(),
        payload: data
      };
      
      this.socket!.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error('Error sending WebSocket message:', error);
      return false;
    }
  }
  
  /**
   * Check if the WebSocket is currently connected
   */
  public isConnected(): boolean {
    return !!this.socket && this.socket.readyState === WebSocket.OPEN;
  }
  
  /**
   * Get the current connection status
   */
  public getConnectionStatus(): ConnectionStatusPayload['status'] {
    if (!this.socket) return 'disconnected';
    
    switch (this.socket.readyState) {
      case WebSocket.CONNECTING:
        return 'reconnecting';
      case WebSocket.OPEN:
        return 'connected';
      default:
        return 'disconnected';
    }
  }
  
  // Private methods
  
  private handleOpen(event: Event): void {
    console.log('WebSocket connection established');
    this.reconnectAttempts = 0;
    this.connecting = false;
    
    // Clean up any existing keepalive
    if (this.keepaliveCleanup) {
      this.keepaliveCleanup();
      this.keepaliveCleanup = null;
    }
    
    // Attach keepalive with visibility-aware pings and HTTP fallback
    if (this.socket) {
      this.keepaliveCleanup = withKeepalive(this.socket, 15000);
    }
    
    // Start ping interval (legacy system - we can eventually remove this)
    // this.startPingInterval();
    
    // Resubscribe to all active channels
    this.resubscribeToChannels();
    
    // Notify status listeners
    this.notifyStatusChange({
      status: 'connected'
    });
  }
  
  private handleMessage(event: MessageEvent): void {
    try {
      const message = JSON.parse(event.data) as WebSocketMessage;
      console.debug('WebSocket message received:', message);
      
      // Handle ping/pong separately
      if (message.event === 'ping') {
        this.sendMessage('pong', { time: new Date().toISOString() });
        return;
      }
      
      // Find the channel for this event type
      let channel: WebSocketChannel | undefined;
      for (const [ch, events] of Object.entries(channelEventMap)) {
        if (events.includes(message.event)) {
          channel = ch as WebSocketChannel;
          break;
        }
      }
      
      // Notify event listeners
      this.notifyEventListeners(message.event, message.payload);
      
      // Notify channel listeners if channel found
      if (channel) {
        this.notifyChannelListeners(channel, message);
      }
      
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  }
  
  private handleError(event: Event): void {
    console.error('WebSocket error:', event);
    
    // Notify status listeners
    this.notifyStatusChange({
      status: 'disconnected',
      reason: 'Error occurred'
    });
  }
  
  private handleClose(event: CloseEvent): void {
    console.log('WebSocket connection closed:', event.code, event.reason);
    
    this.clearIntervals();
    
    // Notify status listeners
    this.notifyStatusChange({
      status: 'disconnected',
      reason: event.reason || 'Connection closed'
    });
    
    // Handle specific close codes
    if (event.code === 1000) {
      // Normal closure, don't reconnect
      return;
    }
    
    if (event.code === 4000) {
      // Our keepalive timeout - reconnect immediately
      console.warn('WebSocket keepalive timeout - reconnecting');
      this.attemptReconnect();
      return;
    }
    
    if (event.code === 1011) {
      // Server error, reconnect with backoff
      console.warn('Server error - reconnecting with backoff');
      this.attemptReconnect();
      return;
    }
    
    // Other unexpected closure - attempt to reconnect
    this.attemptReconnect();
  }
  
  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log('Maximum reconnect attempts reached');
      return;
    }
    
    // Don't attempt reconnect if page is hidden
    if (document.hidden) {
      console.log('Page is hidden, delaying reconnect until visible');
      
      // Add visibility listener to reconnect when tab becomes visible
      const onVisibilityChange = () => {
        if (!document.hidden) {
          console.log('Page visible, attempting reconnect');
          document.removeEventListener('visibilitychange', onVisibilityChange);
          this.attemptReconnect();
        }
      };
      
      document.addEventListener('visibilitychange', onVisibilityChange);
      return;
    }
    
    this.reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(1.5, this.reconnectAttempts - 1), 5000);
    
    console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
    
    this.notifyStatusChange({
      status: 'reconnecting',
      attempt: this.reconnectAttempts
    });
    
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }
    
    this.reconnectTimeout = setTimeout(() => {
      this.reconnectTimeout = null;
      this.connect();
    }, delay);
  }
  
  private startPingInterval(): void {
    this.clearIntervals();
    
    // Send ping every 30 seconds to keep connection alive
    this.pingInterval = setInterval(() => {
      if (this.isConnected()) {
        this.sendMessage('ping', { time: new Date().toISOString() });
      }
    }, 30000);
  }
  
  private clearIntervals(): void {
    // Clean up keepalive
    if (this.keepaliveCleanup) {
      this.keepaliveCleanup();
      this.keepaliveCleanup = null;
    }
    
    // Clean up legacy ping interval
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
    
    // Clean up reconnect timeout
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    
    // Reset connection state
    this.connecting = false;
  }
  
  private sendSubscription(channel: WebSocketChannel, action: 'subscribe' | 'unsubscribe'): void {
    if (!this.isConnected()) return;
    
    this.socket!.send(JSON.stringify({
      type: 'subscription',
      action,
      channel
    }));
  }
  
  private resubscribeToChannels(): void {
    if (!this.isConnected()) return;
    
    this.activeChannels.forEach(channel => {
      this.sendSubscription(channel, 'subscribe');
    });
  }
  
  private notifyEventListeners(eventType: WebSocketEventType, data: any): void {
    const listeners = this.eventListeners.get(eventType);
    if (!listeners) return;
    
    listeners.forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        console.error(`Error in event listener for ${eventType}:`, error);
      }
    });
  }
  
  private notifyChannelListeners(channel: WebSocketChannel, message: WebSocketMessage): void {
    const listeners = this.channelListeners.get(channel);
    if (!listeners) return;
    
    listeners.forEach(callback => {
      try {
        callback(message);
      } catch (error) {
        console.error(`Error in channel listener for ${channel}:`, error);
      }
    });
  }
  
  private notifyStatusChange(status: ConnectionStatusPayload): void {
    this.statusListeners.forEach(callback => {
      try {
        callback(status);
      } catch (error) {
        console.error('Error in status listener:', error);
      }
    });
  }
}

// Create a singleton instance
export const tradingWebSocket = new TradingWebSocketManager();

// Zustand store for global state management
interface WebSocketState {
  connectionStatus: ConnectionStatusPayload['status'];
  connect: (authToken?: string) => void;
  disconnect: () => void;
  subscribeToChannel: (channel: WebSocketChannel) => void;
  unsubscribeFromChannel: (channel: WebSocketChannel) => void;
  sendMessage: <T>(event: WebSocketEventType, data: T) => boolean;
  isConnected: () => boolean;
}

export const useWebSocketStore = create<WebSocketState>((set, get) => ({
  connectionStatus: 'disconnected',
  
  connect: (authToken?: string) => {
    tradingWebSocket.connect(authToken);
    
    // Setup status listener
    tradingWebSocket.addStatusListener((status) => {
      set({ connectionStatus: status.status });
    });
  },
  
  disconnect: () => {
    tradingWebSocket.disconnect();
  },
  
  subscribeToChannel: (channel: WebSocketChannel) => {
    tradingWebSocket.subscribeToChannel(channel);
  },
  
  unsubscribeFromChannel: (channel: WebSocketChannel) => {
    tradingWebSocket.unsubscribeFromChannel(channel);
  },
  
  sendMessage: <T>(event: WebSocketEventType, data: T) => {
    return tradingWebSocket.sendMessage(event, data);
  },
  
  isConnected: () => {
    return tradingWebSocket.isConnected();
  }
}));

// React hooks for easy use in components

/**
 * Hook to subscribe to specific event types
 */
export function useWebSocketEvent<T>(eventType: WebSocketEventType, callback: (data: T) => void) {
  const callbackRef = useRef(callback);
  
  // Update the callback ref when the callback changes
  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);
  
  useEffect(() => {
    // Create a stable callback that uses the ref
    const stableCallback = (data: T) => {
      callbackRef.current(data);
    };
    
    const unsubscribe = tradingWebSocket.addEventTypeListener(eventType, stableCallback);
    return unsubscribe;
  }, [eventType]);
}

/**
 * Hook to subscribe to a specific channel
 */
export function useWebSocketChannel<T>(
  channel: WebSocketChannel,
  autoConnect: boolean = true
) {
  const { connect, connectionStatus, subscribeToChannel, unsubscribeFromChannel } = useWebSocketStore();
  
  useEffect(() => {
    if (autoConnect && connectionStatus === 'disconnected') {
      connect();
    }
    
    subscribeToChannel(channel);
    
    return () => {
      unsubscribeFromChannel(channel);
    };
  }, [channel, autoConnect, connect, subscribeToChannel, unsubscribeFromChannel, connectionStatus]);
  
  return {
    isConnected: connectionStatus === 'connected',
    status: connectionStatus
  };
}

/**
 * Hook to listen for messages on a specific channel
 */
export function useWebSocketChannelMessage<T>(
  channel: WebSocketChannel,
  callback: (message: WebSocketMessage<T>) => void,
  autoConnect: boolean = true
) {
  const callbackRef = useRef(callback);
  
  // Update the callback ref when the callback changes
  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);
  
  // Use the channel hook to handle subscription
  const { isConnected, status } = useWebSocketChannel(channel, autoConnect);
  
  useEffect(() => {
    if (status !== 'connected') return;
    
    // Create a stable callback that uses the ref
    const stableCallback = (message: WebSocketMessage<T>) => {
      callbackRef.current(message);
    };
    
    const unsubscribe = tradingWebSocket.addChannelListener(channel, stableCallback);
    return unsubscribe;
  }, [channel, status]);
  
  return {
    isConnected,
    status
  };
}

/**
 * Hook to handle connection status changes
 */
export function useWebSocketStatus() {
  const connectionStatus = useWebSocketStore(state => state.connectionStatus);
  const connect = useWebSocketStore(state => state.connect);
  const disconnect = useWebSocketStore(state => state.disconnect);
  
  return {
    status: connectionStatus,
    connect,
    disconnect,
    isConnected: connectionStatus === 'connected'
  };
}
