import { WEBSOCKET_CONFIG } from '@/config/dataRefreshConfig';

interface WebSocketMessage {
  type: string;
  channel?: string;
  data?: any;
  timestamp?: string;
}

type MessageHandler = (message: WebSocketMessage) => void;
type ConnectionHandler = () => void;

export class ImprovedWebSocketService {
  private socket: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private isIntentionalClose = false;
  private lastMessageTime = Date.now();
  
  private messageHandlers: Map<string, Set<MessageHandler>> = new Map();
  private connectionHandlers: Set<ConnectionHandler> = new Set();
  private disconnectionHandlers: Set<ConnectionHandler> = new Set();
  
  constructor(url: string) {
    this.url = url;
    
    // Monitor page visibility
    document.addEventListener('visibilitychange', this.handleVisibilityChange);
    
    // Monitor online/offline status
    window.addEventListener('online', this.handleOnline);
    window.addEventListener('offline', this.handleOffline);
  }
  
  connect(): void {
    if (this.socket?.readyState === WebSocket.OPEN || 
        this.socket?.readyState === WebSocket.CONNECTING) {
      return;
    }
    
    this.isIntentionalClose = false;
    
    try {
      console.log('[WS] Connecting to:', this.url);
      this.socket = new WebSocket(this.url);
      
      this.socket.onopen = this.handleOpen;
      this.socket.onmessage = this.handleMessage;
      this.socket.onerror = this.handleError;
      this.socket.onclose = this.handleClose;
    } catch (error) {
      console.error('[WS] Connection error:', error);
      this.scheduleReconnect();
    }
  }
  
  disconnect(): void {
    this.isIntentionalClose = true;
    this.cleanup();
    
    if (this.socket) {
      this.socket.close(1000, 'Client disconnect');
      this.socket = null;
    }
  }
  
  private handleOpen = (): void => {
    console.log('[WS] Connected successfully');
    this.reconnectAttempts = 0;
    this.lastMessageTime = Date.now();
    
    // Start heartbeat
    this.startHeartbeat();
    
    // Notify handlers
    this.connectionHandlers.forEach(handler => handler());
  };
  
  private handleMessage = (event: MessageEvent): void => {
    this.lastMessageTime = Date.now();
    
    try {
      const message: WebSocketMessage = JSON.parse(event.data);
      
      // Handle different message types
      if (message.type === 'pong') {
        // Heartbeat response
        return;
      }
      
      // Notify specific handlers
      const handlers = this.messageHandlers.get(message.type);
      if (handlers) {
        handlers.forEach(handler => handler(message));
      }
      
      // Notify wildcard handlers
      const wildcardHandlers = this.messageHandlers.get('*');
      if (wildcardHandlers) {
        wildcardHandlers.forEach(handler => handler(message));
      }
    } catch (error) {
      console.error('[WS] Message parse error:', error);
    }
  };
  
  private handleError = (event: Event): void => {
    console.error('[WS] WebSocket error:', event);
  };
  
  private handleClose = (event: CloseEvent): void => {
    console.log(`[WS] Disconnected: code=${event.code}, reason=${event.reason}`);
    
    this.cleanup();
    
    // Notify handlers
    this.disconnectionHandlers.forEach(handler => handler());
    
    // Attempt reconnect if not intentional
    if (!this.isIntentionalClose && event.code !== 1000) {
      this.scheduleReconnect();
    }
  };
  
  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= WEBSOCKET_CONFIG.maxReconnectAttempts) {
      console.error('[WS] Max reconnection attempts reached');
      return;
    }
    
    // Calculate backoff delay
    const baseDelay = WEBSOCKET_CONFIG.reconnectInterval;
    const delay = Math.min(
      baseDelay * Math.pow(1.5, this.reconnectAttempts),
      WEBSOCKET_CONFIG.maxReconnectInterval
    );
    
    this.reconnectAttempts++;
    console.log(`[WS] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    
    this.reconnectTimeout = setTimeout(() => {
      this.reconnectTimeout = null;
      this.connect();
    }, delay);
  }
  
  private startHeartbeat(): void {
    this.stopHeartbeat();
    
    this.heartbeatInterval = setInterval(() => {
      if (this.socket?.readyState === WebSocket.OPEN) {
        // Check if we've received any messages recently
        const timeSinceLastMessage = Date.now() - this.lastMessageTime;
        if (timeSinceLastMessage > WEBSOCKET_CONFIG.heartbeatInterval * 2) {
          console.warn('[WS] No messages received for', timeSinceLastMessage, 'ms');
        }
        
        // Send ping
        this.send({ type: 'ping', timestamp: new Date().toISOString() });
      }
    }, WEBSOCKET_CONFIG.heartbeatInterval);
  }
  
  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }
  
  private cleanup(): void {
    this.stopHeartbeat();
    
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
  }
  
  private handleVisibilityChange = (): void => {
    if (document.hidden) {
      console.log('[WS] Page hidden, reducing activity');
      // Could implement reduced heartbeat or temporary disconnect
    } else {
      console.log('[WS] Page visible, resuming normal activity');
      // Ensure connection is healthy
      if (!this.isConnected()) {
        this.connect();
      }
    }
  };
  
  private handleOnline = (): void => {
    console.log('[WS] Network online, attempting reconnect');
    if (!this.isConnected()) {
      this.connect();
    }
  };
  
  private handleOffline = (): void => {
    console.log('[WS] Network offline');
  };
  
  // Public API
  
  isConnected(): boolean {
    return this.socket?.readyState === WebSocket.OPEN;
  }
  
  send(message: any): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(message));
    } else {
      console.warn('[WS] Cannot send message, not connected');
    }
  }
  
  subscribe(channel: string): void {
    this.send({ type: 'subscribe', channel });
  }
  
  unsubscribe(channel: string): void {
    this.send({ type: 'unsubscribe', channel });
  }
  
  on(event: string, handler: MessageHandler): void {
    if (!this.messageHandlers.has(event)) {
      this.messageHandlers.set(event, new Set());
    }
    this.messageHandlers.get(event)!.add(handler);
  }
  
  off(event: string, handler: MessageHandler): void {
    const handlers = this.messageHandlers.get(event);
    if (handlers) {
      handlers.delete(handler);
    }
  }
  
  onConnect(handler: ConnectionHandler): void {
    this.connectionHandlers.add(handler);
  }
  
  onDisconnect(handler: ConnectionHandler): void {
    this.disconnectionHandlers.add(handler);
  }
  
  destroy(): void {
    document.removeEventListener('visibilitychange', this.handleVisibilityChange);
    window.removeEventListener('online', this.handleOnline);
    window.removeEventListener('offline', this.handleOffline);
    
    this.disconnect();
    this.messageHandlers.clear();
    this.connectionHandlers.clear();
    this.disconnectionHandlers.clear();
  }
}

// Singleton instance (deprecated - use WebSocketContext instead)
let wsInstance: ImprovedWebSocketService | null = null;

export function getWebSocketService(): ImprovedWebSocketService {
  if (!wsInstance) {
    // Use backend WebSocket URL directly (bypassing Vite proxy for WebSocket)
    const wsUrl = 'ws://localhost:4000/ws';

    wsInstance = new ImprovedWebSocketService(wsUrl);
  }
  return wsInstance;
}
