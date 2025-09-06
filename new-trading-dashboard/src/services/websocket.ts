import axios, { AxiosError, AxiosRequestConfig } from 'axios';
import React from 'react';
import { WebSocketMessage, WebSocketChannel } from '@/types/api.types';

type MessageHandler<T = any> = (message: WebSocketMessage<T>) => void;
type ErrorHandler = (event: Event) => void;
type ConnectionChangeHandler = () => void;

class WebSocketService {
  private socket: WebSocket | null = null;
  private messageHandlers: Map<string, Set<MessageHandler>> = new Map();
  private channelSubscriptions: Set<WebSocketChannel> = new Set();
  private errorHandlers: Set<ErrorHandler> = new Set();
  private connectHandlers: Set<ConnectionChangeHandler> = new Set();
  private disconnectHandlers: Set<ConnectionChangeHandler> = new Set();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectTimeout: number | null = null;
  private isConnecting = false;
  private pingInterval: number | null = null;
  private url: string;

  constructor(baseUrl?: string) {
    const envBase = (import.meta as any).env?.VITE_WS_BASE_URL as string | undefined;
    const base = envBase ? envBase.replace(/\/$/, '') : ((typeof location !== 'undefined') ? location.origin.replace(/^http/, 'ws') : 'ws://localhost:4000');
    this.url = (baseUrl || base + '/ws');
    // Default handler for ping responses
    this.addMessageHandler('pong', () => console.debug('Received pong from server'));
  }

  public connect(): void {
    if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
      return;
    }

    if (this.isConnecting) {
      return;
    }

    this.isConnecting = true;
    const token = localStorage.getItem('auth_token');
    const url = token ? `${this.url}?token=${token}` : this.url;
    
    try {
      this.socket = new WebSocket(url);
      
      this.socket.onopen = () => {
        console.log('WebSocket connection established');
        this.isConnecting = false;
        this.reconnectAttempts = 0;
        
        // Subscribe to previously subscribed channels
        this.resubscribeToChannels();
        
        // Start ping interval to keep connection alive
        this.startPingInterval();
        
        // Notify connection handlers
        this.connectHandlers.forEach(handler => handler());
      };
      
      this.socket.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          this.handleMessage(message);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
      
      this.socket.onerror = (event) => {
        console.error('WebSocket error:', event);
        this.errorHandlers.forEach(handler => handler(event));
      };
      
      this.socket.onclose = (event) => {
        console.log('WebSocket connection closed:', event.code, event.reason);
        this.isConnecting = false;
        this.clearPingInterval();
        
        // Notify disconnect handlers
        this.disconnectHandlers.forEach(handler => handler());
        
        // Try to reconnect if the connection was closed unexpectedly
        if (event.code !== 1000) { // 1000 is normal closure
          this.attemptReconnect();
        }
      };
    } catch (error) {
      console.error('Error creating WebSocket connection:', error);
      this.isConnecting = false;
      this.attemptReconnect();
    }
  }

  public disconnect(): void {
    if (this.socket) {
      this.clearPingInterval();
      if (this.reconnectTimeout) {
        clearTimeout(this.reconnectTimeout);
        this.reconnectTimeout = null;
      }
      
      if (this.socket.readyState === WebSocket.OPEN) {
        this.socket.close(1000, 'User initiated disconnect');
      }
      
      this.socket = null;
      this.reconnectAttempts = 0;
    }
  }

  public subscribe(channel: WebSocketChannel): void {
    if (!this.channelSubscriptions.has(channel)) {
      this.channelSubscriptions.add(channel);
      
      if (this.isConnected()) {
        this.sendSubscription(channel, 'subscribe');
      }
    }
  }

  public unsubscribe(channel: WebSocketChannel): void {
    if (this.channelSubscriptions.has(channel)) {
      this.channelSubscriptions.delete(channel);
      
      if (this.isConnected()) {
        this.sendSubscription(channel, 'unsubscribe');
      }
    }
  }

  public addMessageHandler<T>(type: string, handler: MessageHandler<T>): () => void {
    if (!this.messageHandlers.has(type)) {
      this.messageHandlers.set(type, new Set());
    }
    
    const handlers = this.messageHandlers.get(type)!;
    handlers.add(handler as MessageHandler);
    
    // Return a function to remove this handler
    return () => {
      if (this.messageHandlers.has(type)) {
        const handlers = this.messageHandlers.get(type)!;
        handlers.delete(handler as MessageHandler);
        
        if (handlers.size === 0) {
          this.messageHandlers.delete(type);
        }
      }
    };
  }

  public onError(handler: ErrorHandler): () => void {
    this.errorHandlers.add(handler);
    
    return () => {
      this.errorHandlers.delete(handler);
    };
  }

  public onConnect(handler: ConnectionChangeHandler): () => void {
    this.connectHandlers.add(handler);
    
    return () => {
      this.connectHandlers.delete(handler);
    };
  }

  public onDisconnect(handler: ConnectionChangeHandler): () => void {
    this.disconnectHandlers.add(handler);
    
    return () => {
      this.disconnectHandlers.delete(handler);
    };
  }

  public isConnected(): boolean {
    return this.socket !== null && this.socket.readyState === WebSocket.OPEN;
  }

  // Private methods
  private handleMessage(message: WebSocketMessage): void {
    // Log incoming messages (can be removed in production)
    console.debug('WebSocket message received:', message);
    
    // Call handlers registered for this message type
    if (this.messageHandlers.has(message.type)) {
      const handlers = this.messageHandlers.get(message.type)!;
      handlers.forEach(handler => handler(message));
    }
    
    // Call handlers registered for all messages on this channel
    if (this.messageHandlers.has(`channel:${message.channel}`)) {
      const handlers = this.messageHandlers.get(`channel:${message.channel}`)!;
      handlers.forEach(handler => handler(message));
    }
    
    // Call handlers registered for all messages
    if (this.messageHandlers.has('*')) {
      const handlers = this.messageHandlers.get('*')!;
      handlers.forEach(handler => handler(message));
    }
  }

  private sendSubscription(channel: WebSocketChannel, action: 'subscribe' | 'unsubscribe'): void {
    if (this.isConnected()) {
      this.socket!.send(JSON.stringify({
        type: 'subscription',
        action,
        channel
      }));
    }
  }

  private resubscribeToChannels(): void {
    if (this.isConnected()) {
      this.channelSubscriptions.forEach(channel => {
        this.sendSubscription(channel, 'subscribe');
      });
    }
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Maximum reconnect attempts reached');
      return;
    }
    
    this.reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts - 1), 30000);
    
    console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
    
    this.reconnectTimeout = window.setTimeout(() => {
      this.reconnectTimeout = null;
      this.connect();
    }, delay);
  }

  private startPingInterval(): void {
    this.clearPingInterval();
    
    // Send ping message every 30 seconds to keep the connection alive
    this.pingInterval = window.setInterval(() => {
      if (this.isConnected()) {
        this.socket!.send(JSON.stringify({ action: 'ping' }));
      }
    }, 30000);
  }

  private clearPingInterval(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }
}

// Create a singleton instance
export const webSocketService = new WebSocketService();

// React hook for using WebSocket in components
export function useWebSocketMessage<T = any>(
  type: string,
  handler: MessageHandler<T>,
  dependencies: any[] = []
): void {
  const handlerRef = React.useRef(handler);
  
  React.useEffect(() => {
    handlerRef.current = handler;
  }, [handler, ...dependencies]);
  
  React.useEffect(() => {
    const removeHandler = webSocketService.addMessageHandler<T>(type, (message) => {
      handlerRef.current(message);
    });
    
    return removeHandler;
  }, [type]);
}

// React hook for subscribing to a channel
export function useWebSocketChannel(
  channel: WebSocketChannel,
  autoConnect: boolean = true
): {
  isConnected: boolean;
  connect: () => void;
  disconnect: () => void;
} {
  const [isConnected, setIsConnected] = React.useState(webSocketService.isConnected());
  
  React.useEffect(() => {
    if (autoConnect) {
      webSocketService.connect();
    }
    
    webSocketService.subscribe(channel);
    
    const onConnect = () => setIsConnected(true);
    const onDisconnect = () => setIsConnected(false);
    
    const removeConnectHandler = webSocketService.onConnect(onConnect);
    const removeDisconnectHandler = webSocketService.onDisconnect(onDisconnect);
    
    return () => {
      webSocketService.unsubscribe(channel);
      removeConnectHandler();
      removeDisconnectHandler();
    };
  }, [channel, autoConnect]);
  
  return {
    isConnected,
    connect: webSocketService.connect.bind(webSocketService),
    disconnect: webSocketService.disconnect.bind(webSocketService)
  };
}

// Default export
export default webSocketService;
