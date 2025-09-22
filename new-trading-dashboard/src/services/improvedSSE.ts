import { SSE_CONFIG } from '@/config/dataRefreshConfig';

interface SSEMessage {
  type: string;
  data: any;
  timestamp?: string;
}

type MessageHandler = (message: SSEMessage) => void;

export class ImprovedSSEService {
  private eventSource: EventSource | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private isIntentionalClose = false;
  private messageHandlers: Map<string, Set<MessageHandler>> = new Map();
  
  constructor(url: string) {
    this.url = url;
    
    // Monitor page visibility
    document.addEventListener('visibilitychange', this.handleVisibilityChange);
  }
  
  connect(): void {
    if (this.eventSource && 
        (this.eventSource.readyState === EventSource.OPEN || 
         this.eventSource.readyState === EventSource.CONNECTING)) {
      console.log('[SSE] Already connected or connecting, skipping');
      return;
    }
    
    this.isIntentionalClose = false;
    
    try {
      console.log('[SSE] Connecting to:', this.url);
      this.eventSource = new EventSource(this.url);
      
      // Handle connection opened
      this.eventSource.onopen = () => {
        console.log('[SSE] Connected successfully');
        this.reconnectAttempts = 0;
      };
      
      // Handle generic messages
      this.eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleMessage('message', data);
        } catch (error) {
          console.error('[SSE] Message parse error:', error);
        }
      };
      
      // Handle specific event types
      this.eventSource.addEventListener('order_update', (event: MessageEvent) => {
        try {
          const data = JSON.parse(event.data);
          this.handleMessage('order_update', data);
        } catch (error) {
          console.error('[SSE] Order update parse error:', error);
        }
      });
      
      // Handle errors
      this.eventSource.onerror = (event) => {
        console.error('[SSE] Connection error:', event);
        
        // EventSource will automatically reconnect, but we'll help it along
        if (this.eventSource?.readyState === EventSource.CLOSED) {
          this.cleanup();
          
          if (!this.isIntentionalClose) {
            this.scheduleReconnect();
          }
        }
      };
    } catch (error) {
      console.error('[SSE] Connection error:', error);
      this.scheduleReconnect();
    }
  }
  
  disconnect(): void {
    this.isIntentionalClose = true;
    this.cleanup();
    
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
  }
  
  private handleMessage(type: string, data: any): void {
    const message: SSEMessage = {
      type,
      data,
      timestamp: new Date().toISOString()
    };
    
    // Notify specific handlers
    const handlers = this.messageHandlers.get(type);
    if (handlers) {
      handlers.forEach(handler => handler(message));
    }
    
    // Notify wildcard handlers
    const wildcardHandlers = this.messageHandlers.get('*');
    if (wildcardHandlers) {
      wildcardHandlers.forEach(handler => handler(message));
    }
  }
  
  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= SSE_CONFIG.maxReconnectAttempts) {
      console.error('[SSE] Max reconnection attempts reached');
      return;
    }
    
    const delay = SSE_CONFIG.reconnectInterval * (this.reconnectAttempts + 1);
    this.reconnectAttempts++;
    
    console.log(`[SSE] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    
    this.reconnectTimeout = setTimeout(() => {
      this.reconnectTimeout = null;
      this.connect();
    }, delay);
  }
  
  private cleanup(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
  }
  
  private handleVisibilityChange = (): void => {
    if (!document.hidden && !this.isConnected() && !this.isIntentionalClose) {
      console.log('[SSE] Page visible, reconnecting');
      // Clear any existing reconnect timeout
      if (this.reconnectTimeout) {
        clearTimeout(this.reconnectTimeout);
        this.reconnectTimeout = null;
      }
      this.connect();
    }
  };
  
  // Public API
  
  isConnected(): boolean {
    return this.eventSource?.readyState === EventSource.OPEN;
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
  
  destroy(): void {
    document.removeEventListener('visibilitychange', this.handleVisibilityChange);
    this.disconnect();
    this.messageHandlers.clear();
  }
}

// Factory function to create SSE connections
export function createSSEConnection(endpoint: string): ImprovedSSEService {
  const url = `${window.location.origin}${endpoint}`;
  return new ImprovedSSEService(url);
}
