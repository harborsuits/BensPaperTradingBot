import { useState, useEffect, useCallback, useRef } from 'react';
import { createReconnectingWebSocket } from '../utils/performance';
import { showErrorToast } from '../utils/toast';

interface WebSocketOptions {
  onOpen?: (event: Event) => void;
  onMessage?: (data: any) => void;
  onError?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  autoReconnect?: boolean;
  autoConnect?: boolean;
}

/**
 * Custom hook for managing WebSocket connections with automatic reconnection
 * and error handling capabilities
 */
const useWebSocket = (url: string, options: WebSocketOptions = {}) => {
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const [connectionAttempts, setConnectionAttempts] = useState<number>(0);
  const wsRef = useRef<ReturnType<typeof createReconnectingWebSocket> | null>(null);
  
  const {
    onOpen,
    onMessage,
    onError,
    onClose,
    reconnectInterval = 2000,
    maxReconnectAttempts = 10,
    autoReconnect = true,
    autoConnect = true
  } = options;

  const connect = useCallback(() => {
    // Don't try to reconnect if we're already connected
    if (wsRef.current?.getWebSocket()?.readyState === WebSocket.OPEN) {
      return;
    }
    
    wsRef.current = createReconnectingWebSocket(url, {
      onOpen: (event) => {
        setIsConnected(true);
        setConnectionAttempts(0);
        onOpen?.(event);
      },
      onMessage: (event) => {
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);
          onMessage?.(data);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
          setLastMessage(event.data);
          onMessage?.(event.data);
        }
      },
      onError: (event) => {
        setIsConnected(false);
        onError?.(event);
        showErrorToast('WebSocket connection error. Attempting to reconnect...');
      },
      onClose: (event) => {
        setIsConnected(false);
        setConnectionAttempts((prev) => prev + 1);
        onClose?.(event);
        
        if (!autoReconnect && connectionAttempts >= maxReconnectAttempts) {
          showErrorToast(`WebSocket disconnected after ${connectionAttempts} attempts`);
        }
      },
      reconnectInterval,
      maxReconnectAttempts: autoReconnect ? Infinity : maxReconnectAttempts,
    });
  }, [
    url, 
    onOpen, 
    onMessage, 
    onError, 
    onClose, 
    reconnectInterval, 
    maxReconnectAttempts,
    autoReconnect,
    connectionAttempts
  ]);

  const disconnect = useCallback(() => {
    wsRef.current?.disconnect();
    setIsConnected(false);
  }, []);

  const sendMessage = useCallback((data: any) => {
    if (typeof data !== 'string') {
      data = JSON.stringify(data);
    }
    
    if (wsRef.current && isConnected) {
      wsRef.current.send(data);
      return true;
    }
    return false;
  }, [isConnected]);

  useEffect(() => {
    if (autoConnect) {
      connect();
    }
    
    return () => {
      disconnect();
    };
  }, [connect, disconnect, autoConnect]);

  return {
    isConnected,
    lastMessage,
    connectionAttempts,
    connect,
    disconnect,
    sendMessage
  };
};

export default useWebSocket;
