/**
 * Performance optimization utilities for handling large datasets and long-running WebSocket connections
 */

/**
 * Debounces a function to prevent it from being called too frequently
 * @param func The function to debounce
 * @param wait The time to wait in milliseconds
 */
export function debounce<T extends (...args: any[]) => any>(func: T, wait: number): (...args: Parameters<T>) => void {
  let timeout: ReturnType<typeof setTimeout> | null = null;
  
  return function(...args: Parameters<T>): void {
    const later = () => {
      timeout = null;
      func(...args);
    };
    
    if (timeout !== null) {
      clearTimeout(timeout);
    }
    timeout = setTimeout(later, wait);
  };
}

/**
 * Throttles a function to limit how often it can be called
 * @param func The function to throttle
 * @param limit The time limit in milliseconds
 */
export function throttle<T extends (...args: any[]) => any>(func: T, limit: number): (...args: Parameters<T>) => void {
  let inThrottle = false;
  
  return function(...args: Parameters<T>): void {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => {
        inThrottle = false;
      }, limit);
    }
  };
}

/**
 * Creates a memoized version of a function that caches results
 * @param func The function to memoize
 */
export function memoize<T extends (...args: any[]) => any>(func: T): T {
  const cache = new Map<string, ReturnType<T>>();
  
  return ((...args: Parameters<T>): ReturnType<T> => {
    const key = JSON.stringify(args);
    if (cache.has(key)) {
      return cache.get(key) as ReturnType<T>;
    }
    const result = func(...args);
    cache.set(key, result);
    return result;
  }) as T;
}

/**
 * Batches updates to reduce re-renders
 * @param callback The callback to execute with batched updates
 * @param delay The time to wait for batching in milliseconds
 */
export function batchUpdates<T>(callback: (items: T[]) => void, delay: number = 100): (item: T) => void {
  let batch: T[] = [];
  let timeout: ReturnType<typeof setTimeout> | null = null;
  
  return (item: T): void => {
    batch.push(item);
    
    if (timeout === null) {
      timeout = setTimeout(() => {
        const currentBatch = [...batch];
        batch = [];
        timeout = null;
        callback(currentBatch);
      }, delay);
    }
  };
}

/**
 * Manages WebSocket reconnection with exponential backoff
 * @param url The WebSocket URL
 * @param options WebSocket options
 */
export function createReconnectingWebSocket(
  url: string, 
  options: {
    onOpen?: (event: Event) => void;
    onMessage?: (event: MessageEvent) => void;
    onError?: (event: Event) => void;
    onClose?: (event: CloseEvent) => void;
    maxReconnectAttempts?: number;
    reconnectInterval?: number;
    maxReconnectInterval?: number;
  } = {}
) {
  const {
    onOpen,
    onMessage,
    onError,
    onClose,
    maxReconnectAttempts = 10,
    reconnectInterval = 1000,
    maxReconnectInterval = 30000
  } = options;
  
  let ws: WebSocket | null = null;
  let reconnectAttempts = 0;
  let currentReconnectInterval = reconnectInterval;
  let reconnectTimeoutId: ReturnType<typeof setTimeout> | null = null;
  
  const connect = () => {
    // Clean up any existing connection
    if (ws) {
      ws.onopen = null;
      ws.onmessage = null;
      ws.onerror = null;
      ws.onclose = null;
      ws.close();
    }
    
    ws = new WebSocket(url);
    
    ws.onopen = (event) => {
      // Reset reconnection parameters on successful connection
      reconnectAttempts = 0;
      currentReconnectInterval = reconnectInterval;
      onOpen?.(event);
    };
    
    ws.onmessage = (event) => {
      onMessage?.(event);
    };
    
    ws.onerror = (event) => {
      onError?.(event);
    };
    
    ws.onclose = (event) => {
      onClose?.(event);
      
      // Don't reconnect if the closure was clean (code 1000)
      if (event.code !== 1000 && reconnectAttempts < maxReconnectAttempts) {
        reconnectTimeoutId = setTimeout(() => {
          reconnectAttempts++;
          // Exponential backoff
          currentReconnectInterval = Math.min(
            currentReconnectInterval * 1.5, 
            maxReconnectInterval
          );
          connect();
        }, currentReconnectInterval);
      }
    };
  };
  
  const disconnect = () => {
    if (reconnectTimeoutId) {
      clearTimeout(reconnectTimeoutId);
    }
    
    if (ws) {
      ws.close(1000);
    }
  };
  
  // Initial connection
  connect();
  
  return {
    send: (data: string | ArrayBufferLike | Blob | ArrayBufferView) => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(data);
      }
    },
    disconnect,
    getWebSocket: () => ws
  };
}
