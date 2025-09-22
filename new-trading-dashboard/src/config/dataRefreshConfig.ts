/**
 * Centralized data refresh configuration
 * All intervals and stale times in milliseconds
 */

export const DATA_REFRESH_CONFIG = {
  // Critical real-time data
  quotes: {
    refetchInterval: 3000,     // 3 seconds
    staleTime: 2000,           // 2 seconds
  },
  trades: {
    refetchInterval: 5000,     // 5 seconds
    staleTime: 3000,           // 3 seconds
  },
  decisions: {
    refetchInterval: 5000,     // 5 seconds
    staleTime: 3000,           // 3 seconds
  },
  orders: {
    refetchInterval: 3000,     // 3 seconds
    staleTime: 2000,           // 2 seconds
  },
  
  // Portfolio data
  paperAccount: {
    refetchInterval: 10000,    // 10 seconds
    staleTime: 5000,           // 5 seconds
  },
  paperPositions: {
    refetchInterval: 10000,    // 10 seconds
    staleTime: 5000,           // 5 seconds
  },
  
  // Strategy and context data
  strategies: {
    refetchInterval: 30000,    // 30 seconds
    staleTime: 15000,          // 15 seconds
  },
  context: {
    refetchInterval: 30000,    // 30 seconds
    staleTime: 15000,          // 15 seconds
  },
  
  // System monitoring
  metrics: {
    refetchInterval: 15000,    // 15 seconds
    staleTime: 10000,          // 10 seconds
  },
  health: {
    refetchInterval: 10000,    // 10 seconds
    staleTime: 5000,           // 5 seconds
  },
  
  // News and insights
  news: {
    refetchInterval: 10000,    // 10 seconds
    staleTime: 5000,           // 5 seconds
  },
  
  // Evolution and optimization
  evoStatus: {
    refetchInterval: 5000,     // 5 seconds - fast updates during evolution
    staleTime: 2000,           // 2 seconds
  },
  evoHistory: {
    refetchInterval: 60000,    // 1 minute - history doesn't change often
    staleTime: 30000,          // 30 seconds
  },
  evoResults: {
    refetchInterval: 10000,    // 10 seconds
    staleTime: 5000,           // 5 seconds
  },
  
  // Default for other queries
  default: {
    refetchInterval: 30000,    // 30 seconds
    staleTime: 15000,          // 15 seconds
  }
};

// WebSocket reconnection config
export const WEBSOCKET_CONFIG = {
  reconnectInterval: 2000,     // 2 seconds initial
  maxReconnectInterval: 10000, // 10 seconds max
  maxReconnectAttempts: 20,    // 20 attempts before giving up
  heartbeatInterval: 15000,    // 15 seconds ping/pong
  connectionTimeout: 5000,     // 5 seconds to establish connection
};

// SSE reconnection config
export const SSE_CONFIG = {
  reconnectInterval: 1000,     // 1 second
  maxReconnectAttempts: 10,    // 10 attempts
};
