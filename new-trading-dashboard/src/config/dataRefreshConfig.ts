/**
 * Centralized configuration for data refresh intervals
 * All intervals are in milliseconds
 */

export const DATA_REFRESH_CONFIG = {
  // Default configuration for QueryClient
  default: {
    refetchInterval: 30_000, // 30 seconds
    staleTime: 20_000,      // 20 seconds
  },
  
  // Critical real-time data (quotes, positions, orders)
  CRITICAL: {
    refetchInterval: 15_000, // 15 seconds
    staleTime: 10_000,      // 10 seconds
  },
  
  // Important operational data (trades, decisions, autoloop status)
  OPERATIONAL: {
    refetchInterval: 30_000, // 30 seconds
    staleTime: 20_000,      // 20 seconds
  },
  
  // Context and analytics data (market context, performance metrics)
  ANALYTICS: {
    refetchInterval: 60_000, // 1 minute
    staleTime: 45_000,      // 45 seconds
  },
  
  // Evolution and research data (evo status, experiments)
  RESEARCH: {
    refetchInterval: 90_000, // 1.5 minutes
    staleTime: 60_000,      // 1 minute
  },
  
  // Static or rarely changing data (news, historical data)
  STATIC: {
    refetchInterval: 300_000, // 5 minutes
    staleTime: 240_000,       // 4 minutes
  },
  
  // Disabled automatic refresh (manual refresh only)
  MANUAL: {
    refetchInterval: false,
    staleTime: Infinity,
  },
  
  // Specific data type configurations (for backwards compatibility)
  // Critical data
  quotes: {
    refetchInterval: 15_000, // 15 seconds
    staleTime: 10_000,      // 10 seconds
  },
  positions: {
    refetchInterval: 15_000, // 15 seconds
    staleTime: 10_000,      // 10 seconds
  },
  orders: {
    refetchInterval: 15_000, // 15 seconds
    staleTime: 10_000,      // 10 seconds
  },
  paperAccount: {
    refetchInterval: 15_000, // 15 seconds
    staleTime: 10_000,      // 10 seconds
  },
  paperPositions: {
    refetchInterval: 15_000, // 15 seconds
    staleTime: 10_000,      // 10 seconds
  },
  autoloopStatus: {
    refetchInterval: 5_000,  // 5 seconds - more critical
    staleTime: 3_000,       // 3 seconds
  },
  
  // Operational data
  trades: {
    refetchInterval: 8_000,  // 8 seconds (7-10s range)
    staleTime: 6_000,       // 6 seconds
  },
  decisions: {
    refetchInterval: 8_000,  // 8 seconds (7-10s range)
    staleTime: 6_000,       // 6 seconds
  },
  strategies: {
    refetchInterval: 60_000, // 60 seconds as per spec
    staleTime: 45_000,      // 45 seconds
  },
  portfolio: {
    refetchInterval: 30_000, // 30 seconds
    staleTime: 20_000,      // 20 seconds
  },
  health: {
    refetchInterval: 30_000, // 30 seconds
    staleTime: 20_000,      // 20 seconds
  },
  safetyStatus: {
    refetchInterval: 30_000, // 30 seconds
    staleTime: 20_000,      // 20 seconds
  },
  
  // Analytics data
  context: {
    refetchInterval: 45_000, // 45 seconds as per spec
    staleTime: 35_000,      // 35 seconds
  },
  marketRegime: {
    refetchInterval: 60_000, // 1 minute
    staleTime: 45_000,      // 45 seconds
  },
  volatility: {
    refetchInterval: 60_000, // 1 minute
    staleTime: 45_000,      // 45 seconds
  },
  sentiment: {
    refetchInterval: 60_000, // 1 minute
    staleTime: 45_000,      // 45 seconds
  },
  performance: {
    refetchInterval: 60_000, // 1 minute
    staleTime: 45_000,      // 45 seconds
  },
  metrics: {
    refetchInterval: 30_000, // 30 seconds as per spec
    staleTime: 20_000,      // 20 seconds
  },
  
  // Research data
  evo: {
    refetchInterval: 90_000, // 1.5 minutes
    staleTime: 60_000,      // 1 minute
  },
  evolution: {
    refetchInterval: 90_000, // 1.5 minutes
    staleTime: 60_000,      // 1 minute
  },
  experiments: {
    refetchInterval: 90_000, // 1.5 minutes
    staleTime: 60_000,      // 1 minute
  },
  brainStatus: {
    refetchInterval: 90_000, // 1.5 minutes
    staleTime: 60_000,      // 1 minute
  },
  pipeline: {
    refetchInterval: 90_000, // 1.5 minutes
    staleTime: 60_000,      // 1 minute
  },
  
  // Static data
  news: {
    refetchInterval: 300_000, // 5 minutes
    staleTime: 240_000,       // 4 minutes
  },
  history: {
    refetchInterval: 300_000, // 5 minutes
    staleTime: 240_000,       // 4 minutes
  },
  report: {
    refetchInterval: 300_000, // 5 minutes
    staleTime: 240_000,       // 4 minutes
  },
  diamonds: {
    refetchInterval: 300_000, // 5 minutes
    staleTime: 240_000,       // 4 minutes
  },
  
  // Missing properties that components are trying to access
  evoHistory: {
    refetchInterval: 90_000, // 1.5 minutes
    staleTime: 60_000,      // 1 minute
  }
};

// Helper function to get config by data type
export function getRefreshConfig(dataType: keyof typeof DATA_REFRESH_CONFIG) {
  return DATA_REFRESH_CONFIG[dataType];
}

// Safe getter with fallback values
export function getSafeRefreshConfig(dataType: string) {
  const config = (DATA_REFRESH_CONFIG as any)[dataType];
  if (!config) {
    // Return sensible defaults based on category
    if (dataType.includes('paper') || dataType.includes('quote') || dataType.includes('position')) {
      return { refetchInterval: 15000, staleTime: 10000 }; // Critical data
    } else if (dataType.includes('evo') || dataType.includes('experiment')) {
      return { refetchInterval: 90000, staleTime: 60000 }; // Research data
    } else if (dataType.includes('news') || dataType.includes('history')) {
      return { refetchInterval: 300000, staleTime: 240000 }; // Static data
    } else {
      return { refetchInterval: 30000, staleTime: 20000 }; // Default operational
    }
  }
  return config;
}

// SSE (Server-Sent Events) configuration
export const SSE_CONFIG = {
  reconnectInterval: 5000,      // 5 seconds base reconnect delay
  maxReconnectAttempts: 10,     // Maximum reconnection attempts
  heartbeatInterval: 30000,     // 30 seconds heartbeat
  connectionTimeout: 10000,     // 10 seconds connection timeout
};

// WebSocket configuration
export const WEBSOCKET_CONFIG = {
  reconnectInterval: 2000,      // 2 seconds base reconnect delay
  maxReconnectAttempts: 15,     // Maximum reconnection attempts
  heartbeatInterval: 30000,     // 30 seconds heartbeat
  pongTimeout: 15000,           // 15 seconds pong timeout
  connectionTimeout: 10000,     // 10 seconds connection timeout
};

// Data type categorization for reference
export const DATA_TYPE_CATEGORIES = {
  CRITICAL: [
    'quotes',
    'positions',
    'orders',
    'paperPositions',
    'paperAccount',
    'autoloopStatus'
  ],
  OPERATIONAL: [
    'trades',
    'decisions',
    'strategies',
    'portfolio',
    'health',
    'safetyStatus'
  ],
  ANALYTICS: [
    'context',
    'marketRegime',
    'volatility',
    'sentiment',
    'performance',
    'metrics'
  ],
  RESEARCH: [
    'evo',
    'evolution',
    'experiments',
    'brainStatus',
    'pipeline'
  ],
  STATIC: [
    'news',
    'history',
    'report',
    'diamonds'
  ]
};
