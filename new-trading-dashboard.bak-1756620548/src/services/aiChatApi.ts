import axios, { AxiosRequestConfig } from 'axios';

// Helper function to make API requests
async function apiRequest<T>(config: AxiosRequestConfig): Promise<{ success: boolean; data?: T; error?: string }> {
  try {
    const response = await axios({
      baseURL: '/api',
      timeout: 15000,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
      },
      ...config
    });
    return { success: true, data: response.data };
  } catch (error: any) {
    return { 
      success: false, 
      error: error.response?.data?.detail || error.message || 'Unknown error' 
    };
  }
}
import { ChatMessage } from '@/hooks/useAIChatWebSocket';

export interface ChatHistoryResponse {
  messages: ChatMessage[];
  hasMore: boolean;
}

// Define comprehensive market information structure
export interface MarketInfo {
  regime?: string;
  regimeConfidence?: number;
  volatility?: number;
  sentiment?: number;
  sentimentFactors?: {
    positive: { factor: string; impact: number }[];
    negative: { factor: string; impact: number }[];
  };
  keyLevels?: Record<string, number[]>;
  majorIndices?: Record<string, { price: number; change: number; volume?: number }>;
  featureIndicators?: {
    name: string;
    value: number;
    trend: 'up' | 'down' | 'neutral';
    relevance: number;
  }[];
  anomalies?: {
    type: string;
    severity: number;
    description: string;
    timestamp: string;
  }[];
}

// Define system health info
export interface SystemHealth {
  memoryUsage?: number;
  cpuUsage?: number;
  apiLatency?: number;
  errorRate?: number;
  dataQuality?: {
    marketDataFreshness?: number;
    missingDataPoints?: number;
    dataSources?: string[];
  };
}

// Enhanced trading context interface with comprehensive orchestrator awareness
export interface TradingContext {
  // Basic context information
  currentTab: string;
  symbol?: string;
  assetClasses: string[];
  timeframe?: string;
  contextPrompt?: string; // AI-friendly context summary
  
  // Portfolio information
  portfolioValue?: number;
  openPositions?: any[];
  recentTrades?: any[];
  performance?: {
    daily?: number;
    weekly?: number;
    monthly?: number;
    ytd?: number;
    byAssetClass?: {
      stocks?: number;
      options?: number;
      crypto?: number;
      forex?: number;
    };
    byStrategy?: Record<string, number>;
  };
  portfolioAllocation?: {
    stocks?: number;
    options?: number;
    crypto?: number;
    forex?: number;
    cash?: number;
  };
  riskMetrics?: {
    var?: number; // Value at Risk
    maxDrawdown?: number;
    beta?: number;
    correlation?: number;
    sharpe?: number;
  };
  
  // Market context
  market?: MarketInfo;
  
  // Trading opportunities and signals
  opportunities?: any[];
  strategies?: string[];
  topStrategies?: string[];
  alerts?: any[];
  tradingPatterns?: string[];
  externalSignals?: string[];
  
  // Execution information
  activeOrders?: number;
  advancedOrders?: string[];
  
  // System information
  systemHealth?: SystemHealth;
  dataQuality?: {
    marketDataFreshness?: number;
    missingDataPoints?: number;
    dataSources?: string[];
    sourceStatus?: Record<string, string>;
    dataLag?: Record<string, number>;
  };
  systemStatus?: {
    healthy?: boolean;
    stagingActive?: boolean;
    dataQuality?: boolean;
    circuitBreakersActive?: boolean;
  };
  
  // Broker intelligence
  brokerRecommendations?: string[];
  
  // News and analysis
  news?: {
    headline?: string;
    impact?: number;
    action?: string;
  }[];
  
  // Backtesting and evolution
  backtestingResults?: {
    id?: string;
    strategy?: string;
    symbols?: string[];
    timeframe?: string;
    returns?: number;
    sharpe?: number;
    winRate?: number;
  }[];
  evolutionaryTesting?: {
    activeRuns?: number;
    recentFindings?: any[];
  };
  
  // Additional data
  symbolData?: any; // Symbol-specific information
  
  // For backward compatibility
  [key: string]: any;
}

export const aiChatApi = {
  /**
   * Send a message to the AI assistant
   * @param message User message content
   * @param contextData Optional trading context data
   */
  async sendMessage(message: string, contextData?: TradingContext) {
    return apiRequest<{ messageId: string }>({
      url: '/ai/chat/send',
      method: 'POST',
      data: {
        message,
        context: contextData || {}
      }
    });
  },

  /**
   * Get chat message history
   * @param limit Optional limit of messages to retrieve
   * @param before Optional timestamp to retrieve messages before
   */
  async getHistory(limit: number = 50, before?: string) {
    return apiRequest<ChatHistoryResponse>({
      url: '/ai/chat/history',
      method: 'GET',
      params: {
        limit,
        before
      }
    });
  },

  /**
   * Get current trading context data
   */
  async getTradingContext() {
    return apiRequest<TradingContext>({
      url: '/ai/context',
      method: 'GET'
    });
  },

  /**
   * Clear chat history
   */
  async clearHistory() {
    return apiRequest<{ success: boolean }>({
      url: '/ai/chat/clear',
      method: 'POST'
    });
  }
};

export default aiChatApi;
