import api from './api';

// Use the apiRequest function that is defined in api.ts
const apiRequest = async <T>(config: any) => {
  return await api.strategy.apiRequest(config);
};

export interface Strategy {
  id: string;
  name: string;
  description: string;
  category: string;
  enabled: boolean;
  performance: {
    daily: number;
    weekly: number;
    monthly: number;
    overall: number;
  };
  marketSuitability: number;
  parameters: Record<string, any>;
  status: 'idle' | 'active' | 'paused' | 'error';
  currentPosition?: number;
  lastTradeTimestamp?: string;
  lastSignal?: {
    direction: 'long' | 'short' | 'neutral';
    strength: number;
    timestamp: string;
  };
}

export interface StrategyRanking {
  id: string;
  name: string;
  score: number;
  marketSuitability: number;
  currentRegime: string;
  performance: number;
  volatility: number;
  enabled: boolean;
  status: 'idle' | 'active' | 'paused' | 'error';
}

// Strategy API methods
export const strategyApi = {
  // Get all strategies
  async getStrategies() {
    return apiRequest<Strategy[]>({ 
      url: '/api/strategies', 
      method: 'GET' 
    });
  },
  
  // Get strategy by ID
  async getStrategy(id: string) {
    return apiRequest<Strategy>({ 
      url: `/api/strategies/${id}`, 
      method: 'GET' 
    });
  },
  
  // Get active strategies
  async getActiveStrategies() {
    return apiRequest<Strategy[]>({ 
      url: '/api/strategies/active', 
      method: 'GET' 
    });
  },
  
  // Get strategy rankings
  async getStrategyRankings() {
    return apiRequest<StrategyRanking[]>({ 
      url: '/api/strategies/rankings', 
      method: 'GET' 
    });
  },
  
  // Toggle strategy enabled status
  async toggleStrategy(id: string) {
    return apiRequest<{ success: boolean; strategy: Strategy }>({ 
      url: `/api/strategies/${id}/toggle`, 
      method: 'POST' 
    });
  },
  
  // Set strategy parameters
  async updateStrategyParameters(id: string, parameters: Record<string, any>) {
    return apiRequest<{ success: boolean; strategy: Strategy }>({ 
      url: `/api/strategies/${id}/parameters`, 
      method: 'POST',
      data: { parameters }
    });
  },
  
  // Get strategy performance
  async getStrategyPerformance(id: string, timeframe: 'day' | 'week' | 'month' | 'year' = 'month') {
    return apiRequest<any>({ 
      url: `/api/strategies/${id}/performance?timeframe=${timeframe}`, 
      method: 'GET' 
    });
  }
};
