import { ApiResponse } from '@/types/api.types';
import { api } from './api';

// Reuse the same apiRequest function as in api.ts
async function apiRequest<T>(config: any): Promise<ApiResponse<T>> {
  try {
    const response = await api(config);
    return { success: true, data: response.data };
  } catch (error: any) {
    return { 
      success: false, 
      error: error.response?.data?.detail || error.message || 'Unknown error' 
    };
  }
}

export type MarketRegime = 'bullish' | 'bearish' | 'neutral' | 'volatile' | 'recovery' | 'unknown';

export interface MarketRegimeData {
  regime: MarketRegime;
  confidence: number;
  since: string;
  description: string;
}

export interface SentimentData {
  overall_score: number; 
  market_sentiment: 'positive' | 'negative' | 'neutral';
  positive_factors: string[];
  negative_factors: string[];
  source: string;
  timestamp: string;
}

export interface SentimentHistoryItem {
  timestamp: string;
  score: number;
  sentiment: 'positive' | 'negative' | 'neutral';
  volume?: number;
}

export interface SentimentAnomaly {
  timestamp: string;
  description: string;
  score: number;
  expected_score: number;
  deviation: number;
  tags?: string[];
}

export interface NewsItem {
  id: string;
  headline: string;
  summary: string;
  url: string;
  source: string;
  published_at: string;
  sentiment_score: number;
  impact: 'high' | 'medium' | 'low';
  categories: string[];
  symbols?: string[];
}

export interface MarketFeature {
  name: string;
  value: number;
  previous: number;
  change: number;
  description: string;
  history: {timestamp: string, value: number}[];
}

export interface MarketAnomaly {
  id: string;
  description: string;
  severity: 'high' | 'medium' | 'low';
  detected_at: string;
  affected_symbols?: string[];
  category: string;
}

export interface AIPrediction {
  timeframe: string;
  direction: 'up' | 'down' | 'sideways';
  confidence: number;
  probability_up: number;
  probability_down: number;
  reasoning: string;
  model: string;
  generated_at: string;
}

export interface MarketContextData {
  market_regime: MarketRegimeData;
  sentiment: SentimentData;
  key_features: MarketFeature[];
  news: NewsItem[];
  anomalies: MarketAnomaly[];
  ai_predictions: AIPrediction[];
  last_updated: string;
}

export interface StrategyPerformanceMetric {
  sharpe_ratio: number;
  win_rate: number;
  avg_return: number;
  max_drawdown?: number;
  calmar_ratio?: number;
  sortino_ratio?: number;
}

export interface StrategyAllocationData {
  timestamp: string;
  allocation: Record<string, number>;
  market_context: {
    regime: MarketRegimeData;
    bias: string;
    volatility: string;
  };
  performance_weight: number;
  performance_metrics: Record<string, StrategyPerformanceMetric>;
}

export const contextApi = {
  /**
   * Fetch current market context data including regime, sentiment, etc.
   */
  async getMarketContext() {
    return apiRequest<MarketContextData>({
      url: '/context',
      method: 'GET'
    });
  },

  /**
   * Fetch only the latest news items
   */
  async getLatestNews(limit = 10) {
    return apiRequest<NewsItem[]>({
      url: `/context/news?limit=${limit}`,
      method: 'GET'
    });
  },

  /**
   * Fetch market regime data
   */
  async getMarketRegime(options?: { symbols?: string[], timeframe?: string, forceRefresh?: boolean }) {
    const params = new URLSearchParams();
    
    if (options?.symbols?.length) {
      params.append('symbols', options.symbols.join(','));
    }
    
    if (options?.timeframe) {
      params.append('timeframe', options.timeframe);
    }
    
    if (options?.forceRefresh) {
      params.append('refresh', 'true');
    }
    
    const queryString = params.toString() ? `?${params.toString()}` : '';
    return apiRequest<MarketRegimeData>({ 
      url: `/context/regime${queryString}`, 
      method: 'GET' 
    });
  },

  /**
   * Fetch sentiment analysis 
   */
  async getSentiment() {
    return apiRequest<SentimentData>({
      url: '/context/sentiment',
      method: 'GET'
    });
  },

  /**
   * Fetch market anomalies
   */
  async getAnomalies() {
    return apiRequest<MarketAnomaly[]>({
      url: '/context/anomalies',
      method: 'GET'
    });
  },

  /**
   * Fetch AI-generated market predictions
   */
  async getPredictions() {
    return apiRequest<AIPrediction[]>({
      url: '/context/predictions',
      method: 'GET'
    });
  },

  /**
   * Fetch specific market features
   */
  async getMarketFeatures() {
    return apiRequest<MarketFeature[]>({
      url: '/context/features',
      method: 'GET'
    });
  },

  /**
   * Fetch historical sentiment data
   */
  async getSentimentHistory(days = 30) {
    return apiRequest<SentimentHistoryItem[]>({
      url: `/context/sentiment/history?days=${days}`,
      method: 'GET'
    });
  },

  /**
   * Fetch sentiment anomalies
   */
  async getSentimentAnomalies(limit = 10) {
    return apiRequest<SentimentAnomaly[]>({
      url: `/context/sentiment/anomalies?limit=${limit}`,
      method: 'GET'
    });
  },
  
  /**
   * Fetch strategy allocation based on market context and performance
   */
  async getStrategyAllocation(options?: { performanceWeight?: number, forceRefresh?: boolean }) {
    const params = new URLSearchParams();
    
    if (options?.performanceWeight !== undefined) {
      params.append('performance_weight', options.performanceWeight.toString());
    }
    
    if (options?.forceRefresh) {
      params.append('refresh', 'true');
    }
    
    const queryString = params.toString() ? `?${params.toString()}` : '';
    return apiRequest<StrategyAllocationData>({
      url: `/context/strategy-allocation${queryString}`,
      method: 'GET'
    });
  }
};
