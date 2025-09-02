import axios, { AxiosError, AxiosRequestConfig } from 'axios';
import { 
  ApiResponse, 
  AuthRequest, 
  AuthResponse,
  MarketContext,
  NewsItem,
  Strategy,
  TradeCandidate,
  Position,
  PortfolioSummary,
  Trade,
  LogEvent,
  AlertNotification,
  EvoTesterConfig,
  EvoTesterProgress,
  EvoStrategy,
  DataSourceStatusModel,
  IngestionMetricsModel,
  DataStatusSummary,
  PriceUpdateModel,
  MarketDataBatchModel,
} from '@/types/api.types';
import { 
  SafetyStatus, 
  CircuitBreakerConfig, 
  CooldownConfig, 
  RiskLimitsConfig 
} from './safetyApi';

// Helpers
function isAxios404(error: unknown): boolean {
  const e = error as AxiosError;
  return !!(e?.response && e.response.status === 404);
}

function isSkippableStatus(error: unknown): boolean {
  const e = error as AxiosError;
  const code = e?.response?.status;
  // Treat transient/fail-closed statuses as skippable to try the next request
  return code === 404 || code === 503 || code === 502 || code === 429;
}

async function tryRequests<T>(requests: Array<() => Promise<ApiResponse<T>>>, fallback: T): Promise<ApiResponse<T>> {
  for (const req of requests) {
    try {
      const res = await req();
      if (res.success) return res;
    } catch (err) {
      if (isSkippableStatus(err)) {
        // try next request in sequence
        continue;
      }
      return { success: false, error: (err as Error).message } as ApiResponse<T>;
    }
  }
  return { success: true, data: fallback } as ApiResponse<T>;
}

// Create axios instance with default config
const isDev = import.meta.env.DEV;
export const api = axios.create({
  // In dev, force same-origin '/api' so MSW and Vite proxy work reliably
  baseURL: isDev
    ? '/api'
    : (import.meta.env.VITE_API_URL ? `${import.meta.env.VITE_API_URL}/api` : '/api'),
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for authentication
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  
  if (token && config.headers) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  
  return config;
});

import { applyApiInterceptors } from './apiInterceptor';

// Apply our enhanced interceptors with better 404 handling
applyApiInterceptors(api);

// Helper function to make API requests
async function apiRequest<T>(config: AxiosRequestConfig): Promise<ApiResponse<T>> {
  try {
    const response = await api(config);
    return { success: true, data: response.data };
  } catch (error) {
    const axiosError = error as AxiosError;
    return { 
      success: false, 
      error: axiosError.response?.data?.detail || axiosError.message || 'Unknown error' 
    };
  }
}

// Authentication endpoints
export const authApi = {
  login: async (credentials: AuthRequest): Promise<ApiResponse<AuthResponse>> => {
    const formData = new URLSearchParams();
    formData.append('username', credentials.username);
    formData.append('password', credentials.password);
    
    return apiRequest<AuthResponse>({
      url: '/auth/token',
      method: 'POST',
      data: formData,
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });
  },
  
  logout: async (): Promise<ApiResponse<null>> => {
    const response = await apiRequest<null>({
      url: '/auth/logout',
      method: 'POST',
    });
    
    if (response.success) {
      localStorage.removeItem('auth_token');
    }
    
    return response;
  },
};

// Context endpoints
export const contextApi = {
  getMarketContext: async () =>
    tryRequests<MarketContext>(
      [
        // Prefer canonical /context first
        () => apiRequest<MarketContext>({ url: '/context', method: 'GET' }),
        // Fallback to /news if /context is unavailable
        () => apiRequest<any>({ url: '/news', method: 'GET' }) as any,
      ],
      {} as unknown as MarketContext
    ),
  
  getMarketFeatures: () => 
    apiRequest<Record<string, { value: number, change: number }>>({ url: '/context/features', method: 'GET' }),
  
  getNews: (limit = 10) => 
    tryRequests<NewsItem[]>(
      [
        () => apiRequest<NewsItem[]>({ url: `/context/news?limit=${limit}`, method: 'GET' }),
        () => apiRequest<NewsItem[]>({ url: `/news?limit=${limit}`, method: 'GET' }),
      ],
      []
    ),
    
  getNewsForSymbol: (symbol: string, limit = 10) => 
    tryRequests<NewsItem[]>(
      [
        () => apiRequest<NewsItem[]>({ url: `/context/news/${symbol}?limit=${limit}`, method: 'GET' }),
        () => apiRequest<NewsItem[]>({ url: `/news/${symbol}?limit=${limit}`, method: 'GET' }),
      ],
      []
    ),
};

// Import Zod schema for strategies
import { Strategy, parseStrategyList } from '@/schemas/strategy';

// Strategy endpoints
export const strategyApi = {
  // Thin aliases
  list: () => 
    apiRequest<Strategy[]>({ url: '/strategies', method: 'GET' }),

  getStrategies: async () => {
    const response = await apiRequest<unknown>({ url: '/strategies', method: 'GET' });
    
    if (response.success && response.data) {
      // Use Zod schema to validate and transform data
      const strategies = parseStrategyList(response.data);
      return { success: true, data: strategies };
    }
    
    return response as ApiResponse<Strategy[]>;
  },
  
  getStrategy: async (id: string) => {
    try {
      const response = await apiRequest<unknown>({ url: `/strategies/${id}`, method: 'GET' });
      
      if (response.success && response.data) {
        // Validate single strategy
        const parsed = Strategy.safeParse(response.data);
        if (parsed.success) {
          return { success: true, data: parsed.data };
        }
      }
      
      return response as ApiResponse<Strategy>;
    } catch (error) {
      // Fallback to safe default
      return { 
        success: true, 
        data: Strategy.parse({ 
          id, 
          name: 'Unknown Strategy', 
          status: 'idle'
        })
      };
    }
  },
    
  getActiveStrategies: () => 
    tryRequests<Strategy[]>(
      [
        // Prefer list first
        () => apiRequest<Strategy[]>({ url: '/strategies', method: 'GET' }),
        () => apiRequest<Strategy[]>({ url: '/strategies/active', method: 'GET' }),
      ],
      []
    ),
    
  getRankedStrategies: (metric: string = 'overall', limit: number = 10) => 
    apiRequest<Strategy[]>({ url: `/strategies/ranked?metric=${metric}&limit=${limit}`, method: 'GET' }),

  // Optional endpoints used by ranking widgets
  ranking: async () => {
    const primary = await apiRequest<Strategy[]>({ url: '/strategies/ranking', method: 'GET' });
    if (primary.success) return primary;
    const fallback = await apiRequest<Strategy[]>({ url: '/strategies', method: 'GET' });
    if (fallback.success && fallback.data) {
      const sorted = [...fallback.data].sort((a: any, b: any) => (b.score ?? b.priority_score ?? 0) - (a.score ?? a.priority_score ?? 0));
      return { success: true, data: sorted } as ApiResponse<Strategy[]>;
    }
    return primary;
  },

  promote: (id: string | number) => 
    apiRequest<{ success: boolean; message: string }>({ url: '/safety/promote', method: 'POST', data: { strategy_id: id } }),

  demote: (id: string | number) => 
    apiRequest<{ success: boolean; message: string }>({ url: '/safety/demote', method: 'POST', data: { strategy_id: id } }),
    
  enableStrategy: (id: string) => 
    apiRequest<Strategy>({ url: `/strategies/${id}/enable`, method: 'POST' }),
    
  disableStrategy: (id: string) => 
    apiRequest<Strategy>({ url: `/strategies/${id}/disable`, method: 'POST' }),
};

// Trade decision endpoints
export const decisionApi = {
  getLatestDecisions: () => 
    tryRequests<TradeCandidate[]>(
      [
        // Prefer decisions first to avoid fail-closed /trades 503 noise until first trade exists
        () => apiRequest<TradeCandidate[]>({ url: '/decisions', method: 'GET' }),
        () => apiRequest<TradeCandidate[]>({ url: '/decisions/latest', method: 'GET' }),
        () => apiRequest<any[]>({ url: '/trades', method: 'GET' }) as any,
      ],
      []
    ),
    
  getDecisionsHistory: (date: string) => 
    tryRequests<TradeCandidate[]>(
      [
        () => apiRequest<TradeCandidate[]>({ url: `/decisions?date=${date}`, method: 'GET' }),
        () => apiRequest<TradeCandidate[]>({ url: `/trades?date=${date}`, method: 'GET' }),
      ],
      []
    ),
};

// Portfolio endpoints
export const portfolioApi = {
  getPortfolio: (account: 'live' | 'paper') => 
    tryRequests<{ summary: PortfolioSummary, positions: Position[] }>(
      [
        // Prefer /portfolio?mode= first
        () => apiRequest<any>({ url: `/portfolio`, method: 'GET', params: { mode: account } }) as any,
        () => apiRequest<any>({ url: `/portfolio`, method: 'GET' }) as any,
        () => apiRequest<{ summary: PortfolioSummary, positions: Position[] }>({ url: `/portfolio/${account}`, method: 'GET' }),
      ],
      { summary: { total_equity: 0, cash_balance: 0, daily_pl: 0, daily_pl_percent: 0 } as unknown as PortfolioSummary, positions: [] as Position[] }
    ),
    
  getTrades: (account: 'live' | 'paper', limit = 50) => 
    tryRequests<Trade[]>(
      [
        async () => {
          // Primary: matches backend trades router (account query param)
          const r = await apiRequest<any>({ url: `/trades?limit=${limit}&account=${account}`, method: 'GET' });
          if (r?.success) {
            const data = r.data;
            const items = Array.isArray(data)
              ? data
              : Array.isArray(data?.items)
                ? data.items
                : Array.isArray(data?.data)
                  ? data.data
                  : Array.isArray(data?.trades)
                    ? data.trades
                    : [];
            return { success: true, data: items } as ApiResponse<Trade[]>;
          }
          return r as ApiResponse<Trade[]>;
        },
        async () => {
          // Fallback: alternate param name some backends use
          const r = await apiRequest<any>({ url: `/trades?limit=${limit}&mode=${account}`, method: 'GET' });
          if (r?.success) {
            const data = r.data;
            const items = Array.isArray(data)
              ? data
              : Array.isArray(data?.items)
                ? data.items
                : Array.isArray(data?.data)
                  ? data.data
                  : Array.isArray(data?.trades)
                    ? data.trades
                    : [];
            return { success: true, data: items } as ApiResponse<Trade[]>;
          }
          return r as ApiResponse<Trade[]>;
        },
        async () => {
          const r = await apiRequest<any>({ url: `/trades?limit=${limit}`, method: 'GET' });
          if (r?.success) {
            const data = r.data;
            const items = Array.isArray(data)
              ? data
              : Array.isArray(data?.items)
                ? data.items
                : Array.isArray(data?.data)
                  ? data.data
                  : Array.isArray(data?.trades)
                    ? data.trades
                    : [];
            return { success: true, data: items } as ApiResponse<Trade[]>;
          }
          return r as ApiResponse<Trade[]>;
        },
        async () => {
          const r = await apiRequest<any>({ url: `/trades`, method: 'GET' });
          if (r?.success) {
            const data = r.data;
            const items = Array.isArray(data)
              ? data
              : Array.isArray(data?.items)
                ? data.items
                : Array.isArray(data?.data)
                  ? data.data
                  : Array.isArray(data?.trades)
                    ? data.trades
                    : [];
            return { success: true, data: items } as ApiResponse<Trade[]>;
          }
          return r as ApiResponse<Trade[]>;
        },
      ],
      []
    ),
    
  closePosition: (account: 'live' | 'paper', symbol: string) => 
    apiRequest<null>({ 
      url: `/portfolio/${account}/positions/${symbol}/close`, method: 'POST' 
    }),
    
  placeOrder: (account: 'live' | 'paper', order: any) => 
    apiRequest<Trade>({ 
      url: `/portfolio/${account}/orders`, method: 'POST', data: order 
    }),

  getSafetyStatus: () => 
    apiRequest<SafetyStatus>({ 
      url: '/safety/status', method: 'GET' 
    }),

  setEmergencyStop: (active: boolean) => 
    apiRequest<{ success: boolean, message: string }>({ 
      url: '/safety/emergency-stop', method: 'POST', data: { active } 
    }),

  setTradingMode: (mode: 'live' | 'paper') => 
    apiRequest<{ success: boolean, message: string }>({ 
      url: '/safety/trading-mode', method: 'POST', data: { mode } 
    }),
};

// Data Ingestion endpoints
export const ingestionApi = {
  getSourcesStatus: async () => tryRequests<DataSourceStatusModel[]>([
    () => apiRequest<DataSourceStatusModel[]>({ url: '/data/sources/status', method: 'GET' }),
  ], []),
  getMetrics: async () => tryRequests<IngestionMetricsModel>([
    // Prefer same-origin /metrics in dev; fall back to /data/metrics
    async () => {
      const apiBase = import.meta.env.DEV ? '' : (import.meta.env.VITE_API_URL ? `${import.meta.env.VITE_API_URL}` : '');
      const direct = await axios.get(apiBase + '/metrics');
      return { success: true, data: direct.data } as ApiResponse<IngestionMetricsModel>;
    },
    () => apiRequest<IngestionMetricsModel>({ url: '/data/metrics', method: 'GET' }),
  ], {} as unknown as IngestionMetricsModel),
  getDataStatus: async () => {
    // Resilient metrics: JSON /metrics → Prometheus /metrics/prom → /data/status → safe fallback
    const apiBase = import.meta.env.DEV ? '' : (import.meta.env.VITE_API_URL ? `${import.meta.env.VITE_API_URL}` : '');
    const res = await tryRequests<DataStatusSummary>([
      // 1) JSON /metrics
      async () => {
        const r = await axios.get(apiBase + '/metrics');
        const m = r?.data ?? {};
        const now = new Date().toISOString();
        const data: DataStatusSummary = {
          timestamp: m.updated_at ?? m.ts ?? now,
          sources: [],
          metrics: {
            totalSymbolsTracked: 0,
            errorRate: 0,
            requestsLastHour: 0,
            averageLatency: 0,
            dataFreshSeconds: Number(
              m.benbot_data_fresh_seconds ?? m.data_fresh_seconds ?? m.fresh_seconds ?? 0
            ) || 0,
            wsConnected: Boolean(
              m.benbot_ws_connected ?? m.ws_connected ?? m.ws ?? false
            ),
          } as any,
        } as any;
        return { success: true, data } as ApiResponse<DataStatusSummary>;
      },
      // 2) Prometheus text /metrics/prom
      async () => {
        const r = await axios.get(apiBase + '/metrics/prom', { responseType: 'text' });
        const text: string = typeof r?.data === 'string' ? r.data : '';
        const findNum = (name: string) => {
          const m = text.match(new RegExp(`^${name}(\\{[^}]*\\})?\\s+(\\d+(?:\\.\\d+)?)$`, 'm'));
          return m ? Number(m[2]) : undefined;
        };
        const fresh = findNum('benbot_data_fresh_seconds') ?? 0;
        const ws = (findNum('benbot_ws_connected') ?? 0) > 0;
        const now = new Date().toISOString();
        const data: DataStatusSummary = {
          timestamp: now,
          sources: [],
          metrics: {
            totalSymbolsTracked: 0,
            errorRate: 0,
            requestsLastHour: 0,
            averageLatency: 0,
            dataFreshSeconds: fresh,
            wsConnected: ws,
          } as any,
        } as any;
        return { success: true, data } as ApiResponse<DataStatusSummary>;
      },
      // 3) Legacy JSON /data/status if present
      () => apiRequest<DataStatusSummary>({ url: '/data/status', method: 'GET' }),
    ], { timestamp: new Date().toISOString(), sources: [], metrics: {} as any } as DataStatusSummary);
    return res;
  },
  getLatestPrices: (symbols: string) => tryRequests<MarketDataBatchModel>([
    () => apiRequest<MarketDataBatchModel>({ url: `/data/prices?symbols=${symbols}`, method: 'GET' }),
  ], {} as unknown as MarketDataBatchModel),
};

// Logging endpoints
export const loggingApi = {
  getLogs: (level: string = 'INFO', limit = 100, offset = 0) =>
    tryRequests<LogEvent[]>(
      [
        async () => {
          const r = await apiRequest<any>({ url: `/logs?level=${level}&limit=${limit}&offset=${offset}`, method: 'GET' });
          if (r?.success) {
            const items = Array.isArray((r as any).data?.items)
              ? (r as any).data.items
              : (Array.isArray((r as any).data) ? (r as any).data : []);
            return { success: true, data: items as LogEvent[] } as ApiResponse<LogEvent[]>;
          }
          return r as ApiResponse<LogEvent[]>;
        },
        async () => {
          const r = await apiRequest<any>({ url: `/events/logs?level=${level}&limit=${limit}&offset=${offset}`, method: 'GET' });
          if (r?.success) {
            const items = Array.isArray((r as any).data?.items)
              ? (r as any).data.items
              : (Array.isArray((r as any).data) ? (r as any).data : []);
            return { success: true, data: items as LogEvent[] } as ApiResponse<LogEvent[]>;
          }
          return r as ApiResponse<LogEvent[]>;
        },
      ],
      []
    ),
    
  getAlerts: (severity?: string, limit = 20, acknowledged?: boolean) => {
    let url = `/alerts?limit=${limit}`;
    if (severity) url += `&severity=${severity}`;
    if (acknowledged !== undefined) url += `&acknowledged=${acknowledged}`;
    return apiRequest<AlertNotification[]>({ url, method: 'GET' });
  },
  
  acknowledgeAlert: (alertId: string) => 
    apiRequest<{status: string, message: string}>({ 
      url: `/alerts/${alertId}/acknowledge`, method: 'POST' 
    }),
};

// EvoTester endpoints
export const evoTesterApi = {
  startEvoTest: (config: EvoTesterConfig) => 
    apiRequest<{ session_id: string }>({ 
      url: '/evotester/start', method: 'POST', data: config 
    }),
    
  stopEvoTest: (sessionId: string) => 
    apiRequest<null>({ 
      url: `/evotester/${sessionId}/stop`, method: 'POST' 
    }),
    
  getEvoStatus: (sessionId: string) => 
    apiRequest<EvoTesterProgress>({ 
      url: `/evotester/${sessionId}/status`, method: 'GET' 
    }),
    
  getEvoResults: (sessionId: string) => 
    apiRequest<EvoStrategy[]>({ 
      url: `/evotester/${sessionId}/results`, method: 'GET' 
    }),
    
  getEvoHistory: () => 
    apiRequest<{id: string, date: string, bestFitness: number}[]>({ 
      url: '/evotester/history', method: 'GET' 
    }),
    
  pauseEvoTest: (sessionId: string) => 
    apiRequest<null>({ 
      url: `/evotester/${sessionId}/pause`, method: 'POST' 
    }),
    
  resumeEvoTest: (sessionId: string) => 
    apiRequest<null>({ 
      url: `/evotester/${sessionId}/resume`, method: 'POST' 
    }),
    
  promoteStrategy: (strategy: EvoStrategy) => 
    apiRequest<Strategy>({ 
      url: '/strategies', method: 'POST', data: strategy 
    }),
};

// Safety endpoints  
export const safetyApi = {
  getCircuitBreakerConfig: () => 
    apiRequest<CircuitBreakerConfig>({ 
      url: '/safety/circuit-breakers/config', method: 'GET' 
    }),
  
  updateCircuitBreakerConfig: (config: CircuitBreakerConfig) => 
    apiRequest<{ success: boolean, message: string }>({ 
      url: '/safety/circuit-breakers/config', method: 'PUT', data: config 
    }),
  
  getCooldownConfig: () => 
    apiRequest<CooldownConfig>({ 
      url: '/safety/cooldowns/config', method: 'GET' 
    }),
  
  updateCooldownConfig: (config: CooldownConfig) => 
    apiRequest<{ success: boolean, message: string }>({ 
      url: '/safety/cooldowns/config', method: 'PUT', data: config 
    }),
  
  getRiskLimitsConfig: () => 
    apiRequest<RiskLimitsConfig>({ 
      url: '/safety/risk-limits/config', method: 'GET' 
    }),
  
  updateRiskLimitsConfig: (config: RiskLimitsConfig) => 
    apiRequest<{ success: boolean, message: string }>({ 
      url: '/safety/risk-limits/config', method: 'PUT', data: config 
    }),
  
  resetCircuitBreaker: () => 
    apiRequest<{ success: boolean, message: string }>({ 
      url: '/safety/circuit-breakers/reset', method: 'POST' 
    }),
  
  resetCooldown: () => 
    apiRequest<{ success: boolean, message: string }>({ 
      url: '/safety/cooldowns/reset', method: 'POST' 
    }),
  
  getSafetyEvents: (limit = 50) => 
    apiRequest<Array<{
      id: string,
      type: 'emergency_stop' | 'circuit_breaker' | 'cooldown' | 'mode_change',
      action: 'activated' | 'deactivated' | 'triggered' | 'reset',
      timestamp: string,
      reason?: string,
      details?: Record<string, any>
    }>>({ 
      url: `/safety/events?limit=${limit}`, method: 'GET' 
    }),
};

export default {
  auth: authApi,
  context: contextApi,
  strategy: strategyApi,
  decision: decisionApi,
  portfolio: portfolioApi,
  logging: loggingApi,
  evoTester: evoTesterApi,
  ingestion: ingestionApi,
  safety: safetyApi,
};
