// Common response structure
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

// Authentication
export interface AuthRequest {
  username: string;
  password: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

// Market context
export interface MarketContext {
  timestamp: string;
  regime: {
    type: 'Bullish' | 'Bearish' | 'Neutral' | 'Volatile';
    confidence: number;
    description: string;
  };
  volatility: {
    value: number;
    change: number;
    classification: 'Low' | 'Medium' | 'High' | 'Extreme';
  };
  sentiment: {
    score: number;
    sources: string[];
    trending_words: string[];
  };
  features: Record<string, number>;
}

// News
export interface NewsItem {
  id: string;
  title: string;
  source: string;
  url: string;
  published_at: string;
  sentiment_score: number;
  symbols: string[];
  summary?: string;
  impact?: 'High' | 'Medium' | 'Low';
}

// Strategy
export interface Strategy {
  id: string;
  name: string;
  description: string;
  status: 'active' | 'idle' | 'testing' | 'disabled';
  priority_score: number;
  asset_class: 'stocks' | 'options' | 'forex' | 'crypto';
  last_signal_time?: string;
  performance: StrategyPerformance;
}

export interface StrategyPerformance {
  sharpe_ratio: number;
  win_rate: number;
  trades_count: number;
  avg_profit_per_trade: number;
  max_drawdown: number;
  period: string;
}

// Trade decision
export interface TradeCandidate {
  id: string;
  symbol: string;
  strategy_id: string;
  strategy_name: string;
  direction: 'buy' | 'sell';
  score: number;
  entry_price: number;
  target_price: number;
  stop_loss: number;
  potential_profit_pct: number;
  risk_reward_ratio: number;
  confidence: number;
  time_validity: string;
  timeframe: string;
  created_at: string;
  status: 'pending' | 'watching' | 'ready' | 'executed' | 'expired' | 'skipped';
  executed: boolean;
  reason?: string;
  tags: string[];
  entry_conditions: string[];
  indicators: {
    name: string;
    value: number;
    signal: 'bullish' | 'bearish' | 'neutral';
  }[];
}

// Portfolio
export interface Position {
  symbol: string;
  quantity: number;
  avg_cost: number;
  last_price: number;
  current_value: number;
  unrealized_pl: number;
  unrealized_pl_percent: number;
  realized_pl: number;
  account: 'live' | 'paper';
  strategy_id: string;
  entry_time: string;
}

export interface PortfolioSummary {
  total_equity: number;
  cash_balance: number;
  buying_power: number;
  daily_pl: number;
  daily_pl_percent: number;
  total_pl: number;
  total_pl_percent: number;
  positions_count: number;
  account: 'live' | 'paper';
  last_updated: string;
}

// Trade history
export interface Trade {
  id: string;
  symbol: string;
  action: 'buy' | 'sell';
  quantity: number;
  price: number;
  total_value: number;
  timestamp: string;
  strategy_id: string;
  strategy_name: string;
  strategy?: string; // Strategy name for display purposes
  account: 'live' | 'paper';
  status: 'filled' | 'pending' | 'canceled';
}

// Logs and alerts
export interface AlertNotification {
  id: string;
  timestamp: string;
  severity: 'critical' | 'warning' | 'info';
  source: string;
  message: string;
  details?: Record<string, any>;
  acknowledged: boolean;
  actionRequired?: boolean;
  actionTaken?: string;
}

export interface LogEvent {
  id: string;
  timestamp: string;
  level: 'INFO' | 'WARNING' | 'ERROR' | 'DEBUG';
  message: string;
  source: string;
  details?: Record<string, any>;
  acknowledged?: boolean;
  requires_action?: boolean;
  related_symbol?: string;
  category?: string;
}

export interface AlertNotification {
  id: string;
  timestamp: string;
  severity: 'critical' | 'warning' | 'info';
  title: string;
  message: string;
  source: string;
  acknowledged: boolean;
  action_taken: boolean;
  related_symbol?: string;
  details?: Record<string, any>;
}

// EvoTester
export interface EvoTesterProgress {
  sessionId: string;
  running: boolean;
  currentGeneration: number;
  totalGenerations: number;
  startTime: string;
  estimatedCompletionTime?: string;
  elapsedTime?: string;
  progress: number;
  bestFitness: number;
  averageFitness: number;
  status: 'running' | 'completed' | 'failed' | 'paused';
  errorMessage?: string;
}

export interface EvoStrategy {
  id?: string;
  name: string;
  description?: string;
  parameters: Record<string, any>;
  fitness: number;
  performance?: {
    sharpeRatio?: number;
    sortinoRatio?: number;
    winRate?: number;
    profitFactor?: number;
    maxDrawdown?: number;
  };
  tags?: string[];
  created?: string;
}

export interface EvoTesterConfig {
  population_size: number;
  generations: number;
  mutation_rate: number;
  crossover_rate: number;
  target_asset: string;
  optimization_metric: 'sharpe' | 'sortino' | 'profit' | 'win_rate';
}

export interface EvoTesterResult {
  session_id: string;
  current_generation: number;
  total_generations: number;
  best_fitness: number;
  avg_fitness: number;
  best_strategy?: EvoStrategy;
  status: 'running' | 'completed' | 'failed' | 'paused';
}

// WebSocket message types
export type WebSocketChannel = 
  'data' |
  'context' | 
  'strategies' | 
  'decisions' | 
  'trading' | 
  'portfolio' | 
  'logs' |
  'alerts' |
  'evotester' |
  'market_data'; // Added for real-time market data updates

export interface WebSocketMessage<T = any> {
  type: string;
  channel?: string;
  data: T;
  timestamp: string;
}

// Data Ingestion Models
export interface DataSourceStatusModel {
  id: string;
  name: string;
  type: string;
  status: string;
  lastUpdate: string;
  nextUpdateExpected?: string;
  healthScore: number;
  latency?: number;
  errorRate?: number;
  message?: string;
}

export interface IngestionMetricsModel {
  totalSymbolsTracked: number;
  activeSymbols: string[];
  symbolsWithErrors: string[];
  requestsLastHour: number;
  dataPointsIngested: number;
  lastFullSyncCompleted: string;
  averageLatency: number;
  errorRate: number;
}

export interface DataStatusSummary {
  timestamp: string;
  sources: DataSourceStatusModel[];
  metrics: IngestionMetricsModel;
}

export interface PriceUpdateModel {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  timestamp: string;
  source: string;
  bid?: number;
  ask?: number;
  high?: number;
  low?: number;
  open?: number;
  previousClose?: number;
  lastUpdated?: string;
}

export interface MarketDataBatchModel {
  prices: Record<string, PriceUpdateModel>;
  timestamp: string;
  batchId: string;
  source: string;
}

// Real-time market data update models
export interface PriceUpdate {
  symbol: string;
  price: number;
  timestamp: string;
  bid?: number;
  ask?: number;
  change?: number;
  changePercent?: number;
  volume?: number;
}

export interface MarketDataUpdate {
  updates: PriceUpdate[];
  timestamp: string;
  source: string;
}

// Trading Safety and Guardrails
export interface SafetyStatus {
  tradingMode: 'live' | 'paper';
  emergencyStopActive: boolean;
  circuitBreakers: {
    active: boolean;
    reason?: string;
    triggeredAt?: string;
    maxDailyLoss?: number;
    currentDailyLoss?: number;
    maxTradesPerDay?: number;
    currentTradeCount?: number;
  };
  cooldowns: {
    active: boolean;
    endsAt?: string;
    remainingSeconds?: number;
    reason?: string;
  };
}

export interface CircuitBreakerConfig {
  enabled: boolean;
  maxDailyLoss: number;
  maxDrawdownPercent: number;
  maxTradesPerDay: number;
  maxConsecutiveLosses: number;
}

export interface CooldownConfig {
  enabled: boolean;
  durationSeconds: number;
  afterConsecutiveLosses: number;
  afterMaxDrawdown: boolean;
}

export interface RiskLimitsConfig {
  maxPositionSizePercent: number;
  maxExposurePerSectorPercent: number;
  maxExposurePerAssetClassPercent: number;
}

export interface SafetyEvent {
  id: string;
  type: 'emergency_stop' | 'circuit_breaker' | 'cooldown' | 'mode_change';
  action: 'activated' | 'deactivated' | 'triggered' | 'reset';
  timestamp: string;
  reason?: string;
  details?: Record<string, any>;
}
