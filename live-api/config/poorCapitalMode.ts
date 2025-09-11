/**
 * Poor-Capital Mode Configuration
 *
 * Realistic trading constraints for small accounts ($500-$5K)
 * Three-ring approach: Explore → Validate → Allocate
 */

export const POOR_CAPITAL_MODE = {
  enabled: true,

  // Universe constraints (no-traps baseline)
  universe: {
    listedOnly: true,
    minPrice: 1,
    maxPrice: 10,
    minDollarADV: 10_000_000, // $10M daily volume
    maxSpreadBps: 20, // 0.20% max spread
    minFloat: 10_000_000, // 10M shares
    skipHaltWindowMin: 30, // Skip symbols within 30 min of halt/T12
    allowSSR: true, // Allow SSR days for long-only
  },

  // Catalyst scoring (40% news, 25% RVOL, 15% gap, 10% SI, 10% context)
  catalysts: {
    w_news: 0.40, // News impact
    w_rvol: 0.25, // Relative volume
    w_gap: 0.15, // Gap quality
    w_si: 0.10, // Short interest/float
    w_context: 0.10, // Insider/ownership/context
    minScore: 0.6, // Must score ≥60% to enter Evo
  },

  // Position sizing (0.7% risk, realistic constraints)
  risk: {
    perTradeRiskPct: 0.007, // 0.7% per trade
    maxPositionNotionalPct: 0.10, // ≤10% of capital
    maxOpenRiskPct: 0.015, // ≤1.5% total open risk
    maxDailyLossPct: 0.01, // 1% daily loss limit
  },

  // Execution rules (protect against slippage/pick-off)
  execution: {
    protectBps: 10, // 0.10% protection on limit orders
    maxSlippageBps: 15, // Block if expected slip > 0.15%
    blockOpenCloseMin: 15, // No new entries first/last 15 min
    wholeSharesMin: 2, // Minimum 2 shares
    minStopDistanceBps: 150, // Min 1.5% stop distance from spread
  },

  // EVO sandbox (micro-allocations with thermostat)
  evoPool: {
    capPctRange: [0.03, 0.06], // 3-6% of paper equity
    perStrategyCapUSD: [25, 100], // $25-100 per strategy
    maxConcurrent: 6, // Max 6 active bots
    ttlDays: 7, // 7-day TTL
    stepLimitPct: 0.20, // ±20% allocation changes
    correlationHintThreshold: 0.75, // Warn if corr >75%
    allowOvernightIf: 'durable_catalyst', // Earnings/FDA/contract tags
  },

  // Promotion gates (strict but realistic for small samples)
  promotion: {
    minTrades: 30, // Need 30 trades for statistical significance
    minSharpe: 1.1, // Sharpe ≥1.1
    maxDD: 0.10, // Max DD ≤10%
    minWin: 0.52, // Win rate ≥52%
    traceCompleteness: 0.98, // DecisionTrace ≥98%
    wfMaxDrop: 0.15, // Walk-forward drop ≤15%
  },

  // Shadow exploration (10% of compute for innovation)
  shadowExplore: {
    enabled: true,
    computeShare: 0.1, // 10% of Evo compute budget
    unconstrainedUniverses: [
      { name: 'mid-cap', minPrice: 10, maxPrice: 50, minADV: 25_000_000 },
      { name: 'large-cap', minPrice: 50, maxPrice: 200, minADV: 100_000_000 },
      { name: 'high-vol', minADV: 50_000_000, volatilityFilter: 'top_20pct' },
    ],
  },

  // Advanced guards (PDT, sector crowding, ADV participation)
  advancedGuards: {
    pdtGuard: {
      maxDayTradesPer5d: 2,
      preferSwing: true,
      blockDayTradesIfNearLimit: true,
    },
    sectorMaxActive: 2, // Max 2 bots per sector
    advParticipationMax: 0.0005, // 0.05% of daily volume
    ssrHandling: {
      raiseMinStopDistance: 2.0, // 2.0x spread for SSR days
      halveMaxSize: true,
    },
    drawdownGovernor: {
      pauseNewEntriesThreshold: 0.007, // Pause at 0.7% PnL
      resumeThreshold: 0.003, // Resume at 0.3% PnL
    },
  },

  // Fitness enhancements (capital efficiency, slippage penalties)
  fitnessEnhancements: {
    capitalEfficiencyWeight: 0.3, // 30% weight on P&L per $ risked
    slippagePenaltyWeight: 0.2, // 20% penalty for slippage costs
    advParticipationPenalty: 0.1, // 10% penalty for moving markets
    wholeShareRoundingPenalty: 0.1, // 10% penalty for odd-lot rounding
    correlationDiversityBonus: 0.1, // 10% bonus for uncorrelated strategies
  },
} as const;

// Type exports for TypeScript
export type PoorCapitalModeConfig = typeof POOR_CAPITAL_MODE;
export type UniverseConstraints = typeof POOR_CAPITAL_MODE.universe;
export type CatalystWeights = typeof POOR_CAPITAL_MODE.catalysts;
export type RiskParameters = typeof POOR_CAPITAL_MODE.risk;
export type ExecutionRules = typeof POOR_CAPITAL_MODE.execution;
export type EvoPoolSettings = typeof POOR_CAPITAL_MODE.evoPool;
export type PromotionGates = typeof POOR_CAPITAL_MODE.promotion;
