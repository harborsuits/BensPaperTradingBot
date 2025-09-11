/**
 * Options Trading Types and Interfaces for EvoTester
 * Extends Poor-Capital Mode with options-specific risk management
 */

export interface OptionContract {
  symbol: string;
  underlyingSymbol: string;
  type: 'call' | 'put';
  strike: number;
  expiry: Date;
  bid: number;
  ask: number;
  last: number;
  volume: number;
  openInterest: number;
  greeks: OptionGreeks;
  iv: number;
  ivRank: number; // 0-100 percentile
}

export interface OptionGreeks {
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  rho: number;
}

export interface OptionChain {
  underlying: string;
  underlyingPrice: number;
  expiry: Date;
  calls: OptionContract[];
  puts: OptionContract[];
  timestamp: Date;
  quality: ChainQuality;
}

export interface ChainQuality {
  spreadScore: number; // 0-1, lower is better
  volumeScore: number; // 0-1, higher is better
  oiScore: number; // 0-1, higher is better
  overall: number; // 0-1 composite score
}

export interface OptionsPositionInput extends PositionSizingInput {
  optionType: 'vertical' | 'long_call' | 'long_put' | 'strangle' | 'straddle';
  longStrike?: number;
  shortStrike?: number;
  expiry: Date;
  ivRank: number;
  expectedMove: number;
  chainQuality: ChainQuality;
  frictionBudget: number; // max (fees + slip)/premium
}

export interface OptionsPositionResult extends PositionSizingResult {
  contracts: number;
  premium: number;
  breakeven: number;
  maxLoss: number;
  maxGain: number;
  greeks: {
    netDelta: number;
    netGamma: number;
    netTheta: number;
    netVega: number;
  };
  friction: {
    estimatedSlippage: number;
    fees: number;
    ratio: number; // (fees + slip)/premium
  };
  riskMetrics: {
    thetaDay: number; // daily theta decay
    gammaRisk: number; // gamma exposure
    vegaDollar: number; // $ exposure to 1% IV change
    deltaDollar: number; // $ exposure to $1 underlying move
  };
}

export interface RouteSelectionContext {
  ivRank: number;
  expectedMove: number;
  trendStrength: number;
  rvol: number;
  chainQuality: number;
  frictionHeadroom: number; // how much friction budget left
  thetaHeadroom: number; // how much theta budget left
  vegaHeadroom: number; // how much vega budget left
  availableRoutes: OptionRoute[];
}

export interface OptionRoute {
  type: 'vertical' | 'long_call' | 'long_put' | 'equity' | 'leveraged_etf';
  confidence: number; // 0-1
  expectedPnL: number;
  maxLoss: number;
  frictionRatio: number;
  thetaDay: number;
  context: string; // why this route was chosen
}

export interface ContextualBanditReward {
  route: string;
  context: RouteSelectionContext;
  outcome: {
    pnl: number;
    friction: number;
    drawdown: number;
    duration: number; // days held
  };
  timestamp: Date;
}

export interface OptionsFitnessMetrics extends CapitalEfficiencyMetrics {
  thetaCollected: number;
  vegaPnl: number;
  gammaPnl: number;
  frictionRatio: number;
  ivRank: number;
  chainQuality: number;
  assignmentRisk: number; // risk of early assignment
}

export interface OptionsTradeDecision {
  route: OptionRoute;
  position: OptionsPositionResult;
  rationale: {
    ivBias: string;
    expectedMoveCheck: string;
    frictionCheck: string;
    greeksCheck: string;
    chainQuality: string;
  };
  exitPlan: {
    profitTarget: number;
    timeStop: Date;
    volCrushStop: number;
    eventRules: string[];
  };
  trace: {
    secondBestRoute: OptionRoute;
    scoreDelta: number;
    chainSnapshot: Partial<OptionChain>;
    riskMetrics: OptionsPositionResult['riskMetrics'];
  };
}

// Options-specific configuration extending Poor-Capital Mode
export interface OptionsConfig {
  routes: {
    enabled: ('vertical' | 'long_call' | 'long_put')[];
    ivRankVerticalBias: number; // 0.6 = prefer verticals when IVR > 60%
    expectedMoveFactor: number; // 0.75 = target must be <= 75% of expected move
  };
  friction: {
    rejectThreshold: number; // 0.20 = reject if friction > 20%
    penaltyThreshold: number; // 0.12 = penalty if friction > 12%
    perContractFee: number; // $0.65 per contract
    slippageModel: 'conservative' | 'realistic' | 'aggressive';
  };
  greeks: {
    gammaPerTradeMax: { [key: string]: number }; // per underlying class
    thetaGovernor: {
      warnThreshold: number; // 0.002 = 0.2% of equity
      clampThreshold: number; // 0.0025 = 0.25% of equity
      consecutiveDays: number; // 2
    };
    vegaBudgetSoftPct: number; // 0.20 = 20% of equity
  };
  events: {
    earningsPolicy: {
      holdOnlyIf: string[]; // ['durable_catalyst']
      maxIVRankHold: number; // 0.80
    };
    exDivPolicy: 'never_hold_short_calls' | 'auto_close' | 'convert_to_long';
  };
  chain: {
    minOI: number; // 100
    minVolume: number; // 50
    maxSpreadBps: number; // 50
    freshnessSeconds: number; // 300 = 5 minutes
  };
}
