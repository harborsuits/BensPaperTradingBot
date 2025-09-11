/**
 * Route Selector for EvoTester
 * Uses contextual bandit learning to choose optimal trading routes
 */

import {
  RouteSelectionContext,
  OptionRoute,
  ContextualBanditReward,
  OptionsConfig
} from './optionsTypes';

interface RoutePerformance {
  route: string;
  totalTrades: number;
  totalPnL: number;
  avgPnL: number;
  winRate: number;
  avgFriction: number;
  avgDrawdown: number;
  lastUpdated: Date;
}

interface ContextualBanditModel {
  // Thompson sampling parameters for each route
  alpha: { [route: string]: number }; // Success count
  beta: { [route: string]: number };  // Failure count

  // Context weights for each feature
  contextWeights: {
    ivRank: { [route: string]: number };
    expectedMove: { [route: string]: number };
    trendStrength: { [route: string]: number };
    rvol: { [route: string]: number };
    chainQuality: { [route: string]: number };
  };

  lastUpdated: Date;
}

export class RouteSelector {
  private config: OptionsConfig;
  private performanceHistory: Map<string, RoutePerformance> = new Map();
  private banditModel: ContextualBanditModel;

  constructor() {
    this.config = {
      routes: {
        enabled: ['vertical', 'long_call', 'long_put'],
        ivRankVerticalBias: 0.6,
        expectedMoveFactor: 0.75
      },
      friction: {
        rejectThreshold: 0.20,
        penaltyThreshold: 0.12,
        perContractFee: 0.65,
        slippageModel: 'realistic'
      },
      greeks: {
        gammaPerTradeMax: {
          default: 0.02,
          leveragedETF: 0.015
        },
        thetaGovernor: {
          warnThreshold: 0.002,
          clampThreshold: 0.0025,
          consecutiveDays: 2
        },
        vegaBudgetSoftPct: 0.20
      },
      events: {
        earningsPolicy: {
          holdOnlyIf: ['durable_catalyst'],
          maxIVRankHold: 0.80
        },
        exDivPolicy: 'auto_close'
      },
      chain: {
        minOI: 100,
        minVolume: 50,
        maxSpreadBps: 50,
        freshnessSeconds: 300
      }
    };

    this.initializeBanditModel();
  }

  /**
   * Select optimal route using contextual bandit learning
   */
  selectRoute(context: RouteSelectionContext): {
    selectedRoute: OptionRoute;
    alternatives: OptionRoute[];
    confidence: number;
    rationale: string;
  } {
    // Score all available routes
    const routeScores = this.scoreRoutes(context);

    // Apply contextual bandit sampling
    const banditScores = this.applyThompsonSampling(routeScores, context);

    // Sort by bandit score
    banditScores.sort((a, b) => b.banditScore - a.banditScore);

    const selected = banditScores[0];
    const alternatives = banditScores.slice(1, 4); // Top 3 alternatives

    return {
      selectedRoute: selected.route,
      alternatives: alternatives.map(a => a.route),
      confidence: selected.confidence,
      rationale: this.generateRationale(selected.route, context)
    };
  }

  /**
   * Update bandit model with trade outcome
   */
  updateBanditModel(reward: ContextualBanditReward): void {
    const route = reward.outcome.route;

    // Update performance history
    const performance = this.performanceHistory.get(route) || {
      route,
      totalTrades: 0,
      totalPnL: 0,
      avgPnL: 0,
      winRate: 0,
      avgFriction: 0,
      avgDrawdown: 0,
      lastUpdated: new Date()
    };

    performance.totalTrades += 1;
    performance.totalPnL += reward.outcome.pnl;
    performance.avgPnL = performance.totalPnL / performance.totalTrades;
    performance.avgFriction = (performance.avgFriction * (performance.totalTrades - 1) + reward.outcome.friction) / performance.totalTrades;
    performance.avgDrawdown = (performance.avgDrawdown * (performance.totalTrades - 1) + reward.outcome.drawdown) / performance.totalTrades;
    performance.winRate = performance.totalPnL > 0 ? 1 : 0; // Simplified
    performance.lastUpdated = new Date();

    this.performanceHistory.set(route, performance);

    // Update bandit parameters
    const success = reward.outcome.pnl > 0 ? 1 : 0;
    const failure = 1 - success;

    if (!this.banditModel.alpha[route]) {
      this.banditModel.alpha[route] = 1; // Beta distribution prior
      this.banditModel.beta[route] = 1;
    }

    this.banditModel.alpha[route] += success;
    this.banditModel.beta[route] += failure;

    // Update context weights using online learning
    this.updateContextWeights(reward);

    this.banditModel.lastUpdated = new Date();
  }

  /**
   * Get route performance statistics
   */
  getRouteStatistics(): { [route: string]: RoutePerformance } {
    const stats: { [route: string]: RoutePerformance } = {};

    for (const [route, performance] of this.performanceHistory) {
      stats[route] = { ...performance };
    }

    return stats;
  }

  /**
   * Reset bandit model (for testing or when market regime changes)
   */
  resetBanditModel(): void {
    this.initializeBanditModel();
    console.log('[RouteSelector] Bandit model reset');
  }

  /**
   * Initialize bandit model with priors
   */
  private initializeBanditModel(): void {
    this.banditModel = {
      alpha: {},
      beta: {},
      contextWeights: {
        ivRank: {},
        expectedMove: {},
        trendStrength: {},
        rvol: {},
        chainQuality: {}
      },
      lastUpdated: new Date()
    };

    // Initialize with weak priors
    const routes = ['vertical', 'long_call', 'long_put', 'equity', 'leveraged_etf'];
    routes.forEach(route => {
      this.banditModel.alpha[route] = 1;
      this.banditModel.beta[route] = 1;

      this.banditModel.contextWeights.ivRank[route] = 0.1;
      this.banditModel.contextWeights.expectedMove[route] = 0.1;
      this.banditModel.contextWeights.trendStrength[route] = 0.1;
      this.banditModel.contextWeights.rvol[route] = 0.1;
      this.banditModel.contextWeights.chainQuality[route] = 0.1;
    });
  }

  /**
   * Score routes based on context and performance history
   */
  private scoreRoutes(context: RouteSelectionContext): Array<{
    route: OptionRoute;
    contextScore: number;
    performanceScore: number;
  }> {
    const scores: Array<{
      route: OptionRoute;
      contextScore: number;
      performanceScore: number;
    }> = [];

    for (const route of context.availableRoutes) {
      const contextScore = this.calculateContextScore(route, context);
      const performanceScore = this.calculatePerformanceScore(route.type);

      scores.push({
        route,
        contextScore,
        performanceScore
      });
    }

    return scores;
  }

  /**
   * Calculate context-based score for a route
   */
  private calculateContextScore(route: OptionRoute, context: RouteSelectionContext): number {
    let score = 0;

    // IV-aware bias
    if (route.type === 'vertical' && context.ivRank > this.config.routes.ivRankVerticalBias) {
      score += 0.3;
    } else if ((route.type === 'long_call' || route.type === 'long_put') && context.ivRank < 0.4) {
      score += 0.2;
    }

    // Expected move validation
    const maxTarget = context.expectedMove * this.config.routes.expectedMoveFactor;
    if (route.confidence <= maxTarget) {
      score += 0.2;
    }

    // Friction headroom
    if (context.frictionHeadroom > 0.8) {
      score += 0.15;
    }

    // Greeks headroom
    if (context.thetaHeadroom > 0.8) {
      score += 0.1;
    }

    if (context.vegaHeadroom > 0.8) {
      score += 0.1;
    }

    return Math.min(1.0, score);
  }

  /**
   * Calculate performance-based score from history
   */
  private calculatePerformanceScore(routeType: string): number {
    const performance = this.performanceHistory.get(routeType);

    if (!performance || performance.totalTrades < 5) {
      return 0.5; // Neutral score for new/untested routes
    }

    let score = 0;

    // P&L score
    if (performance.avgPnL > 0) {
      score += 0.4;
    }

    // Win rate score
    score += performance.winRate * 0.3;

    // Friction efficiency score
    if (performance.avgFriction < 0.15) {
      score += 0.2;
    }

    // Drawdown penalty
    score -= Math.min(0.2, performance.avgDrawdown);

    return Math.max(0, Math.min(1.0, score));
  }

  /**
   * Apply Thompson sampling to route scores
   */
  private applyThompsonSampling(
    routeScores: Array<{
      route: OptionRoute;
      contextScore: number;
      performanceScore: number;
    }>,
    context: RouteSelectionContext
  ): Array<{
    route: OptionRoute;
    banditScore: number;
    confidence: number;
  }> {
    return routeScores.map(({ route, contextScore, performanceScore }) => {
      const routeType = route.type;

      // Get Thompson sample
      const alpha = this.banditModel.alpha[routeType] || 1;
      const beta = this.banditModel.beta[routeType] || 1;

      // Sample from Beta distribution
      const thompsonSample = this.sampleBeta(alpha, beta);

      // Combine with context and performance scores
      const combinedScore = (thompsonSample * 0.4) + (contextScore * 0.35) + (performanceScore * 0.25);

      return {
        route,
        banditScore: combinedScore,
        confidence: thompsonSample
      };
    });
  }

  /**
   * Sample from Beta distribution using gamma approximation
   */
  private sampleBeta(alpha: number, beta: number): number {
    const gamma1 = this.sampleGamma(alpha, 1);
    const gamma2 = this.sampleGamma(beta, 1);
    return gamma1 / (gamma1 + gamma2);
  }

  /**
   * Sample from Gamma distribution (simplified approximation)
   */
  private sampleGamma(shape: number, scale: number): number {
    // Using Marsaglia and Tsang method approximation
    if (shape >= 1) {
      const d = shape - 1/3;
      const c = 1 / Math.sqrt(9 * d);
      let x, v;
      do {
        do {
          x = this.randomNormal();
          v = 1 + c * x;
        } while (v <= 0);
        v = v * v * v;
        const u = Math.random();
        if (u < 1 - 0.0331 * x * x * x * x) break;
        if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) break;
      } while (true);
      return d * v * scale;
    } else {
      // For shape < 1, use acceptance-rejection
      return this.sampleGamma(shape + 1, scale) * Math.pow(Math.random(), 1 / shape);
    }
  }

  /**
   * Generate normal random variable
   */
  private randomNormal(): number {
    // Box-Muller transform
    const u1 = Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  /**
   * Update context weights using online learning
   */
  private updateContextWeights(reward: ContextualBanditReward): void {
    const route = reward.outcome.route;
    const success = reward.outcome.pnl > 0 ? 1 : 0;
    const learningRate = 0.01;

    // Update weights based on reward
    const context = reward.context;

    this.updateWeight('ivRank', route, context.ivRank, success, learningRate);
    this.updateWeight('expectedMove', route, context.expectedMove / context.expectedMove, success, learningRate);
    this.updateWeight('trendStrength', route, context.trendStrength, success, learningRate);
    this.updateWeight('rvol', route, context.rvol, success, learningRate);
    this.updateWeight('chainQuality', route, context.chainQuality, success, learningRate);
  }

  /**
   * Update individual weight using gradient descent
   */
  private updateWeight(
    feature: keyof ContextualBanditModel['contextWeights'],
    route: string,
    featureValue: number,
    success: number,
    learningRate: number
  ): void {
    const currentWeight = this.banditModel.contextWeights[feature][route] || 0;
    const gradient = (success - 0.5) * featureValue; // Simplified gradient

    this.banditModel.contextWeights[feature][route] = currentWeight + learningRate * gradient;
  }

  /**
   * Generate human-readable rationale for route selection
   */
  private generateRationale(route: OptionRoute, context: RouteSelectionContext): string {
    let rationale = `Selected ${route.type} route. `;

    if (route.type === 'vertical' && context.ivRank > this.config.routes.ivRankVerticalBias) {
      rationale += `High IV rank (${(context.ivRank * 100).toFixed(0)}%) favors defined-risk verticals. `;
    }

    if (route.type.includes('long') && context.ivRank < 0.4) {
      rationale += `Low IV rank (${(context.ivRank * 100).toFixed(0)}%) supports long options. `;
    }

    if (context.frictionHeadroom > 0.8) {
      rationale += `Good friction headroom (${(context.frictionHeadroom * 100).toFixed(0)}%). `;
    }

    if (context.thetaHeadroom > 0.8) {
      rationale += `Strong theta collection potential. `;
    }

    return rationale;
  }
}

// Singleton instance
export const routeSelector = new RouteSelector();
