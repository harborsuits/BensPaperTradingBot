/**
 * Poor-Capital Mode Services (JavaScript version for immediate testing)
 *
 * This is a temporary JavaScript version of the TypeScript services
 * for testing purposes before full TypeScript compilation is set up.
 */

// Poor-Capital Mode Configuration
const POOR_CAPITAL_MODE = {
  enabled: true,
  universe: {
    listedOnly: true,
    minPrice: 1,
    maxPrice: 10,
    minDollarADV: 10_000_000,
    maxSpreadBps: 20,
    minFloat: 10_000_000,
    skipHaltWindowMin: 30,
    allowSSR: true,
  },
  catalysts: {
    w_news: 0.40,
    w_rvol: 0.25,
    w_gap: 0.15,
    w_si: 0.10,
    w_context: 0.10,
    minScore: 0.6,
  },
  risk: {
    perTradeRiskPct: 0.007,
    maxPositionNotionalPct: 0.10,
    maxOpenRiskPct: 0.015,
    maxDailyLossPct: 0.01,
  },
  execution: {
    protectBps: 10,
    maxSlippageBps: 15,
    blockOpenCloseMin: 15,
    wholeSharesMin: 2,
    minStopDistanceBps: 150,
  },
  advancedGuards: {
    advParticipationMax: 0.0005, // 0.05% of ADV
    pdtGuard: {
      maxDayTradesPer5d: 2,
      preferSwing: true,
      blockDayTradesIfNearLimit: true,
    },
    sectorMaxActive: 2,
    ssrHandling: {
      raiseMinStopDistance: 2.0,
      halveMaxSize: true,
    },
    drawdownGovernor: {
      pauseNewEntriesThreshold: 0.007,
      resumeThreshold: 0.003,
    },
    crowding: {
      sectorMaxActive: 2,
      corrPenaltyThreshold: 0.75,
    },
    governor: {
      dayPnlFreezePct: -0.007,
    },
    pdt: {
      maxDayTrades5d: 2,
      preferSwingNearClose: true,
    },
  },

  // New enhancement settings
  fitnessEnhancements: {
    capitalEfficiencyFloor: 0.5, // $0.50 P&L per $ risked minimum
    frictionCap: 0.25, // (slippage+fees)/grossPnl max
    capitalEfficiencyWeight: 0.10,
    frictionPenaltyWeight: 0.10,
  },

  riskTilt: {
    min: 0.005, // 0.5%
    base: 0.007, // 0.7%
    slope: 0.007, // Additional risk per conviction unit
    max: 0.012, // 1.2%
    clampOnSSR: true,
  },

  leveragedETF: {
    allowOvernight: false,
    maxSlippageBps: 10,
    riskPctMax: 0.006, // 0.6%
    sessionWindow: ['10:00', '15:30'],
  },

  overnight: {
    allowTags: ['earnings_beat_guidance', 'fda_approval', 'contract_win'],
    halveSize: true,
    widenStopMult: 1.5,
  },

  ttlRenew: {
    minSharpe: 1.0,
    minPnL: 0,
    minPnlPerRisk: 0.8,
    maxCorr: 0.75,
  },
};

// Catalyst Scorer
class CatalystScorer {
  constructor() {
    this.config = POOR_CAPITAL_MODE.catalysts;
  }

  scoreSymbol(data) {
    const newsImpact = this.scoreNewsImpact(data);
    const relativeVolume = this.scoreRelativeVolume(data);
    const gapQuality = this.scoreGapQuality(data);
    const shortInterestRatio = this.scoreShortInterest(data);
    const contextScore = this.scoreContext(data);

    const totalScore =
      (newsImpact * this.config.w_news) +
      (relativeVolume * this.config.w_rvol) +
      (gapQuality * this.config.w_gap) +
      (shortInterestRatio * this.config.w_si) +
      (contextScore * this.config.w_context);

    return {
      newsImpact,
      relativeVolume,
      gapQuality,
      shortInterestRatio,
      contextScore,
      totalScore,
      passesThreshold: totalScore >= this.config.minScore,
    };
  }

  scoreNewsImpact(data) {
    let score = 0;
    if (data.fdaNews) score += 0.4;
    if (data.recentEarnings) score += 0.3;
    if (data.contractWin) score += 0.2;
    if (data.strategicDeal) score += 0.2;
    if (data.newsSentiment !== undefined) {
      if (data.newsSentiment > 0.5) score *= 1.2;
      else if (data.newsSentiment < -0.3) score *= 0.8;
    }
    return Math.min(1.0, score);
  }

  scoreRelativeVolume(data) {
    if (!data.avgVolume20d || data.avgVolume20d === 0) return 0;
    const rvol = data.volume / data.avgVolume20d;
    if (rvol >= 3.0) return 1.0;
    if (rvol >= 2.0) return 0.8;
    if (rvol >= 1.5) return 0.6;
    if (rvol >= 1.2) return 0.4;
    if (rvol >= 1.0) return 0.2;
    return 0;
  }

  scoreGapQuality(data) {
    if (!data.gapPercent || !data.preMarketVolume) return 0;
    let score = 0;
    const gapMagnitude = Math.min(Math.abs(data.gapPercent), 0.15) / 0.15;
    score += gapMagnitude * 0.4;
    if (data.preMarketVolume && data.avgVolume20d) {
      const preMarketRvol = data.preMarketVolume / (data.avgVolume20d * 0.1);
      if (preMarketRvol > 2.0) score += 0.3;
      else if (preMarketRvol > 1.0) score += 0.2;
    }
    if (data.gapPercent > 0) score *= 1.1;
    else if (data.gapPercent < -0.05) score *= 0.9;
    return Math.min(1.0, score);
  }

  scoreShortInterest(data) {
    if (!data.shortInterest || !data.floatShares) return 0.5;
    const siRatio = data.shortInterest / data.floatShares;
    if (siRatio >= 0.05 && siRatio <= 0.15) return 1.0;
    if (siRatio >= 0.03 && siRatio <= 0.20) return 0.8;
    if (siRatio >= 0.02 && siRatio <= 0.25) return 0.6;
    if (siRatio > 0.25) return 0.3;
    if (siRatio < 0.02) return 0.4;
    return 0.5;
  }

  scoreContext(data) {
    let score = 0;
    if (data.insiderBuying) score += 0.4;
    if (data.institutionalOwnership) {
      if (data.institutionalOwnership > 0.7) score += 0.3;
      else if (data.institutionalOwnership > 0.5) score += 0.2;
      else if (data.institutionalOwnership > 0.3) score += 0.1;
    }
    if (data.price < 3 && data.volume > data.avgVolume20d * 2) {
      score += 0.2;
    }
    return Math.min(1.0, score);
  }

  passesUniverseFilter(data) {
    const universe = POOR_CAPITAL_MODE.universe;
    if (data.price < universe.minPrice || data.price > universe.maxPrice) return false;
    if (!data.avgVolume20d || data.avgVolume20d < universe.minDollarADV) return false;
    if (data.spreadBps > universe.maxSpreadBps) return false;
    if (!data.floatShares || data.floatShares < universe.minFloat) return false;
    return true;
  }

  getTopCandidates(symbolsData, maxCandidates = 50) {
    return symbolsData
      .filter(data => this.passesUniverseFilter(data))
      .map(data => ({
        ...data,
        catalystScore: this.scoreSymbol(data)
      }))
      .filter(item => item.catalystScore.passesThreshold)
      .sort((a, b) => b.catalystScore.totalScore - a.catalystScore.totalScore)
      .slice(0, maxCandidates);
  }
}

// Position Sizer
class PositionSizer {
  constructor() {
    this.config = POOR_CAPITAL_MODE;
  }

  calculatePosition(input) {
    const riskPercent = this.config.risk.perTradeRiskPct;
    const maxPositionPercent = this.config.risk.maxPositionNotionalPct;

    const riskAmount = input.capital * riskPercent;
    const stopDistancePercent = Math.abs(input.entryPrice - input.stopPrice) / input.entryPrice;

    if (stopDistancePercent === 0) {
      return this.createRejectedResult('Invalid stop price (same as entry)');
    }

    const safeNotional = riskAmount / stopDistancePercent;
    const maxNotional = input.capital * maxPositionPercent;
    const targetNotional = Math.min(safeNotional, maxNotional);

    let shares = Math.floor(targetNotional / input.entryPrice);
    let actualNotional = shares * input.entryPrice;

    if (shares < this.config.execution.wholeSharesMin) {
      const minShares = this.config.execution.wholeSharesMin;
      const minNotional = minShares * input.entryPrice;
      const minRiskPercent = (minNotional * stopDistancePercent) / input.capital;

      if (minRiskPercent <= riskPercent * 1.5) {
        shares = minShares;
        actualNotional = minNotional;
      } else {
        return this.createRejectedResult(
          `Position too small: ${minShares} shares = ${(minRiskPercent * 100).toFixed(2)}% risk (max ${riskPercent * 100}%)`
        );
      }
    }

    const advParticipationPercent = actualNotional / (input.avgDailyVolume * input.entryPrice);
    if (advParticipationPercent > this.config.advancedGuards.advParticipationMax) {
      const maxNotionalFromADV = input.avgDailyVolume * input.entryPrice * this.config.advancedGuards.advParticipationMax;
      shares = Math.floor(maxNotionalFromADV / input.entryPrice);
      actualNotional = shares * input.entryPrice;

      if (shares < this.config.execution.wholeSharesMin) {
        return this.createRejectedResult(
          `ADV participation too high: ${(advParticipationPercent * 100).toFixed(2)}% (max ${(this.config.advancedGuards.advParticipationMax * 100).toFixed(2)}%)`
        );
      }
    }

    const slippageEstimate = this.estimateSlippage(input, shares);
    if (slippageEstimate > this.config.execution.maxSlippageBps) {
      return this.createRejectedResult(
        `Expected slippage too high: ${slippageEstimate}bps (max ${this.config.execution.maxSlippageBps}bps)`
      );
    }

    return {
      shares,
      notional: actualNotional,
      riskAmount,
      riskPercent,
      stopDistancePercent,
      advParticipationPercent,
      slippageEstimateBps: slippageEstimate,
      canExecute: true,
      wholeSharesCost: actualNotional,
    };
  }

  estimateSlippage(input, shares) {
    const notional = shares * input.entryPrice;
    const participationPercent = notional / (input.avgDailyVolume * input.entryPrice);

    let slippageBps = input.spreadBps;
    if (participationPercent > 0.0001) {
      slippageBps += participationPercent * 10000;
    }
    if (input.volatility && input.volatility > 0.05) {
      slippageBps += (input.volatility - 0.05) * 200;
    }
    slippageBps += this.config.execution.protectBps;
    return Math.round(slippageBps);
  }

  createRejectedResult(reason) {
    return {
      shares: 0,
      notional: 0,
      riskAmount: 0,
      riskPercent: 0,
      stopDistancePercent: 0,
      advParticipationPercent: 0,
      slippageEstimateBps: 0,
      canExecute: false,
      rejectionReason: reason,
    };
  }
}

// Singleton instances
const catalystScorer = new CatalystScorer();
const positionSizer = new PositionSizer();

module.exports = {
  POOR_CAPITAL_MODE,
  catalystScorer,
  positionSizer
};
