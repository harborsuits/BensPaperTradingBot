/**
 * Catalyst Scorer for Poor-Capital Mode
 *
 * Scores symbols based on news impact, RVOL, gap quality, SI/float, and context
 * Only symbols scoring ≥60% enter EvoTester
 */

import { POOR_CAPITAL_MODE } from '../../config/poorCapitalMode';

export interface CatalystScores {
  newsImpact: number; // 0-1
  relativeVolume: number; // 0-1
  gapQuality: number; // 0-1
  shortInterestRatio: number; // 0-1
  contextScore: number; // 0-1
  totalScore: number; // 0-1
  passesThreshold: boolean;
}

export interface MarketData {
  symbol: string;
  price: number;
  volume: number;
  avgVolume20d: number;
  spreadBps: number;
  floatShares: number;
  shortInterest?: number;
  gapPercent?: number;
  preMarketVolume?: number;
  newsSentiment?: number; // -1 to 1
  insiderBuying?: boolean;
  institutionalOwnership?: number;
  recentEarnings?: boolean;
  fdaNews?: boolean;
  contractWin?: boolean;
  strategicDeal?: boolean;
}

export class CatalystScorer {
  private config = POOR_CAPITAL_MODE.catalysts;

  /**
   * Score a symbol based on all catalyst factors
   */
  scoreSymbol(data: MarketData): CatalystScores {
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

  /**
   * News impact scoring (40% weight)
   */
  private scoreNewsImpact(data: MarketData): number {
    let score = 0;

    // Major catalysts get high scores
    if (data.fdaNews) score += 0.4; // FDA approval/rejection
    if (data.recentEarnings) score += 0.3; // Recent earnings
    if (data.contractWin) score += 0.2; // Contract/announcement
    if (data.strategicDeal) score += 0.2; // M&A/partnership

    // News sentiment modifier
    if (data.newsSentiment !== undefined) {
      if (data.newsSentiment > 0.5) score *= 1.2; // Positive news boost
      else if (data.newsSentiment < -0.3) score *= 0.8; // Negative news penalty
    }

    return Math.min(1.0, score);
  }

  /**
   * Relative volume scoring (25% weight)
   */
  private scoreRelativeVolume(data: MarketData): number {
    if (!data.avgVolume20d || data.avgVolume20d === 0) return 0;

    const rvol = data.volume / data.avgVolume20d;

    // Score based on RVOL thresholds
    if (rvol >= 3.0) return 1.0; // Extreme volume
    if (rvol >= 2.0) return 0.8; // Very high volume
    if (rvol >= 1.5) return 0.6; // High volume
    if (rvol >= 1.2) return 0.4; // Moderate volume
    if (rvol >= 1.0) return 0.2; // Normal volume

    return 0; // Below average volume
  }

  /**
   * Gap quality scoring (15% weight)
   */
  private scoreGapQuality(data: MarketData): number {
    if (!data.gapPercent || !data.preMarketVolume) return 0;

    let score = 0;

    // Gap size (up to 0.3 for large gaps)
    const gapMagnitude = Math.min(Math.abs(data.gapPercent), 0.15) / 0.15;
    score += gapMagnitude * 0.4;

    // Pre-market shelf strength (volume before open)
    if (data.preMarketVolume && data.avgVolume20d) {
      const preMarketRvol = data.preMarketVolume / (data.avgVolume20d * 0.1); // Compare to typical pre-market
      if (preMarketRvol > 2.0) score += 0.3; // Strong pre-market interest
      else if (preMarketRvol > 1.0) score += 0.2; // Moderate pre-market interest
    }

    // Gap direction preference (up gaps slightly preferred for momentum)
    if (data.gapPercent > 0) score *= 1.1;
    else if (data.gapPercent < -0.05) score *= 0.9; // Large down gaps penalized

    return Math.min(1.0, score);
  }

  /**
   * Short interest ratio scoring (10% weight)
   */
  private scoreShortInterest(data: MarketData): number {
    if (!data.shortInterest || !data.floatShares) return 0.5; // Neutral if no data

    const siRatio = data.shortInterest / data.floatShares;

    // Optimal SI ratio is 5-15% (squeeze potential without extreme pessimism)
    if (siRatio >= 0.05 && siRatio <= 0.15) return 1.0;
    if (siRatio >= 0.03 && siRatio <= 0.20) return 0.8;
    if (siRatio >= 0.02 && siRatio <= 0.25) return 0.6;
    if (siRatio > 0.25) return 0.3; // Too high SI = extreme pessimism
    if (siRatio < 0.02) return 0.4; // Too low SI = limited squeeze potential

    return 0.5;
  }

  /**
   * Context scoring (10% weight)
   */
  private scoreContext(data: MarketData): number {
    let score = 0;

    // Insider buying (strong signal)
    if (data.insiderBuying) score += 0.4;

    // Institutional ownership (stability indicator)
    if (data.institutionalOwnership) {
      if (data.institutionalOwnership > 0.7) score += 0.3; // High institutional interest
      else if (data.institutionalOwnership > 0.5) score += 0.2;
      else if (data.institutionalOwnership > 0.3) score += 0.1;
    }

    // Size-appropriate context
    if (data.price < 3 && data.volume > data.avgVolume20d * 2) {
      score += 0.2; // Small caps with volume = attention
    }

    // Recent catalysts get bonus
    const daysSinceEvent = 0; // Would be calculated from data
    if (daysSinceEvent <= 3) score *= 1.2; // Recent events more relevant

    return Math.min(1.0, score);
  }

  /**
   * Filter symbols that pass universe constraints
   */
  passesUniverseFilter(data: MarketData): boolean {
    const universe = POOR_CAPITAL_MODE.universe;

    // Price range
    if (data.price < universe.minPrice || data.price > universe.maxPrice) {
      return false;
    }

    // Volume requirement
    if (!data.avgVolume20d || data.avgVolume20d < universe.minDollarADV) {
      return false;
    }

    // Spread constraint
    if (data.spreadBps > universe.maxSpreadBps) {
      return false;
    }

    // Float requirement
    if (!data.floatShares || data.floatShares < universe.minFloat) {
      return false;
    }

    return true;
  }

  /**
   * Get top scoring symbols for EvoTester
   */
  getTopCandidates(symbolsData: MarketData[], maxCandidates = 50): MarketData[] {
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

// Singleton instance
export const catalystScorer = new CatalystScorer();
