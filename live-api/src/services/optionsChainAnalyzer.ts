/**
 * Options Chain Analyzer for EvoTester
 * Analyzes options chains for quality, liquidity, and trading opportunities
 */

import {
  OptionChain,
  OptionContract,
  ChainQuality,
  OptionGreeks
} from './optionsTypes';

export class OptionsChainAnalyzer {
  /**
   * Analyze options chain quality and liquidity
   */
  analyzeChain(chain: OptionChain): ChainQuality {
    const spreadScore = this.calculateSpreadScore(chain);
    const volumeScore = this.calculateVolumeScore(chain);
    const oiScore = this.calculateOpenInterestScore(chain);

    const overall = (spreadScore * 0.4 + volumeScore * 0.35 + oiScore * 0.25);

    return {
      spreadScore,
      volumeScore,
      oiScore,
      overall
    };
  }

  /**
   * Find optimal strike prices for given strategy
   */
  findOptimalStrikes(
    chain: OptionChain,
    strategy: 'vertical' | 'long_call' | 'long_put',
    underlyingPrice: number,
    expectedMove: number,
    riskPreference: 'conservative' | 'moderate' | 'aggressive' = 'moderate'
  ): {
    longStrike?: number;
    shortStrike?: number;
    premium: number;
    maxLoss: number;
    maxGain: number;
    probability: number;
  } {
    const allStrikes = this.getStrikePrices(chain);

    switch (strategy) {
      case 'vertical':
        return this.findVerticalStrikes(chain, underlyingPrice, expectedMove, riskPreference);
      case 'long_call':
        return this.findLongCallStrike(chain, underlyingPrice, expectedMove, riskPreference);
      case 'long_put':
        return this.findLongPutStrike(chain, underlyingPrice, expectedMove, riskPreference);
      default:
        throw new Error(`Unsupported strategy: ${strategy}`);
    }
  }

  /**
   * Calculate spread quality score (0-1, higher is better)
   */
  private calculateSpreadScore(chain: OptionChain): number {
    const allOptions = [...chain.calls, ...chain.puts];
    if (allOptions.length === 0) return 0;

    const spreads = allOptions.map(opt => (opt.ask - opt.bid) / opt.bid);
    const avgSpread = spreads.reduce((sum, spread) => sum + spread, 0) / spreads.length;

    // Convert to score (lower spread = higher score)
    return Math.max(0, Math.min(1, 1 - (avgSpread / 0.1))); // 10% spread = 0 score
  }

  /**
   * Calculate volume quality score (0-1, higher is better)
   */
  private calculateVolumeScore(chain: OptionChain): number {
    const allOptions = [...chain.calls, ...chain.puts];
    if (allOptions.length === 0) return 0;

    const volumes = allOptions.map(opt => opt.volume);
    const avgVolume = volumes.reduce((sum, vol) => sum + vol, 0) / volumes.length;

    // Score based on volume thresholds
    if (avgVolume >= 1000) return 1.0;
    if (avgVolume >= 500) return 0.8;
    if (avgVolume >= 100) return 0.6;
    if (avgVolume >= 50) return 0.4;
    if (avgVolume >= 10) return 0.2;
    return 0.0;
  }

  /**
   * Calculate open interest quality score (0-1, higher is better)
   */
  private calculateOpenInterestScore(chain: OptionChain): number {
    const allOptions = [...chain.calls, ...chain.puts];
    if (allOptions.length === 0) return 0;

    const ois = allOptions.map(opt => opt.openInterest);
    const avgOI = ois.reduce((sum, oi) => sum + oi, 0) / ois.length;

    // Score based on OI thresholds
    if (avgOI >= 5000) return 1.0;
    if (avgOI >= 2000) return 0.8;
    if (avgOI >= 1000) return 0.7;
    if (avgOI >= 500) return 0.6;
    if (avgOI >= 200) return 0.4;
    if (avgOI >= 50) return 0.2;
    return 0.0;
  }

  /**
   * Get all unique strike prices from chain
   */
  private getStrikePrices(chain: OptionChain): number[] {
    const strikes = new Set<number>();
    [...chain.calls, ...chain.puts].forEach(option => strikes.add(option.strike));
    return Array.from(strikes).sort((a, b) => a - b);
  }

  /**
   * Find optimal vertical spread strikes
   */
  private findVerticalStrikes(
    chain: OptionChain,
    underlyingPrice: number,
    expectedMove: number,
    riskPreference: 'conservative' | 'moderate' | 'aggressive'
  ): {
    longStrike?: number;
    shortStrike?: number;
    premium: number;
    maxLoss: number;
    maxGain: number;
    probability: number;
  } {
    const strikes = this.getStrikePrices(chain);
    const movePercent = expectedMove / underlyingPrice;

    let bestSpread = {
      longStrike: strikes[0],
      shortStrike: strikes[0],
      premium: 0,
      maxLoss: 0,
      maxGain: 0,
      probability: 0
    };

    // Try different spread widths based on risk preference
    const widthMultiplier = riskPreference === 'conservative' ? 1.5 :
                           riskPreference === 'moderate' ? 2.0 : 2.5;
    const targetWidth = expectedMove * widthMultiplier;

    for (let i = 0; i < strikes.length - 1; i++) {
      for (let j = i + 1; j < strikes.length; j++) {
        const longStrike = strikes[i];
        const shortStrike = strikes[j];
        const width = shortStrike - longStrike;

        if (width < targetWidth * 0.5 || width > targetWidth * 1.5) continue;

        // Find call options for this spread
        const longCall = chain.calls.find(c => c.strike === longStrike);
        const shortCall = chain.calls.find(c => c.strike === shortStrike);

        if (!longCall || !shortCall) continue;

        const premium = longCall.ask - shortCall.bid;
        if (premium <= 0) continue;

        const maxLoss = premium;
        const maxGain = width - premium;

        // Calculate probability of profit (simplified)
        const breakEven = longStrike + premium;
        const probability = this.calculateProbabilityOfProfit(underlyingPrice, breakEven, expectedMove);

        // Update best if better risk-adjusted return
        const riskAdjReturn = (probability * maxGain - (1 - probability) * maxLoss) / maxLoss;
        const currentRiskAdjReturn = (bestSpread.probability * bestSpread.maxGain -
                                    (1 - bestSpread.probability) * bestSpread.maxLoss) / bestSpread.maxLoss;

        if (riskAdjReturn > currentRiskAdjReturn) {
          bestSpread = {
            longStrike,
            shortStrike,
            premium,
            maxLoss,
            maxGain,
            probability
          };
        }
      }
    }

    return bestSpread;
  }

  /**
   * Find optimal long call strike
   */
  private findLongCallStrike(
    chain: OptionChain,
    underlyingPrice: number,
    expectedMove: number,
    riskPreference: 'conservative' | 'moderate' | 'aggressive'
  ): {
    longStrike?: number;
    premium: number;
    maxLoss: number;
    maxGain: number;
    probability: number;
  } {
    const strikes = this.getStrikePrices(chain).filter(s => s >= underlyingPrice);
    const targetPrice = underlyingPrice + expectedMove;

    let bestOption = {
      longStrike: strikes[0],
      premium: 0,
      maxLoss: 0,
      maxGain: 0,
      probability: 0
    };

    for (const strike of strikes) {
      const callOption = chain.calls.find(c => c.strike === strike);
      if (!callOption) continue;

      const premium = callOption.ask;
      const maxLoss = premium;
      const maxGain = Math.max(0, targetPrice - strike - premium);

      // Skip if strike is too far from current price
      const strikeDistance = Math.abs(strike - underlyingPrice) / underlyingPrice;
      const maxDistance = riskPreference === 'conservative' ? 0.1 :
                         riskPreference === 'moderate' ? 0.15 : 0.25;

      if (strikeDistance > maxDistance) continue;

      // Calculate probability of profit
      const probability = this.calculateProbabilityOfProfit(underlyingPrice, strike + premium, expectedMove);

      // Update best if higher expected value
      const expectedValue = probability * maxGain - (1 - probability) * maxLoss;
      const currentExpectedValue = bestOption.probability * bestOption.maxGain -
                                 (1 - bestOption.probability) * bestOption.maxLoss;

      if (expectedValue > currentExpectedValue) {
        bestOption = {
          longStrike: strike,
          premium,
          maxLoss,
          maxGain,
          probability
        };
      }
    }

    return bestOption;
  }

  /**
   * Find optimal long put strike
   */
  private findLongPutStrike(
    chain: OptionChain,
    underlyingPrice: number,
    expectedMove: number,
    riskPreference: 'conservative' | 'moderate' | 'aggressive'
  ): {
    longStrike?: number;
    premium: number;
    maxLoss: number;
    maxGain: number;
    probability: number;
  } {
    const strikes = this.getStrikePrices(chain).filter(s => s <= underlyingPrice);
    const targetPrice = underlyingPrice - expectedMove;

    let bestOption = {
      longStrike: strikes[0],
      premium: 0,
      maxLoss: 0,
      maxGain: 0,
      probability: 0
    };

    for (const strike of strikes) {
      const putOption = chain.puts.find(p => p.strike === strike);
      if (!putOption) continue;

      const premium = putOption.ask;
      const maxLoss = premium;
      const maxGain = Math.max(0, strike - targetPrice - premium);

      // Skip if strike is too far from current price
      const strikeDistance = Math.abs(strike - underlyingPrice) / underlyingPrice;
      const maxDistance = riskPreference === 'conservative' ? 0.1 :
                         riskPreference === 'moderate' ? 0.15 : 0.25;

      if (strikeDistance > maxDistance) continue;

      // Calculate probability of profit
      const probability = this.calculateProbabilityOfProfit(underlyingPrice, strike - premium, -expectedMove);

      // Update best if higher expected value
      const expectedValue = probability * maxGain - (1 - probability) * maxLoss;
      const currentExpectedValue = bestOption.probability * bestOption.maxGain -
                                 (1 - bestOption.probability) * bestOption.maxLoss;

      if (expectedValue > currentExpectedValue) {
        bestOption = {
          longStrike: strike,
          premium,
          maxLoss,
          maxGain,
          probability
        };
      }
    }

    return bestOption;
  }

  /**
   * Calculate probability of profit for a given scenario
   */
  private calculateProbabilityOfProfit(currentPrice: number, targetPrice: number, expectedMove: number): number {
    // Simplified normal distribution approximation
    const move = Math.abs(targetPrice - currentPrice);
    const stdDev = expectedMove / 1.96; // 95% confidence interval

    if (stdDev === 0) return targetPrice > currentPrice ? 0.5 : 0.5;

    const zScore = move / stdDev;

    // Use error function approximation for normal CDF
    const cdf = 0.5 * (1 + this.erf(zScore / Math.sqrt(2)));

    return targetPrice > currentPrice ? 1 - cdf : cdf;
  }

  /**
   * Error function approximation
   */
  private erf(x: number): number {
    const a1 =  0.254829592;
    const a2 = -0.284496736;
    const a3 =  1.421413741;
    const a4 = -1.453152027;
    const a5 =  1.061405429;
    const p  =  0.3275911;

    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x);

    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

    return sign * y;
  }

  /**
   * Validate if chain meets minimum quality standards
   */
  validateChain(chain: OptionChain): {
    valid: boolean;
    reasons: string[];
    score: ChainQuality;
  } {
    const score = this.analyzeChain(chain);
    const reasons: string[] = [];

    if (score.spreadScore < 0.5) {
      reasons.push('Poor spread quality');
    }

    if (score.volumeScore < 0.3) {
      reasons.push('Insufficient volume');
    }

    if (score.oiScore < 0.3) {
      reasons.push('Insufficient open interest');
    }

    if (score.overall < 0.5) {
      reasons.push('Overall chain quality too low');
    }

    return {
      valid: reasons.length === 0,
      reasons,
      score
    };
  }
}

// Singleton instance
export const optionsChainAnalyzer = new OptionsChainAnalyzer();
