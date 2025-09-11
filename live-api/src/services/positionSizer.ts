/**
 * Position Sizer for Poor-Capital Mode
 *
 * Realistic position sizing with whole shares, slippage protection, and ADV participation limits
 * Ensures strategies work with small accounts ($500-$5K)
 */

import { POOR_CAPITAL_MODE } from '../../config/poorCapitalMode';

export interface PositionSizingInput {
  capital: number;
  entryPrice: number;
  stopPrice: number;
  spreadBps: number;
  avgDailyVolume: number;
  marketCap?: number;
  volatility?: number;
  isLeveragedETF?: boolean;
  ssrActive?: boolean;
  conviction?: number; // 0-1 scale for dynamic risk tilt
  symbol?: string;
}

export interface PositionSizingResult {
  shares: number;
  notional: number;
  riskAmount: number;
  riskPercent: number;
  stopDistancePercent: number;
  advParticipationPercent: number;
  slippageEstimateBps: number;
  canExecute: boolean;
  rejectionReason?: string;
  adjustedStopPrice?: number;
  wholeSharesCost?: number;
}

export class PositionSizer {
  private config = POOR_CAPITAL_MODE;

  /**
   * Calculate optimal position size with all constraints
   */
  calculatePosition(input: PositionSizingInput): PositionSizingResult {
    // Dynamic risk tilt based on conviction
    let riskPercent = this.calculateDynamicRisk(input);

    // Leveraged ETF adjustments
    if (input.isLeveragedETF) {
      riskPercent = Math.min(riskPercent, this.config.leveragedETF.riskPctMax);
    }

    // SSR clamp
    if (input.ssrActive && this.config.riskTilt.clampOnSSR) {
      riskPercent = Math.min(riskPercent, this.config.riskTilt.min);
    }

    const maxPositionPercent = this.config.risk.maxPositionNotionalPct;

    // Step 1: Calculate risk-based position size
    const riskAmount = input.capital * riskPercent;
    const stopDistancePercent = Math.abs(input.entryPrice - input.stopPrice) / input.entryPrice;

    if (stopDistancePercent === 0) {
      return this.createRejectedResult('Invalid stop price (same as entry)');
    }

    // Safe position size based on risk
    const safeNotional = riskAmount / stopDistancePercent;

    // Cap at maximum position size
    const maxNotional = input.capital * maxPositionPercent;
    const targetNotional = Math.min(safeNotional, maxNotional);

    // Step 2: Calculate shares and apply whole-share constraint
    let shares = Math.floor(targetNotional / input.entryPrice);
    let actualNotional = shares * input.entryPrice;

    // Enforce minimum share requirement
    if (shares < this.config.execution.wholeSharesMin) {
      // Try to find a valid smaller position
      const minShares = this.config.execution.wholeSharesMin;
      const minNotional = minShares * input.entryPrice;
      const minRiskPercent = (minNotional * stopDistancePercent) / input.capital;

      if (minRiskPercent <= riskPercent * 1.5) { // Allow 50% risk increase for minimum position
        shares = minShares;
        actualNotional = minNotional;
      } else {
        return this.createRejectedResult(
          `Position too small: ${minShares} shares = ${(minRiskPercent * 100).toFixed(2)}% risk (max ${riskPercent * 100}%)`
        );
      }
    }

    // Step 3: Check ADV participation limit
    const advParticipationPercent = actualNotional / (input.avgDailyVolume * input.entryPrice);
    if (advParticipationPercent > this.config.advancedGuards.advParticipationMax) {
      // Reduce position size to meet ADV limit
      const maxNotionalFromADV = input.avgDailyVolume * input.entryPrice * this.config.advancedGuards.advParticipationMax;
      shares = Math.floor(maxNotionalFromADV / input.entryPrice);
      actualNotional = shares * input.entryPrice;

      if (shares < this.config.execution.wholeSharesMin) {
        return this.createRejectedResult(
          `ADV participation too high: ${(advParticipationPercent * 100).toFixed(2)}% (max ${(this.config.advancedGuards.advParticipationMax * 100).toFixed(2)}%)`
        );
      }
    }

    // Step 4: Validate stop distance from spread
    const minStopDistanceBps = this.config.execution.minStopDistanceBps;
    const actualStopDistanceBps = (stopDistancePercent * 10000);

    if (actualStopDistanceBps < minStopDistanceBps) {
      // Adjust stop to meet minimum distance
      const minStopDistancePercent = minStopDistanceBps / 10000;
      const adjustedStopPrice = input.entryPrice * (1 - minStopDistancePercent);
      const adjustedRiskAmount = actualNotional * minStopDistancePercent;

      return {
        shares,
        notional: actualNotional,
        riskAmount: adjustedRiskAmount,
        riskPercent: (adjustedRiskAmount / input.capital),
        stopDistancePercent: minStopDistancePercent,
        advParticipationPercent,
        slippageEstimateBps: this.estimateSlippage(input, shares),
        canExecute: true,
        adjustedStopPrice,
        wholeSharesCost: actualNotional,
      };
    }

    // Step 5: Estimate slippage and check limits
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

  /**
   * Calculate dynamic risk based on conviction
   */
  private calculateDynamicRisk(input: PositionSizingInput): number {
    const baseRisk = this.config.risk.perTradeRiskPct;
    const conviction = input.conviction || 0;

    // Dynamic risk tilt: risk = min + slope * conviction
    let dynamicRisk = this.config.riskTilt.min + (this.config.riskTilt.slope * conviction);

    // Clamp to bounds
    dynamicRisk = Math.max(this.config.riskTilt.min, Math.min(this.config.riskTilt.max, dynamicRisk));

    // Blend with base risk (weighted average)
    return (baseRisk * 0.7) + (dynamicRisk * 0.3);
  }

  /**
   * Estimate slippage for a given position size
   */
  private estimateSlippage(input: PositionSizingInput, shares: number): number {
    const notional = shares * input.entryPrice;
    const participationPercent = notional / (input.avgDailyVolume * input.entryPrice);

    // Leveraged ETF gets tighter slippage cap
    const maxSlippage = input.isLeveragedETF ?
      this.config.leveragedETF.maxSlippageBps :
      this.config.execution.maxSlippageBps;

    // Base slippage from spread
    let slippageBps = input.spreadBps;

    // Add participation-based slippage
    if (participationPercent > 0.0001) { // 0.01% participation
      slippageBps += participationPercent * 10000; // 1% participation adds 10bps
    }

    // Add volatility-based slippage
    if (input.volatility && input.volatility > 0.05) {
      slippageBps += (input.volatility - 0.05) * 200; // High volatility increases slippage
    }

    // Add protection buffer
    slippageBps += this.config.execution.protectBps;

    // Clamp to max for leveraged ETFs
    return Math.min(Math.round(slippageBps), maxSlippage);
  }

  /**
   * Check if position size is valid for current market conditions
   */
  validatePositionSize(
    capital: number,
    shares: number,
    price: number,
    stopPrice: number,
    spreadBps: number,
    avgDailyVolume: number
  ): { valid: boolean; reason?: string } {

    const notional = shares * price;
    const stopDistancePercent = Math.abs(price - stopPrice) / price;

    // Check capital constraints
    if (notional > capital * this.config.risk.maxPositionNotionalPct) {
      return {
        valid: false,
        reason: `Position too large: ${(notional / capital * 100).toFixed(1)}% (max ${this.config.risk.maxPositionNotionalPct * 100}%)`
      };
    }

    // Check risk constraints
    const riskAmount = notional * stopDistancePercent;
    const riskPercent = riskAmount / capital;

    if (riskPercent > this.config.risk.perTradeRiskPct * 1.5) { // Allow 50% overage for edge cases
      return {
        valid: false,
        reason: `Risk too high: ${(riskPercent * 100).toFixed(2)}% (max ${(this.config.risk.perTradeRiskPct * 100).toFixed(2)}%)`
      };
    }

    // Check ADV participation
    const advParticipationPercent = notional / (avgDailyVolume * price);
    if (advParticipationPercent > this.config.advancedGuards.advParticipationMax) {
      return {
        valid: false,
        reason: `ADV participation too high: ${(advParticipationPercent * 100).toFixed(3)}% (max ${(this.config.advancedGuards.advParticipationMax * 100).toFixed(3)}%)`
      };
    }

    // Check stop distance from spread
    const stopDistanceBps = stopDistancePercent * 10000;
    if (stopDistanceBps < this.config.execution.minStopDistanceBps) {
      return {
        valid: false,
        reason: `Stop too close to entry: ${stopDistanceBps.toFixed(0)}bps (min ${this.config.execution.minStopDistanceBps}bps)`
      };
    }

    return { valid: true };
  }

  /**
   * Get position sizing recommendations for a strategy
   */
  getSizingRecommendations(
    capital: number,
    strategy: {
      winRate: number;
      avgWin: number;
      avgLoss: number;
      maxDrawdown: number;
    }
  ): {
    recommendedRiskPercent: number;
    maxPositionPercent: number;
    confidenceAdjustment: number;
  } {

    // Adjust risk based on strategy metrics
    let riskMultiplier = 1.0;

    // Higher win rate allows slightly higher risk
    if (strategy.winRate > 0.6) riskMultiplier *= 1.2;
    else if (strategy.winRate < 0.45) riskMultiplier *= 0.8;

    // Profit factor adjustment
    const profitFactor = (strategy.winRate * strategy.avgWin) / ((1 - strategy.winRate) * Math.abs(strategy.avgLoss));
    if (profitFactor > 2.0) riskMultiplier *= 1.1;
    else if (profitFactor < 1.2) riskMultiplier *= 0.9;

    // Max DD adjustment (lower risk for volatile strategies)
    if (strategy.maxDrawdown > 0.15) riskMultiplier *= 0.9;

    const recommendedRiskPercent = Math.min(
      this.config.risk.perTradeRiskPct * riskMultiplier,
      this.config.risk.perTradeRiskPct * 1.5 // Cap at 150% of base
    );

    const maxPositionPercent = Math.min(
      this.config.risk.maxPositionNotionalPct,
      recommendedRiskPercent * 2 // Allow 2x risk for position sizing
    );

    return {
      recommendedRiskPercent,
      maxPositionPercent,
      confidenceAdjustment: riskMultiplier,
    };
  }

  private createRejectedResult(reason: string): PositionSizingResult {
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

// Singleton instance
export const positionSizer = new PositionSizer();
