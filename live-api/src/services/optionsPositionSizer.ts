/**
 * Options Position Sizer for EvoTester
 * Extends Poor-Capital Mode with options-specific risk management
 */

import { POOR_CAPITAL_MODE } from '../config/poorCapitalMode';
import {
  OptionsPositionInput,
  OptionsPositionResult,
  OptionRoute,
  RouteSelectionContext,
  OptionsConfig,
  OptionContract,
  ChainQuality
} from './optionsTypes';
import { PositionSizingResult, PositionSizingInput } from './positionSizer';

export class OptionsPositionSizer {
  private config: OptionsConfig;

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
  }

  /**
   * Calculate optimal options position size with all constraints
   */
  calculateOptionsPosition(input: OptionsPositionInput): OptionsPositionResult {
    // Validate chain quality and freshness
    if (!this.validateChainQuality(input.chainQuality)) {
      return this.createRejectedPosition('Poor chain quality');
    }

    if (!this.validateExpectedMove(input)) {
      return this.createRejectedPosition('Target exceeds expected move');
    }

    // Calculate premium and contracts
    const premium = this.calculatePremium(input);
    const contracts = this.calculateContracts(input, premium);

    if (contracts === 0) {
      return this.createRejectedPosition('Position too small for constraints');
    }

    // Calculate Greeks and risk metrics
    const greeks = this.calculateNetGreeks(input, contracts);
    const riskMetrics = this.calculateRiskMetrics(input, greeks, contracts);

    // Validate friction budget
    const friction = this.calculateFriction(input, premium, contracts);
    if (friction.ratio > this.config.friction.rejectThreshold) {
      return this.createRejectedPosition(`Friction too high: ${(friction.ratio * 100).toFixed(1)}%`);
    }

    // Validate Greeks budgets
    if (!this.validateGreeksBudgets(input, riskMetrics)) {
      return this.createRejectedPosition('Greeks budgets exceeded');
    }

    return {
      shares: contracts * 100, // Each contract = 100 shares
      notional: premium * contracts,
      riskAmount: Math.abs(input.capital * input.conviction * this.config.routes.expectedMoveFactor),
      riskPercent: Math.abs(input.conviction * this.config.routes.expectedMoveFactor),
      stopDistancePercent: input.expectedMove / input.entryPrice,
      advParticipationPercent: 0, // Not applicable for options
      slippageEstimateBps: friction.estimatedSlippage,
      canExecute: true,
      rejectionReason: '',
      contracts,
      premium,
      breakeven: this.calculateBreakeven(input),
      maxLoss: this.calculateMaxLoss(input, premium, contracts),
      maxGain: this.calculateMaxGain(input, premium, contracts),
      greeks,
      friction,
      riskMetrics
    };
  }

  /**
   * Select optimal route based on market context
   */
  selectOptimalRoute(context: RouteSelectionContext): OptionRoute {
    const routes = this.scoreRoutes(context);

    // Sort by confidence score
    routes.sort((a, b) => b.confidence - a.confidence);

    return routes[0];
  }

  /**
   * Score all available routes for given context
   */
  private scoreRoutes(context: RouteSelectionContext): OptionRoute[] {
    const routes: OptionRoute[] = [];

    for (const routeType of context.availableRoutes) {
      const score = this.scoreRoute(routeType, context);
      routes.push({
        ...routeType,
        confidence: score.confidence,
        expectedPnL: score.expectedPnL,
        maxLoss: score.maxLoss,
        frictionRatio: score.frictionRatio,
        thetaDay: score.thetaDay,
        context: score.rationale
      });
    }

    return routes;
  }

  /**
   * Score individual route based on context
   */
  private scoreRoute(route: OptionRoute, context: RouteSelectionContext): {
    confidence: number;
    expectedPnL: number;
    maxLoss: number;
    frictionRatio: number;
    thetaDay: number;
    rationale: string;
  } {
    let confidence = 0.5; // Base confidence
    let expectedPnL = 0;
    let maxLoss = 0;
    let frictionRatio = 0;
    let thetaDay = 0;
    let rationale = '';

    // IV-aware bias
    if (route.type === 'vertical' && context.ivRank > this.config.routes.ivRankVerticalBias) {
      confidence += 0.2;
      rationale += 'High IV favors defined-risk verticals. ';
    } else if ((route.type === 'long_call' || route.type === 'long_put') && context.ivRank < 0.4) {
      confidence += 0.15;
      rationale += 'Low IV favors long options. ';
    }

    // Friction headroom
    if (context.frictionHeadroom > 0.8) {
      confidence += 0.1;
      rationale += 'Good friction headroom. ';
    }

    // Theta headroom
    if (context.thetaHeadroom > 0.8) {
      confidence += 0.1;
      rationale += 'Good theta headroom. ';
    }

    // Vega headroom
    if (context.vegaHeadroom > 0.8) {
      confidence += 0.1;
      rationale += 'Good vega headroom. ';
    }

    return {
      confidence: Math.min(1.0, confidence),
      expectedPnL,
      maxLoss,
      frictionRatio,
      thetaDay,
      rationale
    };
  }

  /**
   * Validate chain quality meets minimum standards
   */
  private validateChainQuality(quality: ChainQuality): boolean {
    return quality.overall > 0.6 && // Decent overall quality
           quality.spreadScore > 0.7 && // Good spreads
           quality.volumeScore > 0.5 && // Reasonable volume
           quality.oiScore > 0.5; // Reasonable open interest
  }

  /**
   * Validate expected move vs target
   */
  private validateExpectedMove(input: OptionsPositionInput): boolean {
    const maxTarget = input.expectedMove * this.config.routes.expectedMoveFactor;
    return input.conviction <= maxTarget;
  }

  /**
   * Calculate premium for the position
   */
  private calculatePremium(input: OptionsPositionInput): number {
    // This would integrate with actual options pricing
    // For now, return estimated premium based on expected move
    const basePremium = input.expectedMove * input.conviction * 0.3; // Rough estimate
    return Math.max(basePremium, 0.05); // Minimum $0.05 per contract
  }

  /**
   * Calculate number of contracts
   */
  private calculateContracts(input: OptionsPositionInput, premium: number): number {
    const maxRisk = input.capital * POOR_CAPITAL_MODE.risk.perTradeRiskPct;
    const maxPremium = maxRisk / input.conviction; // Conservative sizing

    const contracts = Math.floor(maxPremium / premium);

    // Must have at least 1 contract
    return Math.max(1, contracts);
  }

  /**
   * Calculate net Greeks for the position
   */
  private calculateNetGreeks(input: OptionsPositionInput, contracts: number): OptionsPositionResult['greeks'] {
    // This would integrate with actual options data
    // For now, return estimated Greeks
    return {
      netDelta: contracts * 0.5, // Simplified
      netGamma: contracts * 0.02,
      netTheta: contracts * -0.02,
      netVega: contracts * 0.1
    };
  }

  /**
   * Calculate risk metrics
   */
  private calculateRiskMetrics(
    input: OptionsPositionInput,
    greeks: OptionsPositionResult['greeks'],
    contracts: number
  ): OptionsPositionResult['riskMetrics'] {
    return {
      thetaDay: Math.abs(greeks.netTheta) * input.entryPrice,
      gammaRisk: Math.abs(greeks.netGamma) * input.entryPrice,
      vegaDollar: Math.abs(greeks.netVega) * input.entryPrice,
      deltaDollar: Math.abs(greeks.netDelta) * input.entryPrice
    };
  }

  /**
   * Calculate friction (fees + slippage)
   */
  private calculateFriction(
    input: OptionsPositionInput,
    premium: number,
    contracts: number
  ): OptionsPositionResult['friction'] {
    const fees = contracts * this.config.friction.perContractFee;
    const slippage = this.estimateSlippage(input, premium, contracts);
    const totalCost = fees + slippage;
    const ratio = totalCost / (premium * contracts);

    return {
      estimatedSlippage: slippage,
      fees,
      ratio
    };
  }

  /**
   * Estimate slippage for options position
   */
  private estimateSlippage(input: OptionsPositionInput, premium: number, contracts: number): number {
    // Options slippage model - more conservative than equities
    const baseSlippage = input.spreadBps * premium / 100; // Spread-based
    const marketImpact = Math.sqrt(contracts) * premium * 0.01; // Size-based

    let slippage = baseSlippage + marketImpact;

    // Adjust for option type
    if (input.optionType === 'vertical') {
      slippage *= 1.2; // Verticals have wider spreads
    }

    // Adjust for liquidity
    if (input.chainQuality.volumeScore < 0.5) {
      slippage *= 1.5; // Illiquid = more slippage
    }

    return Math.round(slippage * 100) / 100; // Round to cents
  }

  /**
   * Calculate breakeven price
   */
  private calculateBreakeven(input: OptionsPositionInput): number {
    // Simplified - would need actual strike prices
    return input.entryPrice + (input.expectedMove * input.conviction);
  }

  /**
   * Calculate maximum loss
   */
  private calculateMaxLoss(input: OptionsPositionInput, premium: number, contracts: number): number {
    if (input.optionType === 'vertical') {
      // Defined risk - max loss is net debit paid
      return premium * contracts;
    } else {
      // Long options - max loss is premium paid
      return premium * contracts;
    }
  }

  /**
   * Calculate maximum gain
   */
  private calculateMaxGain(input: OptionsPositionInput, premium: number, contracts: number): number {
    if (input.optionType === 'vertical') {
      // Limited upside - credit received minus debit paid
      return (input.expectedMove * input.entryPrice - premium) * contracts;
    } else {
      // Long options - theoretically unlimited
      return (input.expectedMove * input.entryPrice - premium) * contracts;
    }
  }

  /**
   * Validate Greeks budgets
   */
  private validateGreeksBudgets(input: OptionsPositionInput, riskMetrics: OptionsPositionResult['riskMetrics']): boolean {
    // Gamma check
    const gammaMax = this.config.greeks.gammaPerTradeMax[input.isLeveragedETF ? 'leveragedETF' : 'default'];
    if (Math.abs(riskMetrics.gammaRisk) > gammaMax * input.capital) {
      return false;
    }

    // Theta check (governor)
    if (riskMetrics.thetaDay > this.config.greeks.thetaGovernor.clampThreshold * input.capital) {
      return false;
    }

    return true;
  }

  /**
   * Create rejected position result
   */
  private createRejectedPosition(reason: string): OptionsPositionResult {
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
      contracts: 0,
      premium: 0,
      breakeven: 0,
      maxLoss: 0,
      maxGain: 0,
      greeks: { netDelta: 0, netGamma: 0, netTheta: 0, netVega: 0 },
      friction: { estimatedSlippage: 0, fees: 0, ratio: 0 },
      riskMetrics: { thetaDay: 0, gammaRisk: 0, vegaDollar: 0, deltaDollar: 0 }
    };
  }
}

// Singleton instance
export const optionsPositionSizer = new OptionsPositionSizer();
