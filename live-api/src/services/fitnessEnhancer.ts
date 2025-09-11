/**
 * Fitness Enhancer for Poor-Capital Mode
 *
 * Adds capital efficiency, slippage penalties, and realistic trading costs to fitness scoring
 * Ensures EvoTester optimizes for strategies that work with small accounts
 */

import { POOR_CAPITAL_MODE } from '../../config/poorCapitalMode';
import { OptionsFitnessMetrics } from './optionsTypes';

export interface StrategyMetrics {
  sharpeRatio: number;
  totalReturn: number;
  maxDrawdown: number;
  winRate: number;
  profitFactor: number;
  totalTrades: number;
  avgTradeDuration?: number;
}

export interface CapitalEfficiencyMetrics {
  totalPnL: number;
  totalCapitalAtRisk: number;
  slippageCostPercent: number;
  commissionCostPercent: number;
  advParticipationAvg: number;
  wholeShareEfficiency: number;
  correlationDiversityScore: number;
}

export interface EnhancedFitnessResult {
  baseFitness: number;
  enhancedFitness: number;
  capitalEfficiencyScore: number;
  slippagePenalty: number;
  advParticipationPenalty: number;
  wholeSharePenalty: number;
  diversityBonus: number;
  breakdown: {
    sharpe: number;
    returns: number;
    risk: number;
    capitalEfficiency: number;
    executionCosts: number;
    marketImpact: number;
    diversification: number;
  };
}

export class FitnessEnhancer {
  private config = POOR_CAPITAL_MODE.fitnessEnhancements;

  /**
   * Calculate enhanced fitness score with Poor-Capital Mode considerations
   */
  calculateEnhancedFitness(
    strategyMetrics: StrategyMetrics,
    capitalMetrics: CapitalEfficiencyMetrics,
    peerStrategies?: StrategyMetrics[]
  ): EnhancedFitnessResult {

    // Base fitness (traditional metrics)
    const baseFitness = this.calculateBaseFitness(strategyMetrics);

    // Enhanced components
    const capitalEfficiency = this.calculateCapitalEfficiency(capitalMetrics);
    const slippagePenalty = this.calculateSlippagePenalty(capitalMetrics);
    const advPenalty = this.calculateADVParticipationPenalty(capitalMetrics);
    const wholeSharePenalty = this.calculateWholeSharePenalty(capitalMetrics);
    const diversityBonus = this.calculateDiversityBonus(capitalMetrics, peerStrategies);

    // New Poor-Capital Mode enhancements
    const capitalEfficiencyFloorPenalty = this.calculateCapitalEfficiencyFloorPenalty(capitalMetrics);
    const frictionCapPenalty = this.calculateFrictionCapPenalty(capitalMetrics);

    // Weighted combination
    const capitalEfficiencyScore = capitalEfficiency * this.config.capitalEfficiencyWeight;
    const executionCosts = -(slippagePenalty + advPenalty + wholeSharePenalty) * this.config.slippagePenaltyWeight;
    const marketImpact = -advPenalty * this.config.advParticipationPenalty;
    const roundingImpact = -wholeSharePenalty * this.config.wholeShareRoundingPenalty;
    const diversification = diversityBonus * this.config.correlationDiversityBonus;

    // New enhancement terms
    const capitalFloorImpact = -capitalEfficiencyFloorPenalty * POOR_CAPITAL_MODE.fitnessEnhancements.capitalEfficiencyWeight;
    const frictionImpact = -frictionCapPenalty * POOR_CAPITAL_MODE.fitnessEnhancements.frictionPenaltyWeight;

    const enhancedFitness = baseFitness + capitalEfficiencyScore + executionCosts + marketImpact + roundingImpact + diversification + capitalFloorImpact + frictionImpact;

    return {
      baseFitness,
      enhancedFitness,
      capitalEfficiencyScore,
      slippagePenalty,
      advParticipationPenalty: advPenalty,
      wholeSharePenalty,
      diversityBonus,
      breakdown: {
        sharpe: baseFitness * 0.4, // Sharpe is 40% of base
        returns: baseFitness * 0.3, // Returns are 30% of base
        risk: baseFitness * 0.3, // Risk-adjusted is 30% of base
        capitalEfficiency: capitalEfficiencyScore,
        executionCosts,
        marketImpact,
        diversification,
        capitalFloorImpact,
        frictionImpact,
      },
    };
  }

  /**
   * Calculate base fitness from traditional metrics
   */
  private calculateBaseFitness(metrics: StrategyMetrics): number {
    let fitness = 0;

    // Sharpe ratio (40% weight)
    const sharpeScore = Math.max(0, Math.min(metrics.sharpeRatio, 3.0)) / 3.0; // Cap at 3.0
    fitness += sharpeScore * 0.4;

    // Risk-adjusted returns (30% weight)
    const riskAdjustedReturn = metrics.totalReturn * (1 - metrics.maxDrawdown);
    const returnScore = Math.max(0, Math.min(riskAdjustedReturn, 2.0)) / 2.0; // Cap at 200%
    fitness += returnScore * 0.3;

    // Win rate and consistency (30% weight)
    const consistencyScore = metrics.winRate * Math.min(metrics.profitFactor, 3.0) / 3.0;
    fitness += consistencyScore * 0.3;

    return fitness;
  }

  /**
   * Calculate capital efficiency (P&L per dollar risked)
   */
  private calculateCapitalEfficiency(metrics: CapitalEfficiencyMetrics): number {
    if (metrics.totalCapitalAtRisk === 0) return 0;

    const pnlPerDollarRisked = metrics.totalPnL / metrics.totalCapitalAtRisk;

    // Score based on efficiency
    if (pnlPerDollarRisked >= 0.02) return 1.0; // $2 profit per $100 risked (excellent)
    if (pnlPerDollarRisked >= 0.015) return 0.8; // $1.50 profit per $100 risked (very good)
    if (pnlPerDollarRisked >= 0.01) return 0.6; // $1 profit per $100 risked (good)
    if (pnlPerDollarRisked >= 0.005) return 0.4; // $0.50 profit per $100 risked (fair)
    if (pnlPerDollarRisked >= 0.002) return 0.2; // $0.20 profit per $100 risked (poor)
    if (pnlPerDollarRisked >= 0) return 0.1; // Break-even

    return 0; // Losing strategy
  }

  /**
   * Calculate slippage penalty
   */
  private calculateSlippagePenalty(metrics: CapitalEfficiencyMetrics): number {
    const slippagePercent = metrics.slippageCostPercent;

    // Penalty increases with slippage
    if (slippagePercent <= 0.001) return 0; // ≤0.1% slippage (excellent)
    if (slippagePercent <= 0.002) return 0.1; // ≤0.2% slippage (good)
    if (slippagePercent <= 0.005) return 0.3; // ≤0.5% slippage (fair)
    if (slippagePercent <= 0.01) return 0.6; // ≤1.0% slippage (poor)
    if (slippagePercent <= 0.02) return 0.8; // ≤2.0% slippage (very poor)

    return 1.0; // >2.0% slippage (unacceptable)
  }

  /**
   * Calculate ADV participation penalty
   */
  private calculateADVParticipationPenalty(metrics: CapitalEfficiencyMetrics): number {
    const participation = metrics.advParticipationAvg;

    // Penalty for moving too much of the daily volume
    if (participation <= 0.0001) return 0; // ≤0.01% of ADV (excellent)
    if (participation <= 0.0002) return 0.1; // ≤0.02% of ADV (very good)
    if (participation <= 0.0005) return 0.3; // ≤0.05% of ADV (good)
    if (participation <= 0.001) return 0.6; // ≤0.10% of ADV (fair)
    if (participation <= 0.002) return 0.8; // ≤0.20% of ADV (poor)

    return 1.0; // >0.20% of ADV (unacceptable market impact)
  }

  /**
   * Calculate whole share efficiency penalty
   */
  private calculateWholeSharePenalty(metrics: CapitalEfficiencyMetrics): number {
    const efficiency = metrics.wholeShareEfficiency;

    // Penalty for strategies that don't work well with whole shares
    if (efficiency >= 0.95) return 0; // ≥95% efficient (excellent)
    if (efficiency >= 0.90) return 0.1; // ≥90% efficient (very good)
    if (efficiency >= 0.80) return 0.2; // ≥80% efficient (good)
    if (efficiency >= 0.70) return 0.4; // ≥70% efficient (fair)
    if (efficiency >= 0.60) return 0.6; // ≥60% efficient (poor)
    if (efficiency >= 0.50) return 0.8; // ≥50% efficient (very poor)

    return 1.0; // <50% efficient (unacceptable)
  }

  /**
   * Calculate diversity bonus (reward uncorrelated strategies)
   */
  private calculateDiversityBonus(
    metrics: CapitalEfficiencyMetrics,
    peerStrategies?: StrategyMetrics[]
  ): number {
    if (!peerStrategies || peerStrategies.length === 0) return 0;

    const diversityScore = metrics.correlationDiversityScore;

    // Bonus for uncorrelated strategies
    if (diversityScore >= 0.8) return 0.3; // Very uncorrelated (good diversification)
    if (diversityScore >= 0.6) return 0.2; // Moderately uncorrelated
    if (diversityScore >= 0.4) return 0.1; // Somewhat uncorrelated
    if (diversityScore >= 0.2) return 0.05; // Slightly uncorrelated

    return 0; // Correlated (no diversification benefit)
  }

  /**
   * Get fitness improvement suggestions
   */
  getImprovementSuggestions(result: EnhancedFitnessResult): string[] {
    const suggestions: string[] = [];

    if (result.capitalEfficiencyScore < 0.5) {
      suggestions.push("Improve capital efficiency - focus on higher P&L per dollar risked");
    }

    if (result.slippagePenalty > 0.3) {
      suggestions.push("Reduce slippage - use limit orders and avoid large positions in illiquid stocks");
    }

    if (result.advParticipationPenalty > 0.3) {
      suggestions.push("Reduce market impact - decrease position sizes in low-volume stocks");
    }

    if (result.wholeSharePenalty > 0.3) {
      suggestions.push("Improve whole-share efficiency - avoid strategies requiring fractional shares");
    }

    if (result.diversityBonus < 0.1) {
      suggestions.push("Improve diversification - reduce correlation with other strategies");
    }

    if (result.baseFitness < 0.6) {
      suggestions.push("Improve core metrics - increase Sharpe ratio and risk-adjusted returns");
    }

    return suggestions;
  }

  /**
   * Check if strategy meets Poor-Capital Mode promotion criteria
   */
  meetsPromotionCriteria(
    strategyMetrics: StrategyMetrics,
    capitalMetrics: CapitalEfficiencyMetrics
  ): { eligible: boolean; reasons: string[] } {

    const reasons: string[] = [];
    let eligible = true;

    const promotion = POOR_CAPITAL_MODE.promotion;

    // Core metrics
    if (strategyMetrics.totalTrades < promotion.minTrades) {
      eligible = false;
      reasons.push(`Insufficient trades: ${strategyMetrics.totalTrades}/${promotion.minTrades}`);
    }

    if (strategyMetrics.sharpeRatio < promotion.minSharpe) {
      eligible = false;
      reasons.push(`Sharpe ratio too low: ${strategyMetrics.sharpeRatio.toFixed(2)}/${promotion.minSharpe}`);
    }

    if (strategyMetrics.maxDrawdown > promotion.maxDD) {
      eligible = false;
      reasons.push(`Max drawdown too high: ${(strategyMetrics.maxDrawdown * 100).toFixed(1)}%/${(promotion.maxDD * 100).toFixed(1)}%`);
    }

    if (strategyMetrics.winRate < promotion.minWin) {
      eligible = false;
      reasons.push(`Win rate too low: ${(strategyMetrics.winRate * 100).toFixed(1)}%/${(promotion.minWin * 100).toFixed(1)}%`);
    }

    // Capital efficiency
    const pnlPerDollarRisked = capitalMetrics.totalPnL / capitalMetrics.totalCapitalAtRisk;
    if (pnlPerDollarRisked < 0.005) { // $0.50 per $100 risked minimum
      eligible = false;
      reasons.push(`Poor capital efficiency: $${(pnlPerDollarRisked * 100).toFixed(2)} per $100 risked`);
    }

    // Execution costs
    if (capitalMetrics.slippageCostPercent > 0.015) { // 1.5% max slippage
      eligible = false;
      reasons.push(`Slippage too high: ${(capitalMetrics.slippageCostPercent * 100).toFixed(2)}%`);
    }

    return { eligible, reasons };
  }

  /**
   * Enhanced fitness calculation for options strategies
   */
  calculateOptionsEnhancedFitness(
    strategyMetrics: StrategyMetrics,
    optionsMetrics: OptionsFitnessMetrics,
    peerStrategies: StrategyMetrics[] = []
  ): EnhancedFitnessResult {
    // Base fitness (traditional metrics)
    const baseFitness = this.calculateBaseFitness(strategyMetrics);

    // Enhanced components
    const capitalEfficiency = this.calculateCapitalEfficiency(optionsMetrics);
    const slippagePenalty = this.calculateSlippagePenalty(optionsMetrics);
    const advPenalty = this.calculateADVParticipationPenalty(optionsMetrics);
    const wholeSharePenalty = this.calculateWholeSharePenalty(optionsMetrics);
    const diversityBonus = this.calculateDiversityBonus(optionsMetrics, peerStrategies);

    // Options-specific enhancements
    const frictionPenalty = this.calculateOptionsFrictionPenalty(optionsMetrics);
    const thetaEfficiency = this.calculateThetaEfficiency(optionsMetrics);
    const vegaEfficiency = this.calculateVegaEfficiency(optionsMetrics);
    const assignmentRiskPenalty = this.calculateAssignmentRiskPenalty(optionsMetrics);

    // Weighted combination
    const capitalEfficiencyScore = capitalEfficiency * this.config.capitalEfficiencyWeight;
    const executionCosts = -(slippagePenalty + advPenalty + wholeSharePenalty) * this.config.slippagePenaltyWeight;
    const marketImpact = -advPenalty * this.config.advParticipationPenalty;
    const roundingImpact = -wholeSharePenalty * this.config.wholeShareRoundingPenalty;
    const diversification = diversityBonus * this.config.correlationDiversityBonus;

    // Options-specific terms
    const optionsFrictionImpact = -frictionPenalty * POOR_CAPITAL_MODE.fitnessEnhancements.frictionPenaltyWeight;
    const thetaEfficiencyBonus = thetaEfficiency * 0.15; // 15% weight on theta efficiency
    const vegaEfficiencyBonus = vegaEfficiency * 0.10; // 10% weight on vega efficiency
    const assignmentRiskImpact = -assignmentRiskPenalty * 0.05; // 5% penalty for assignment risk

    const enhancedFitness = baseFitness + capitalEfficiencyScore + executionCosts + marketImpact +
                          roundingImpact + diversification + optionsFrictionImpact +
                          thetaEfficiencyBonus + vegaEfficiencyBonus + assignmentRiskImpact;

    return {
      baseFitness,
      enhancedFitness,
      capitalEfficiencyScore,
      slippagePenalty,
      advParticipationPenalty: advPenalty,
      wholeSharePenalty,
      diversityBonus,
      breakdown: {
        sharpe: baseFitness * 0.4,
        returns: baseFitness * 0.3,
        risk: baseFitness * 0.3,
        capitalEfficiency: capitalEfficiencyScore,
        executionCosts,
        marketImpact,
        diversification,
      },
    };
  }

  /**
   * Calculate options-specific friction penalty
   */
  private calculateOptionsFrictionPenalty(metrics: OptionsFitnessMetrics): number {
    const frictionCap = POOR_CAPITAL_MODE.fitnessEnhancements.frictionCap;

    if (metrics.frictionRatio <= frictionCap) return 0;

    // Penalty increases as friction exceeds cap
    return Math.min(1.0, (metrics.frictionRatio - frictionCap) / frictionCap);
  }

  /**
   * Calculate theta efficiency bonus
   */
  private calculateThetaEfficiency(metrics: OptionsFitnessMetrics): number {
    if (metrics.thetaCollected <= 0) return 0;

    // Bonus for strategies that collect theta efficiently
    const thetaPerRisk = metrics.thetaCollected / metrics.totalCapitalAtRisk;

    // Scale bonus (max 1.0 for strategies collecting >2% theta per unit risk)
    return Math.min(1.0, thetaPerRisk / 0.02);
  }

  /**
   * Calculate vega efficiency bonus
   */
  private calculateVegaEfficiency(metrics: OptionsFitnessMetrics): number {
    if (metrics.totalPnL <= 0) return 0;

    // Bonus for strategies that profit from IV changes efficiently
    const vegaPnL = metrics.vegaPnl;
    const totalPnL = Math.abs(metrics.totalPnL);

    if (totalPnL === 0) return 0;

    const vegaContribution = vegaPnL / totalPnL;

    // Bonus for strategies where vega contributes positively to returns
    return Math.max(0, Math.min(1.0, vegaContribution));
  }

  /**
   * Calculate assignment risk penalty
   */
  private calculateAssignmentRiskPenalty(metrics: OptionsFitnessMetrics): number {
    // Penalty for strategies with high assignment risk
    return Math.min(1.0, metrics.assignmentRisk);
  }

  /**
   * Validate options strategy eligibility
   */
  validateOptionsStrategy(
    strategyMetrics: StrategyMetrics,
    optionsMetrics: OptionsFitnessMetrics
  ): { eligible: boolean; reasons: string[] } {
    const reasons: string[] = [];

    // Friction check
    if (optionsMetrics.frictionRatio > POOR_CAPITAL_MODE.fitnessEnhancements.frictionCap) {
      reasons.push(`Friction ratio too high: ${(optionsMetrics.frictionRatio * 100).toFixed(1)}%`);
    }

    // Assignment risk check
    if (optionsMetrics.assignmentRisk > 0.3) {
      reasons.push(`Assignment risk too high: ${(optionsMetrics.assignmentRisk * 100).toFixed(1)}%`);
    }

    // Chain quality check
    if (optionsMetrics.chainQuality < 0.5) {
      reasons.push(`Chain quality too low: ${(optionsMetrics.chainQuality * 100).toFixed(1)}%`);
    }

    // IV rank check
    if (optionsMetrics.ivRank > 0.9) {
      reasons.push(`IV rank too high: ${(optionsMetrics.ivRank * 100).toFixed(1)}%`);
    }

    return { eligible: reasons.length === 0, reasons };
  }

  /**
   * Calculate capital efficiency floor penalty
   * Penalizes strategies that don't achieve minimum $0.50 P&L per $ risked
   */
  private calculateCapitalEfficiencyFloorPenalty(metrics: CapitalEfficiencyMetrics): number {
    if (metrics.totalCapitalAtRisk === 0) return 0;

    const pnlPerDollarRisked = metrics.totalPnL / metrics.totalCapitalAtRisk;
    const floor = POOR_CAPITAL_MODE.fitnessEnhancements.capitalEfficiencyFloor;

    if (pnlPerDollarRisked >= floor) return 0; // No penalty if above floor

    // Penalty increases as efficiency drops below floor
    return Math.max(0, (floor - pnlPerDollarRisked) / floor);
  }

  /**
   * Calculate friction cap penalty
   * Penalizes strategies where (slippage+fees)/grossPnl > 25%
   */
  private calculateFrictionCapPenalty(metrics: CapitalEfficiencyMetrics): number {
    if (metrics.totalPnL === 0) return 0;

    const frictionRatio = (metrics.slippageCostPercent + 0.001) / Math.abs(metrics.totalPnL); // Add small fee estimate
    const cap = POOR_CAPITAL_MODE.fitnessEnhancements.frictionCap;

    if (frictionRatio <= cap) return 0; // No penalty if below cap

    // Penalty increases as friction exceeds cap
    return Math.min(1.0, (frictionRatio - cap) / cap);
  }
}

// Singleton instance
export const fitnessEnhancer = new FitnessEnhancer();
