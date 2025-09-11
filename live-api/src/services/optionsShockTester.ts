/**
 * Options Shock Tester for EvoTester
 * Pre-trade risk validation with price and volatility shocks
 */

interface ShockTestResult {
  passed: boolean;
  worstCasePnL: number;
  stressTests: {
    priceShock: {
      up1Sigma: number;
      down1Sigma: number;
    };
    ivShock: {
      up10Pts: number;
      down10Pts: number;
    };
    combinedShock: {
      worstCase: number;
      thetaBudget: number;
      deltaBudget: number;
    };
  };
  reasons: string[];
  recommendation: 'approve' | 'reject' | 'review';
}

interface OptionsPosition {
  contracts: number;
  optionType: 'vertical' | 'long_call' | 'long_put';
  longStrike?: number;
  shortStrike?: number;
  expiry: Date;
  currentPremium: number;
  greeks: {
    delta: number;
    gamma: number;
    theta: number;
    vega: number;
  };
}

export class OptionsShockTester {
  private readonly MAX_PREMIUM_RISK = 0.02; // 2% of account max loss per trade
  private readonly SHOCK_PRICE_SIGMA = 1.0;
  private readonly SHOCK_IV_PTS = 10;
  private readonly THETA_BUDGET_MAX = 0.0025; // 0.25% of equity per day

  /**
   * Run comprehensive shock test on proposed options position
   */
  testPosition(
    position: OptionsPosition,
    accountSize: number,
    expectedMove: number,
    ivRank: number
  ): ShockTestResult {
    const results: ShockTestResult = {
      passed: true,
      worstCasePnL: 0,
      stressTests: {
        priceShock: { up1Sigma: 0, down1Sigma: 0 },
        ivShock: { up10Pts: 0, down10Pts: 0 },
        combinedShock: { worstCase: 0, thetaBudget: 0, deltaBudget: 0 }
      },
      reasons: [],
      recommendation: 'approve'
    };

    // Price shock tests (±1σ)
    results.stressTests.priceShock = this.testPriceShocks(position, expectedMove);

    // IV shock tests (±10 pts)
    results.stressTests.ivShock = this.testIVShocks(position, ivRank);

    // Combined worst-case scenario
    results.stressTests.combinedShock = this.testCombinedShock(
      position,
      expectedMove,
      ivRank,
      accountSize
    );

    // Validate against risk limits
    const maxLossThreshold = accountSize * this.MAX_PREMIUM_RISK;
    const thetaBudgetThreshold = accountSize * this.THETA_BUDGET_MAX;

    results.worstCasePnL = Math.min(
      results.stressTests.priceShock.down1Sigma,
      results.stressTests.priceShock.up1Sigma,
      results.stressTests.ivShock.up10Pts,
      results.stressTests.ivShock.down10Pts,
      results.stressTests.combinedShock.worstCase
    );

    // Check risk thresholds
    if (Math.abs(results.worstCasePnL) > maxLossThreshold) {
      results.passed = false;
      results.reasons.push(`Worst-case P&L (${results.worstCasePnL.toFixed(2)}) exceeds max premium risk ($${maxLossThreshold.toFixed(2)})`);
    }

    if (Math.abs(results.stressTests.combinedShock.thetaBudget) > thetaBudgetThreshold) {
      results.passed = false;
      results.reasons.push(`Theta budget (${results.stressTests.combinedShock.thetaBudget.toFixed(2)}) exceeds daily limit ($${thetaBudgetThreshold.toFixed(2)})`);
    }

    // Determine recommendation
    if (!results.passed) {
      results.recommendation = results.reasons.length > 1 ? 'reject' : 'review';
    }

    return results;
  }

  /**
   * Test price shocks (±1σ)
   */
  private testPriceShocks(position: OptionsPosition, expectedMove: number): { up1Sigma: number; down1Sigma: number } {
    const sigmaMove = expectedMove * this.SHOCK_PRICE_SIGMA;

    return {
      up1Sigma: this.calculatePnL(position, sigmaMove),
      down1Sigma: this.calculatePnL(position, -sigmaMove)
    };
  }

  /**
   * Test IV shocks (±10 pts)
   */
  private testIVShocks(position: OptionsPosition, currentIVR: number): { up10Pts: number; down10Pts: number } {
    const ivShock = this.SHOCK_IV_PTS / 100; // Convert to decimal

    return {
      up10Pts: this.calculateIVPnL(position, currentIVR + ivShock),
      down10Pts: this.calculateIVPnL(position, Math.max(0.01, currentIVR - ivShock))
    };
  }

  /**
   * Test combined worst-case scenario
   */
  private testCombinedShock(
    position: OptionsPosition,
    expectedMove: number,
    ivRank: number,
    accountSize: number
  ): { worstCase: number; thetaBudget: number; deltaBudget: number } {
    // Worst case: -1σ price + -10 IV pts + time decay
    const priceShock = -expectedMove * this.SHOCK_PRICE_SIGMA;
    const ivShock = -this.SHOCK_IV_PTS / 100;
    const timeShock = 1; // 1 day decay

    const worstCase = this.calculateCombinedPnL(position, priceShock, ivRank + ivShock, timeShock);

    // Calculate budget impacts
    const thetaBudget = Math.abs(position.greeks.theta) * position.contracts * 100; // Daily theta impact
    const deltaBudget = Math.abs(position.greeks.delta) * position.contracts * 100 * expectedMove; // Dollar delta impact

    return {
      worstCase,
      thetaBudget,
      deltaBudget
    };
  }

  /**
   * Calculate P&L for given price movement
   */
  private calculatePnL(position: OptionsPosition, priceMove: number): number {
    switch (position.optionType) {
      case 'vertical':
        return this.calculateVerticalPnL(position, priceMove);
      case 'long_call':
        return this.calculateLongCallPnL(position, priceMove);
      case 'long_put':
        return this.calculateLongPutPnL(position, priceMove);
      default:
        return 0;
    }
  }

  /**
   * Calculate vertical spread P&L
   */
  private calculateVerticalPnL(position: OptionsPosition, priceMove: number): number {
    if (!position.longStrike || !position.shortStrike) return 0;

    const width = Math.abs(position.shortStrike - position.longStrike);
    const isBullPut = position.longStrike < position.shortStrike;

    // Simplified: assume max loss at expiration
    return -position.currentPremium * position.contracts;
  }

  /**
   * Calculate long call P&L
   */
  private calculateLongCallPnL(position: OptionsPosition, priceMove: number): number {
    if (!position.longStrike) return 0;

    // Simplified: assume expiration value
    return -position.currentPremium * position.contracts;
  }

  /**
   * Calculate long put P&L
   */
  private calculateLongPutPnL(position: OptionsPosition, priceMove: number): number {
    if (!position.longStrike) return 0;

    // Simplified: assume expiration value
    return -position.currentPremium * position.contracts;
  }

  /**
   * Calculate P&L impact from IV change
   */
  private calculateIVPnL(position: OptionsPosition, newIV: number): number {
    // Simplified: vega impact
    const ivChange = newIV - 0.5; // Assume current IV = 50%
    return position.greeks.vega * position.contracts * 100 * ivChange;
  }

  /**
   * Calculate combined P&L (price + IV + time)
   */
  private calculateCombinedPnL(
    position: OptionsPosition,
    priceMove: number,
    newIV: number,
    daysDecay: number
  ): number {
    const pricePnL = this.calculatePnL(position, priceMove);
    const ivPnL = this.calculateIVPnL(position, newIV);
    const thetaPnL = position.greeks.theta * position.contracts * 100 * daysDecay;

    return pricePnL + ivPnL + thetaPnL;
  }

  /**
   * Generate shock test summary for DecisionTrace
   */
  generateShockSummary(result: ShockTestResult): string {
    let summary = `Shock Test: ${result.passed ? '✅ PASSED' : '❌ FAILED'} (${result.recommendation.toUpperCase()})\n`;

    summary += `Worst-case P&L: $${result.worstCasePnL.toFixed(2)}\n`;

    summary += `Price Shock (±1σ):\n`;
    summary += `  +1σ: $${result.stressTests.priceShock.up1Sigma.toFixed(2)}\n`;
    summary += `  -1σ: $${result.stressTests.priceShock.down1Sigma.toFixed(2)}\n`;

    summary += `IV Shock (±10pts):\n`;
    summary += `  +10pts: $${result.stressTests.ivShock.up10Pts.toFixed(2)}\n`;
    summary += `  -10pts: $${result.stressTests.ivShock.down10Pts.toFixed(2)}\n`;

    summary += `Combined Shock:\n`;
    summary += `  Worst: $${result.stressTests.combinedShock.worstCase.toFixed(2)}\n`;
    summary += `  Theta Budget: $${result.stressTests.combinedShock.thetaBudget.toFixed(2)}\n`;

    if (result.reasons.length > 0) {
      summary += `\nIssues:\n${result.reasons.map(r => `• ${r}`).join('\n')}`;
    }

    return summary;
  }
}

module.exports = { OptionsShockTester };
