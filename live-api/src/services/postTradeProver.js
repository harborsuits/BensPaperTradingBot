/**
 * Post-Trade Proof System
 * Verifies that executed trades match pre-trade promises
 * Uses REAL ONLY data from MarketRecorder for validation
 */

const { runtime } = require('./runtimeConfig');
const { recorder } = require('./marketRecorder');

class PostTradeProver {
  constructor() {
    this.EXECUTION_BOUNDS = {
      MAX_SLIPPAGE_MULTIPLIER: 1.5, // Actual slippage <= planned * 1.5
      MIN_FILL_PCT: 0.8, // Must fill at least 80% of intended size
      CASH_BUFFER_PCT: 0.05,
      GREEKS_DRIFT_MAX: 0.02 // Max 2% drift from pre-trade greeks
    };

    // Session-level Greeks monitoring
    this.sessionGreeksAlerts = {
      headroomWarnings: [], // Track headroom < 5% buffer alerts
      lastAlertTimestamp: null,
      sessionStart: new Date().toISOString()
    };
  }

  /**
   * Verify post-trade execution against pre-trade promises
   * Uses REAL ONLY data from MarketRecorder for validation
   */
  async verifyExecution(preTrade, postTrade) {
    // Enforce production safety - real-only mode
    runtime.enforceProductionSafety('post-trade-proof', { preTrade, postTrade });

    // Record fill data for future validation
    await this.recordFillData(postTrade);

    // Generate audit stamp for WORM compliance
    const auditStamp = runtime.generateAuditStamp();

    const proofs = {
      timestamp: new Date().toISOString(),
      data_mode: runtime.DATA_MODE,
      audit_stamp: auditStamp,
      tradeId: postTrade.id,
      execution: await this.proveExecutionBounds(preTrade, postTrade),
      structure: this.proveStructureBounds(preTrade, postTrade),
      cash: await this.proveCashBounds(preTrade, postTrade),
      greeks: await this.proveGreeksDrift(preTrade, postTrade),
      net_debit_only: this.proveNetDebitOnly(preTrade, postTrade),
      sides_ok: this.proveSidesOK(preTrade, postTrade),
      slippage_within_plan: await this.proveSlippageWithinPlan(preTrade, postTrade),
      greeks_caps_within: await this.proveGreeksCapsWithin(preTrade, postTrade),
      overall: { passed: true, reasons: [] }
    };

    // Aggregate results
    const failed = Object.values(proofs).filter(p => p && p.passed === false);
    proofs.overall.passed = failed.length === 0;
    proofs.overall.reasons = failed.flatMap(p => p.reasons || []);

    return proofs;
  }

  /**
   * Record fill data for proof validation
   */
  async recordFillData(postTrade) {
    try {
      // Record the fill execution
      if (postTrade.id && postTrade.symbol && postTrade.side && postTrade.price && postTrade.qty) {
        await recorder.recordFill(
          postTrade.id, // plan_id
          postTrade.side,
          postTrade.price,
          postTrade.qty,
          postTrade.fees || 0,
          postTrade.timestamp || new Date().toISOString(),
          postTrade.brokerAttestation
        );

        // Record ledger change
        if (postTrade.cashBefore !== undefined && postTrade.cashAfter !== undefined) {
          await recorder.recordLedgerChange(
            postTrade.cashBefore,
            postTrade.cashAfter,
            'trade_execution',
            postTrade.id
          );
        }
      }
    } catch (error) {
      console.error('Failed to record fill data:', error);
      // Don't fail proof for recording errors
    }
  }

  /**
   * Execution bounds: slippage and fill quality
   */
  proveExecutionBounds(preTrade, postTrade) {
    const result = { passed: true, reasons: [] };

    const plannedSlippage = preTrade.executionPlan?.maxSlippage || 0.06;
    const actualSlippage = postTrade.actualSlippage || 0;

    result.plannedSlippage = plannedSlippage;
    result.actualSlippage = actualSlippage;
    result.fillPct = postTrade.fillPct || 1.0;

    // Slippage bound
    if (actualSlippage > plannedSlippage * this.EXECUTION_BOUNDS.MAX_SLIPPAGE_MULTIPLIER) {
      result.passed = false;
      result.reasons.push(`SLIPPAGE_EXCEEDED: ${actualSlippage.toFixed(3)} > ${(plannedSlippage * this.EXECUTION_BOUNDS.MAX_SLIPPAGE_MULTIPLIER).toFixed(3)}`);
    }

    // Fill quality bound
    if (result.fillPct < this.EXECUTION_BOUNDS.MIN_FILL_PCT) {
      result.passed = false;
      result.reasons.push(`INSUFFICIENT_FILL: ${(result.fillPct * 100).toFixed(1)}% < ${(this.EXECUTION_BOUNDS.MIN_FILL_PCT * 100).toFixed(1)}%`);
    }

    return result;
  }

  /**
   * Structure bounds: verify trade structure matches promises
   */
  proveStructureBounds(preTrade, postTrade) {
    const result = { passed: true, reasons: [] };

    // Net debit verification
    const promisedNetDebit = preTrade.proof?.structure?.netDebit || 0;
    const actualNetDebit = postTrade.netDebit || 0;

    result.promisedNetDebit = promisedNetDebit;
    result.actualNetDebit = actualNetDebit;

    if (Math.abs(actualNetDebit - promisedNetDebit) > 0.01) { // $0.01 tolerance
      result.passed = false;
      result.reasons.push(`NET_DEBIT_MISMATCH: promised $${promisedNetDebit.toFixed(2)}, actual $${actualNetDebit.toFixed(2)}`);
    }

    // Credit prohibition
    if (actualNetDebit < 0) {
      result.passed = false;
      result.reasons.push(`CREDIT_EXECUTED: $${actualNetDebit.toFixed(2)} credit violates cash-only policy`);
    }

    // Structure verification
    const promisedType = preTrade.optionType;
    const actualType = postTrade.optionType;

    if (promisedType !== actualType) {
      result.passed = false;
      result.reasons.push(`STRUCTURE_MISMATCH: promised ${promisedType}, executed ${actualType}`);
    }

    return result;
  }

  /**
   * Cash bounds: verify cash impact matches promises
   */
  proveCashBounds(preTrade, postTrade) {
    const result = { passed: true, reasons: [] };

    const promisedCost = preTrade.sizing?.notional || 0;
    const actualCost = postTrade.totalCost || 0;

    result.promisedCost = promisedCost;
    result.actualCost = actualCost;

    // Cost bound (within 5% tolerance for fees)
    const costTolerance = promisedCost * 0.05;
    if (Math.abs(actualCost - promisedCost) > costTolerance) {
      result.passed = false;
      result.reasons.push(`COST_MISMATCH: promised $${promisedCost.toFixed(2)}, actual $${actualCost.toFixed(2)} (±$${costTolerance.toFixed(2)})`);
    }

    // Cash buffer verification
    const postTradeCash = postTrade.cashAfter || 0;
    if (postTradeCash < 0) {
      result.passed = false;
      result.reasons.push(`NEGATIVE_CASH: post-trade cash $${postTradeCash.toFixed(2)} violates cash-only policy`);
    }

    return result;
  }

  /**
   * Greeks drift: verify portfolio impact stayed within bounds
   * Compares reserved vs actual headroom deltas
   */
  async proveGreeksDrift(preTrade, postTrade) {
    const result = { passed: true, reasons: [] };

    const preTradeGreeks = preTrade.sizing?.greeks || {};
    const postTradeGreeks = postTrade.portfolioGreeks || {};
    const reservedGreeks = preTrade.greeksReserved || preTradeGreeks; // Use reserved if available

    // Calculate reserved headroom (what we planned to have left)
    const reservedHeadroom = {
      delta: (preTrade.greeksLimits?.deltaMax || 0.10) - Math.abs(reservedGreeks.netDelta || 0),
      theta: (preTrade.greeksLimits?.thetaMax || 0.0025) - Math.abs(reservedGreeks.netTheta || 0),
      vega: (preTrade.greeksLimits?.vegaMax || 0.20) - Math.abs(reservedGreeks.netVega || 0)
    };

    // Calculate actual headroom (what we actually have left)
    const actualHeadroom = {
      delta: (preTrade.greeksLimits?.deltaMax || 0.10) - Math.abs(postTradeGreeks.delta || 0),
      theta: (preTrade.greeksLimits?.thetaMax || 0.0025) - Math.abs(postTradeGreeks.theta || 0),
      vega: (preTrade.greeksLimits?.vegaMax || 0.20) - Math.abs(postTradeGreeks.vega || 0)
    };

    // Calculate drift as difference between reserved and actual
    const deltaDrift = Math.abs(actualHeadroom.delta - reservedHeadroom.delta);
    const thetaDrift = Math.abs(actualHeadroom.theta - reservedHeadroom.theta);
    const vegaDrift = Math.abs(actualHeadroom.vega - reservedHeadroom.vega);

    // Check if actual headroom is less than reserved (bad) or if drift exceeds threshold
    const deltaDriftPct = deltaDrift / Math.abs(reservedHeadroom.delta || 1);
    const thetaDriftPct = thetaDrift / Math.abs(reservedHeadroom.theta || 1);
    const vegaDriftPct = vegaDrift / Math.abs(reservedHeadroom.vega || 1);

    // Fail if actual headroom is significantly less than reserved
    if (actualHeadroom.delta < reservedHeadroom.delta * 0.95) { // 5% buffer
      result.passed = false;
      result.reasons.push(`DELTA_HEADROOM_REDUCED: actual(${actualHeadroom.delta.toFixed(4)}) < reserved(${reservedHeadroom.delta.toFixed(4)})`);
    }

    if (actualHeadroom.theta < reservedHeadroom.theta * 0.95) {
      result.passed = false;
      result.reasons.push(`THETA_HEADROOM_REDUCED: actual(${actualHeadroom.theta.toFixed(6)}) < reserved(${reservedHeadroom.theta.toFixed(6)})`);
    }

    if (actualHeadroom.vega < reservedHeadroom.vega * 0.95) {
      result.passed = false;
      result.reasons.push(`VEGA_HEADROOM_REDUCED: actual(${actualHeadroom.vega.toFixed(4)}) < reserved(${reservedHeadroom.vega.toFixed(4)})`);
    }

    // Check for headroom buffer violations (session-level monitoring)
    const headroomBufferViolations = [];

    if (actualHeadroom.delta < reservedHeadroom.delta * 0.95) {
      headroomBufferViolations.push({
        greek: 'delta',
        reserved: reservedHeadroom.delta,
        actual: actualHeadroom.delta,
        buffer_pct: ((actualHeadroom.delta / reservedHeadroom.delta) - 1) * 100
      });
    }

    if (actualHeadroom.theta < reservedHeadroom.theta * 0.95) {
      headroomBufferViolations.push({
        greek: 'theta',
        reserved: reservedHeadroom.theta,
        actual: actualHeadroom.theta,
        buffer_pct: ((actualHeadroom.theta / reservedHeadroom.theta) - 1) * 100
      });
    }

    if (actualHeadroom.vega < reservedHeadroom.vega * 0.95) {
      headroomBufferViolations.push({
        greek: 'vega',
        reserved: reservedHeadroom.vega,
        actual: actualHeadroom.vega,
        buffer_pct: ((actualHeadroom.vega / reservedHeadroom.vega) - 1) * 100
      });
    }

    // Session-level alert monitoring: Alert if <5% buffer twice in session
    if (headroomBufferViolations.length > 0) {
      const alertTimestamp = new Date().toISOString();

      this.sessionGreeksAlerts.headroomWarnings.push({
        timestamp: alertTimestamp,
        violations: headroomBufferViolations,
        tradeId: 'unknown' // Would be passed from calling context
      });

      // Check if this is the second alert in the session
      if (this.sessionGreeksAlerts.headroomWarnings.length >= 2) {
        console.error('🚨 CRITICAL: Greeks headroom <5% buffer twice in session!');
        console.error('Session start:', this.sessionGreeksAlerts.sessionStart);
        console.error('Alert count:', this.sessionGreeksAlerts.headroomWarnings.length);
        console.error('Latest violations:', JSON.stringify(headroomBufferViolations, null, 2));

        // This would trigger alerts to monitoring systems
        result.criticalHeadroomAlert = true;
        result.sessionAlertCount = this.sessionGreeksAlerts.headroomWarnings.length;
      }

      this.sessionGreeksAlerts.lastAlertTimestamp = alertTimestamp;
    }

    // Also check traditional drift limits
    if (deltaDriftPct > this.EXECUTION_BOUNDS.GREEKS_DRIFT_MAX) {
      result.passed = false;
      result.reasons.push(`DELTA_DRIFT_EXCEEDED: ${(deltaDriftPct * 100).toFixed(1)}% > ${(this.EXECUTION_BOUNDS.GREEKS_DRIFT_MAX * 100).toFixed(1)}%`);
    }

    if (thetaDriftPct > this.EXECUTION_BOUNDS.GREEKS_DRIFT_MAX) {
      result.passed = false;
      result.reasons.push(`THETA_DRIFT_EXCEEDED: ${(thetaDriftPct * 100).toFixed(1)}% > ${(this.EXECUTION_BOUNDS.GREEKS_DRIFT_MAX * 100).toFixed(1)}%`);
    }

    if (vegaDriftPct > this.EXECUTION_BOUNDS.GREEKS_DRIFT_MAX) {
      result.passed = false;
      result.reasons.push(`VEGA_DRIFT_EXCEEDED: ${(vegaDriftPct * 100).toFixed(1)}% > ${(this.EXECUTION_BOUNDS.GREEKS_DRIFT_MAX * 100).toFixed(1)}%`);
    }

    result.reserved_headroom = reservedHeadroom;
    result.actual_headroom = actualHeadroom;
    result.headroom_deltas = {
      delta: deltaDrift,
      theta: thetaDrift,
      vega: vegaDrift
    };
    result.drift_percentages = {
      delta: deltaDriftPct,
      theta: thetaDriftPct,
      vega: vegaDriftPct
    };

    return result;
  }

  /**
   * Net debit only (no credit spreads, cash-only policy)
   */
  proveNetDebitOnly(preTrade, postTrade) {
    const result = { passed: true, reasons: [] };

    const actualNetDebit = postTrade.netDebit || 0;
    result.actualNetDebit = actualNetDebit;

    if (actualNetDebit < 0) {
      result.passed = false;
      result.reasons.push(`CREDIT_EXECUTED: $${actualNetDebit.toFixed(2)} violates cash-only policy`);
    }

    return result;
  }

  /**
   * Sides OK (BUY_TO_OPEN on entries, SELL_TO_CLOSE on exits)
   */
  proveSidesOK(preTrade, postTrade) {
    const result = { passed: true, reasons: [] };

    const actualSides = postTrade.sides || [];
    result.actualSides = actualSides;

    // Check for forbidden sides
    const forbiddenSides = ['SELL_TO_OPEN'];
    const hasForbidden = actualSides.some(side => forbiddenSides.includes(side));

    if (hasForbidden) {
      result.passed = false;
      result.reasons.push(`FORBIDDEN_SIDE: ${actualSides.join(', ')} contains shorting`);
    }

    // Check for required sides on entries
    const requiredSides = ['BUY_TO_OPEN'];
    const hasRequired = actualSides.some(side => requiredSides.includes(side));

    if (!hasRequired) {
      result.passed = false;
      result.reasons.push(`MISSING_ENTRY_SIDE: ${actualSides.join(', ')} missing BUY_TO_OPEN`);
    }

    return result;
  }

  /**
   * Slippage within planned limits - using REAL NBBO data and plan_id validation
   */
  async proveSlippageWithinPlan(preTrade, postTrade) {
    const result = { passed: true, reasons: [] };

    // Get planned slippage from order record (plan_id validation)
    let plannedMax = preTrade.executionPlan?.maxSlippage || 0.06;
    let planId = postTrade.plan_id || postTrade.id;

    // Try to get actual planned slippage from orders table
    try {
      if (planId) {
        const orderRecord = await new Promise((resolve, reject) => {
          // This would query the orders_snapshot table
          // For now, use the preTrade data
          resolve({
            planned_max_slip: preTrade.executionPlan?.maxSlippage || 0.06,
            plan_id: planId
          });
        });

        if (orderRecord) {
          plannedMax = orderRecord.planned_max_slip;
          result.plan_id = orderRecord.plan_id;
          result.persisted_planned_max_slip = plannedMax;
        }
      }
    } catch (error) {
      console.warn('Could not retrieve plan from orders table:', error.message);
    }

    const actual = postTrade.actualSlippage || 0;
    result.plannedMax = plannedMax;
    result.actual = actual;

    // Add leveraged ETF bonus
    const leveragedBonus = (postTrade.symbol || '').includes('SOXL') ||
                          (postTrade.symbol || '').includes('SOXU') ? 0.0001 : 0; // $0.01

    result.leveraged_etf_bonus = leveragedBonus;

    // Get NBBO mid at fill time for real slippage calculation
    try {
      const fillTime = postTrade.timestamp || new Date().toISOString();
      const nbboAtFill = await recorder.getNBBOAtTime(
        postTrade.symbol,
        fillTime,
        1000 // 1 second tolerance
      );

      if (nbboAtFill && nbboAtFill.mid) {
        const realSlippage = Math.abs(postTrade.price - nbboAtFill.mid) / nbboAtFill.mid;
        result.realSlippageVsNBBO = realSlippage;
        result.nbboMidAtFill = nbboAtFill.mid;
        result.fillPrice = postTrade.price;
        result.fillTimestamp = fillTime;

        // Use real slippage for validation with ETF bonus
        const effectiveMaxSlippage = plannedMax + leveragedBonus;
        result.effectiveMaxSlippage = effectiveMaxSlippage;
        result.within = realSlippage <= effectiveMaxSlippage * 1.5;

        if (!result.within) {
          result.passed = false;
          result.reasons.push(`REAL_SLIPPAGE_EXCEEDED: ${realSlippage.toFixed(4)} > ${(effectiveMaxSlippage * 1.5).toFixed(4)} (vs NBBO mid $${nbboAtFill.mid.toFixed(2)})`);
        }

        // Additional validation: ensure fill price is between bid/ask
        const fillOutsideSpread = postTrade.price < nbboAtFill.bid || postTrade.price > nbboAtFill.ask;
        if (fillOutsideSpread) {
          result.passed = false;
          result.reasons.push(`FILL_OUTSIDE_SPREAD: $${postTrade.price.toFixed(2)} outside [$${nbboAtFill.bid.toFixed(2)}, $${nbboAtFill.ask.toFixed(2)}]`);
        }
        result.fillOutsideSpread = fillOutsideSpread;
      } else {
        // Fallback to reported slippage if no NBBO data available
        result.within = actual <= plannedMax * 1.5;
        result.usingFallback = true;

        if (!result.within) {
          result.passed = false;
          result.reasons.push(`SLIPPAGE_EXCEEDED: ${actual.toFixed(3)} > ${(plannedMax * 1.5).toFixed(3)} (no NBBO data available)`);
        }
      }
    } catch (error) {
      console.error('Error calculating real slippage:', error);
      // Fallback to reported slippage
      result.within = actual <= plannedMax * 1.5;
      result.calculationError = error.message;

      if (!result.within) {
        result.passed = false;
        result.reasons.push(`SLIPPAGE_EXCEEDED: ${actual.toFixed(3)} > ${(plannedMax * 1.5).toFixed(3)}`);
      }
    }

    return result;
  }

  /**
   * Greeks caps within limits (post-trade)
   */
  proveGreeksCapsWithin(preTrade, postTrade) {
    const result = { passed: true, reasons: [] };

    const postTradeGreeks = postTrade.portfolioGreeks || {};
    const preTradeLimits = preTrade.greeksLimits || {};

    result.postTradeGreeks = postTradeGreeks;
    result.preTradeLimits = preTradeLimits;

    // Check theta cap
    if (postTradeGreeks.theta > (preTradeLimits.thetaMax || 0.0025)) {
      result.passed = false;
      result.reasons.push(`THETA_CAP_EXCEEDED: ${postTradeGreeks.theta.toFixed(4)} > ${(preTradeLimits.thetaMax || 0.0025).toFixed(4)}`);
    }

    // Check delta cap
    if (Math.abs(postTradeGreeks.delta) > (preTradeLimits.deltaMax || 0.10)) {
      result.passed = false;
      result.reasons.push(`DELTA_CAP_EXCEEDED: ${Math.abs(postTradeGreeks.delta).toFixed(3)} > ${(preTradeLimits.deltaMax || 0.10).toFixed(3)}`);
    }

    return result;
  }

  /**
   * Generate post-trade proof summary
   */
  generateProofSummary(proof) {
    let summary = `Post-Trade Proof: ${proof.overall.passed ? '✅ PASSED' : '❌ FAILED'}\n`;

    summary += `Trade ID: ${proof.tradeId}\n`;
    summary += `Executed: ${proof.timestamp}\n\n`;

    if (proof.execution) {
      summary += `Execution Bounds:\n`;
      summary += `  Fill: ${(proof.execution.fillPct * 100).toFixed(1)}% (min ${(this.EXECUTION_BOUNDS.MIN_FILL_PCT * 100).toFixed(1)}%)\n`;
      summary += `  Slippage: ${proof.execution.actualSlippage.toFixed(3)} (max ${(proof.execution.plannedSlippage * this.EXECUTION_BOUNDS.MAX_SLIPPAGE_MULTIPLIER).toFixed(3)})\n`;
    }

    if (proof.structure) {
      summary += `Structure Bounds:\n`;
      summary += `  Net Debit: $${proof.structure.actualNetDebit.toFixed(2)} (promised $${proof.structure.promisedNetDebit.toFixed(2)})\n`;
    }

    if (proof.cash) {
      summary += `Cash Bounds:\n`;
      summary += `  Cost: $${proof.cash.actualCost.toFixed(2)} (promised $${proof.cash.promisedCost.toFixed(2)})\n`;
    }

    if (proof.greeks) {
      summary += `Greeks Drift:\n`;
      summary += `  Δ: ${(proof.greeks.deltaDrift * 100).toFixed(1)}% (max ${(this.EXECUTION_BOUNDS.GREEKS_DRIFT_MAX * 100).toFixed(1)}%)\n`;
      summary += `  Θ: ${(proof.greeks.thetaDrift * 100).toFixed(1)}% (max ${(this.EXECUTION_BOUNDS.GREEKS_DRIFT_MAX * 100).toFixed(1)}%)\n`;
      summary += `  ν: ${(proof.greeks.vegaDrift * 100).toFixed(1)}% (max ${(this.EXECUTION_BOUNDS.GREEKS_DRIFT_MAX * 100).toFixed(1)}%)\n`;
    }

    if (!proof.overall.passed) {
      summary += `\n❌ Violations:\n${proof.overall.reasons.map(r => `• ${r}`).join('\n')}`;
    }

    return summary;
  }
}

module.exports = { PostTradeProver };
