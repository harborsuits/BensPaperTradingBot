/**
 * Crypto Post-Trade Proof System
 * Verifies crypto trades against real exchange data
 * Uses REAL ONLY data from CryptoMarketRecorder for validation
 */

const { runtime } = require('./runtimeConfig');
const { CryptoMarketRecorder } = require('./cryptoMarketRecorder');

class CryptoPostTradeProver {
  constructor() {
    this.EXECUTION_BOUNDS = {
      MAX_SLIPPAGE_MULTIPLIER: 1.5, // Actual slippage <= planned * 1.5
      MIN_FILL_PCT: 0.95, // Must fill at least 95% for crypto (higher than options)
      CASH_BUFFER_PCT: 0.02, // Smaller buffer for crypto
      ORDER_BOOK_DEPTH_MIN: 5, // Minimum order book depth
      MAX_ORDER_BOOK_AGE_MS: 2000 // Order book max age (crypto moves fast)
    };

    this.cryptoRecorder = new CryptoMarketRecorder();
  }

  /**
   * Verify crypto post-trade execution against pre-trade promises
   */
  async verifyExecution(preTrade, postTrade) {
    // Enforce production safety - real-only mode
    runtime.enforceProductionSafety('crypto-post-trade-proof', { preTrade, postTrade });

    // Record crypto trade data for future validation
    await this.recordCryptoFillData(postTrade);

    const proofs = {
      timestamp: new Date().toISOString(),
      data_mode: runtime.DATA_MODE,
      tradeId: postTrade.id,
      execution: await this.proveCryptoExecutionBounds(preTrade, postTrade),
      structure: this.proveCryptoStructureBounds(preTrade, postTrade),
      cash: await this.proveCryptoCashBounds(preTrade, postTrade),
      orderbook: await this.proveCryptoOrderBookDepth(preTrade, postTrade),
      slippage: await this.proveCryptoSlippageWithinPlan(preTrade, postTrade),
      fees: this.proveCryptoFeeCompliance(preTrade, postTrade),
      overall: { passed: true, reasons: [] }
    };

    // Aggregate results
    const failed = Object.values(proofs).filter(p => p && p.passed === false);
    proofs.overall.passed = failed.length === 0;
    proofs.overall.reasons = failed.flatMap(p => p.reasons || []);

    return proofs;
  }

  /**
   * Record crypto fill data for proof validation
   */
  async recordCryptoFillData(postTrade) {
    try {
      // Record in both regular and crypto-specific tables
      if (postTrade.id && postTrade.symbol && postTrade.side && postTrade.price && postTrade.qty) {
        // Record fill in regular fills table
        await this.cryptoRecorder.recordFill(
          postTrade.id,
          postTrade.side,
          postTrade.price,
          postTrade.qty,
          postTrade.fees || 0,
          postTrade.timestamp || new Date().toISOString(),
          postTrade.brokerAttestation
        );

        // Record crypto-specific data
        if (postTrade.orderBook) {
          await this.cryptoRecorder.recordCryptoOrderBook(
            postTrade.symbol,
            postTrade.orderBook,
            postTrade.timestamp || new Date().toISOString(),
            new Date().toISOString(),
            postTrade.exchange || 'binance'
          );
        }
      }
    } catch (error) {
      console.error('Failed to record crypto fill data:', error);
      // Don't fail proof for recording errors
    }
  }

  /**
   * Prove crypto execution bounds - higher standards than options
   */
  async proveCryptoExecutionBounds(preTrade, postTrade) {
    const result = { passed: true, reasons: [] };

    const plannedSlippage = preTrade.executionPlan?.maxSlippage || 0.03; // 3% for crypto
    const actualSlippage = postTrade.actualSlippage || 0;

    result.plannedSlippage = plannedSlippage;
    result.actualSlippage = actualSlippage;
    result.fillPct = postTrade.fillPct || 1.0;

    // Slippage bound - crypto can move fast
    if (actualSlippage > plannedSlippage * this.EXECUTION_BOUNDS.MAX_SLIPPAGE_MULTIPLIER) {
      result.passed = false;
      result.reasons.push(`CRYPTO_SLIPPAGE_EXCEEDED: ${actualSlippage.toFixed(4)} > ${(plannedSlippage * this.EXECUTION_BOUNDS.MAX_SLIPPAGE_MULTIPLIER).toFixed(4)}`);
    }

    // Fill quality bound - crypto requires higher fill rates
    if (result.fillPct < this.EXECUTION_BOUNDS.MIN_FILL_PCT) {
      result.passed = false;
      result.reasons.push(`INSUFFICIENT_CRYPTO_FILL: ${(result.fillPct * 100).toFixed(1)}% < ${(this.EXECUTION_BOUNDS.MIN_FILL_PCT * 100).toFixed(1)}%`);
    }

    return result;
  }

  /**
   * Prove crypto structure bounds - net debit only, no leverage abuse
   */
  proveCryptoStructureBounds(preTrade, postTrade) {
    const result = { passed: true, reasons: [] };

    // Net debit verification
    const promisedNetDebit = preTrade.proof?.structure?.netDebit || 0;
    const actualNetDebit = postTrade.netDebit || 0;

    result.promisedNetDebit = promisedNetDebit;
    result.actualNetDebit = actualNetDebit;

    if (Math.abs(actualNetDebit - promisedNetDebit) > 0.001) { // $0.001 tolerance for crypto
      result.passed = false;
      result.reasons.push(`CRYPTO_NET_DEBIT_MISMATCH: promised $${promisedNetDebit.toFixed(4)}, actual $${actualNetDebit.toFixed(4)}`);
    }

    // No credit trades allowed
    if (actualNetDebit < 0) {
      result.passed = false;
      result.reasons.push(`CRYPTO_CREDIT_EXECUTED: $${actualNetDebit.toFixed(4)} credit violates cash-only policy`);
    }

    return result;
  }

  /**
   * Prove crypto cash bounds - verify balances after trade
   */
  async proveCryptoCashBounds(preTrade, postTrade) {
    const result = { passed: true, reasons: [] };

    const promisedCost = preTrade.sizing?.notional || 0;
    const actualCost = postTrade.totalCost || 0;

    result.promisedCost = promisedCost;
    result.actualCost = actualCost;

    // Cost bound with fee consideration
    const costTolerance = promisedCost * 0.05; // 5% tolerance for fees
    if (Math.abs(actualCost - promisedCost) > costTolerance) {
      result.passed = false;
      result.reasons.push(`CRYPTO_COST_MISMATCH: promised $${promisedCost.toFixed(4)}, actual $${actualCost.toFixed(4)} (±$${costTolerance.toFixed(4)})`);
    }

    // Cash buffer verification
    const postTradeCash = postTrade.cashAfter || 0;
    const minCashBuffer = promisedCost * this.EXECUTION_BOUNDS.CASH_BUFFER_PCT;

    if (postTradeCash < minCashBuffer) {
      result.passed = false;
      result.reasons.push(`CRYPTO_INSUFFICIENT_CASH_BUFFER: $${postTradeCash.toFixed(4)} < $${minCashBuffer.toFixed(4)}`);
    }

    return result;
  }

  /**
   * Prove crypto order book depth - ensure sufficient liquidity
   */
  async proveCryptoOrderBookDepth(preTrade, postTrade) {
    const result = { passed: true, reasons: [] };

    try {
      const orderBook = await this.cryptoRecorder.getCryptoOrderBookAtTime(
        postTrade.symbol,
        postTrade.timestamp,
        postTrade.exchange || 'binance',
        this.EXECUTION_BOUNDS.MAX_ORDER_BOOK_AGE_MS
      );

      if (!orderBook) {
        result.passed = false;
        result.reasons.push(`NO_CRYPTO_ORDER_BOOK: No order book data available at trade time`);
        result.orderBookAvailable = false;
        return result;
      }

      result.orderBookDepth = {
        bids: orderBook.bids.length,
        asks: orderBook.asks.length,
        timestamp: orderBook.ts_recv,
        age: Date.now() - new Date(orderBook.ts_recv).getTime()
      };

      // Check minimum depth
      if (orderBook.bids.length < this.EXECUTION_BOUNDS.ORDER_BOOK_DEPTH_MIN ||
          orderBook.asks.length < this.EXECUTION_BOUNDS.ORDER_BOOK_DEPTH_MIN) {
        result.passed = false;
        result.reasons.push(`INSUFFICIENT_CRYPTO_DEPTH: ${orderBook.bids.length} bids, ${orderBook.asks.length} asks (min ${this.EXECUTION_BOUNDS.ORDER_BOOK_DEPTH_MIN})`);
      }

      // Check order book freshness
      const age = Date.now() - new Date(orderBook.ts_recv).getTime();
      if (age > this.EXECUTION_BOUNDS.MAX_ORDER_BOOK_AGE_MS) {
        result.passed = false;
        result.reasons.push(`STALE_CRYPTO_ORDER_BOOK: ${age}ms old (max ${this.EXECUTION_BOUNDS.MAX_ORDER_BOOK_AGE_MS}ms)`);
      }

    } catch (error) {
      result.passed = false;
      result.reasons.push(`CRYPTO_ORDER_BOOK_ERROR: ${error.message}`);
    }

    return result;
  }

  /**
   * Prove crypto slippage within planned limits using real order book
   */
  async proveCryptoSlippageWithinPlan(preTrade, postTrade) {
    const result = { passed: true, reasons: [] };

    const plannedMax = preTrade.executionPlan?.maxSlippage || 0.03; // 3% default for crypto
    const actual = postTrade.actualSlippage || 0;

    result.plannedMax = plannedMax;
    result.actual = actual;

    // Try to get order book at fill time for real slippage calculation
    try {
      const orderBook = await this.cryptoRecorder.getCryptoOrderBookAtTime(
        postTrade.symbol,
        postTrade.timestamp,
        postTrade.exchange || 'binance',
        2000 // 2 second tolerance
      );

      if (orderBook) {
        // Calculate slippage against best bid/ask at fill time
        const bestBid = orderBook.bids[0]?.[0] || postTrade.price;
        const bestAsk = orderBook.asks[0]?.[0] || postTrade.price;

        // Use mid price of best bid/ask as reference
        const referencePrice = (bestBid + bestAsk) / 2;
        const realSlippage = Math.abs(postTrade.price - referencePrice) / referencePrice;

        result.realSlippageVsOrderBook = realSlippage;
        result.referencePrice = referencePrice;
        result.bestBid = bestBid;
        result.bestAsk = bestAsk;
        result.fillPrice = postTrade.price;

        // Use real slippage for validation
        result.within = realSlippage <= plannedMax * 1.5;

        if (!result.within) {
          result.passed = false;
          result.reasons.push(`CRYPTO_REAL_SLIPPAGE_EXCEEDED: ${realSlippage.toFixed(4)} > ${(plannedMax * 1.5).toFixed(4)} (vs order book mid $${referencePrice.toFixed(2)})`);
        }
      } else {
        // Fallback to reported slippage
        result.within = actual <= plannedMax * 1.5;
        result.usingFallback = true;

        if (!result.within) {
          result.passed = false;
          result.reasons.push(`CRYPTO_SLIPPAGE_EXCEEDED: ${actual.toFixed(4)} > ${(plannedMax * 1.5).toFixed(4)}`);
        }
      }
    } catch (error) {
      console.error('Error calculating crypto slippage:', error);
      result.within = actual <= plannedMax * 1.5;
      result.calculationError = error.message;

      if (!result.within) {
        result.passed = false;
        result.reasons.push(`CRYPTO_SLIPPAGE_EXCEEDED: ${actual.toFixed(4)} > ${(plannedMax * 1.5).toFixed(4)}`);
      }
    }

    return result;
  }

  /**
   * Prove crypto fee compliance - exchange fees within expected ranges
   */
  proveCryptoFeeCompliance(preTrade, postTrade) {
    const result = { passed: true, reasons: [] };

    const actualFees = postTrade.fees || 0;
    const notionalValue = postTrade.price * postTrade.qty;
    const feePct = actualFees / notionalValue;

    result.actualFees = actualFees;
    result.feePct = feePct;
    result.notionalValue = notionalValue;

    // Typical crypto exchange fees: 0.1% maker, 0.1% taker
    const maxExpectedFeePct = 0.002; // 0.2% max expected

    if (feePct > maxExpectedFeePct) {
      result.passed = false;
      result.reasons.push(`CRYPTO_FEES_TOO_HIGH: ${(feePct * 100).toFixed(3)}% > ${(maxExpectedFeePct * 100).toFixed(3)}%`);
    }

    // Minimum fee check (ensure fees were actually charged)
    const minExpectedFeePct = 0.0005; // 0.05% minimum
    if (feePct < minExpectedFeePct && notionalValue > 100) {
      result.passed = false;
      result.reasons.push(`CRYPTO_FEES_TOO_LOW: ${(feePct * 100).toFixed(3)}% < ${(minExpectedFeePct * 100).toFixed(3)}% (possible fee evasion)`);
    }

    return result;
  }

  /**
   * Generate crypto post-trade proof summary
   */
  generateProofSummary(proof) {
    let summary = `Crypto Post-Trade Proof: ${proof.overall.passed ? '✅ PASSED' : '❌ FAILED'}\n`;

    summary += `Trade ID: ${proof.tradeId}\n`;
    summary += `Executed: ${proof.timestamp}\n\n`;

    if (proof.execution) {
      summary += `Execution Bounds:\n`;
      summary += `  Fill: ${(proof.execution.fillPct * 100).toFixed(1)}% (min ${(this.EXECUTION_BOUNDS.MIN_FILL_PCT * 100).toFixed(1)}%)\n`;
      summary += `  Slippage: ${proof.execution.actualSlippage.toFixed(4)} (max ${(proof.execution.plannedSlippage * this.EXECUTION_BOUNDS.MAX_SLIPPAGE_MULTIPLIER).toFixed(4)})\n`;
    }

    if (proof.orderbook) {
      summary += `Order Book Depth:\n`;
      summary += `  Bids: ${proof.orderbook.orderBookDepth?.bids || 0}, Asks: ${proof.orderbook.orderBookDepth?.asks || 0}\n`;
      summary += `  Age: ${proof.orderbook.orderBookDepth?.age || 'N/A'}ms\n`;
    }

    if (proof.fees) {
      summary += `Fee Compliance:\n`;
      summary += `  Fee: ${(proof.fees.feePct * 100).toFixed(3)}% of $${proof.fees.notionalValue.toFixed(2)}\n`;
    }

    if (!proof.overall.passed) {
      summary += `\n❌ Violations:\n${proof.overall.reasons.map(r => `• ${r}`).join('\n')}`;
    }

    return summary;
  }
}

module.exports = { CryptoPostTradeProver };
