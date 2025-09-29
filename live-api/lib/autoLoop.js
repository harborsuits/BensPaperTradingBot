'use strict';

const EventEmitter = require('events');
const http = require('http');
const { nanoid } = require('nanoid');
const { currentHealth } = require('./health');
const { buildTradePlan } = require('./planning');
const { preTradeGate } = require('./gate');
const { EnhancedRiskGate } = require('./enhancedGate');
const { DecisionCoordinator } = require('./decisionCoordinator');
const { StrategyAllocator } = require('./strategyAllocator');
const { CONFIG } = require('./config');
const ExpectedValueCalculator = require('./expectedValueCalculator');
const tradingConfig = require('./tradingConfig');
const tradingThresholds = require('../config/tradingThresholds');
// Poor capital mode configuration
const POOR_CAPITAL_MODE = {
  enabled: true,
  universe: {
    minPrice: 0.10,     // Allow penny stocks down to $0.10
    maxPrice: 1000,     // Allow higher priced stocks
    maxSpreadBps: 500   // Allow wider spreads (5%) for testing
  },
  risk: {
    maxRiskPerTrade: 0.002, // 0.2%
    maxPositionSize: 0.02, // 2%
  }
};
const fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));

// Brain integrator will be injected from minimal_server.js
let brainIntegrator = null;

// Strategy manager will be injected from server.js
let strategyManager = null;

// Import decisions bus for publishing proposals
// let decisionsBus;
// try {
//   decisionsBus = require('../src/decisions/bus');
// } catch (error) {
//   console.warn('Could not load decisions bus in AutoLoop:', error.message);
//   decisionsBus = { publish: () => {}, recent: () => [] };
// }

// Initialize enhanced components
const riskGate = new EnhancedRiskGate();
const coordinator = new DecisionCoordinator();
const allocator = new StrategyAllocator();

// Initialize circuit breaker for system protection
let circuitBreaker;
try {
  const CircuitBreaker = require('../services/circuitBreaker');
  circuitBreaker = new CircuitBreaker({
    maxFailuresPerWindow: 5,
    windowSizeMs: 60000, // 1 minute
    cooldownPeriodMs: 300000, // 5 minutes
    maxApiErrors: 3,
    maxOrderFailures: 5,
    maxDrawdownPercent: 0.05, // 5%
    maxDailyLossPercent: 0.02 // 2%
  });
  console.log('[AutoLoop] Circuit breaker initialized');
} catch (error) {
  console.error('[AutoLoop] Failed to initialize circuit breaker:', error.message);
}

// Initialize data validator for market data validation
let dataValidator;
try {
  const DataValidator = require('../services/dataValidator');
  dataValidator = new DataValidator({
    maxQuoteAgeMs: 5000, // 5 seconds
    maxSpreadPercent: 0.05, // 5%
    minVolume: 1000,
    minAvgVolume: 100000
  });
  console.log('[AutoLoop] Data validator initialized');
} catch (error) {
  console.error('[AutoLoop] Failed to initialize data validator:', error.message);
}


class AutoLoop extends EventEmitter {
  constructor(options = {}) {
    super();
    this.interval = options.interval || 30_000;
    this.symbols = options.symbols || ['SPY'];
    this.enabled = !!options.enabled;
    this.timer = null;
    this.isRunning = false;
    this.evCalculator = new ExpectedValueCalculator();
    this.lastRun = null;
    this.status = 'STOPPED';
    
    // Brain integrator for unified decision making
    this.brainIntegrator = options.brainIntegrator || null;
    
    // Strategy manager for accessing strategies
    strategyManager = options.strategyManager || null;
    
    // Capital management limits from AI policy
    this.capitalLimits = {
      maxTotalCapital: 60000,    // Increased to allow more positions
      maxPerTrade: 2000,         // Reasonable per-trade limit
      maxOpenTrades: 1000,       // No artificial limit - let capital/risk rules decide
      maxDailyTrades: 50,        // Prevent excessive trading
      minCashBuffer: 1000        // Always keep some cash
    };
    
    // Track usage
    this.dailyStats = {
      tradesExecuted: 0,
      capitalDeployed: 0,
      lastReset: new Date().toDateString()
    };
    
    // Performance recorder for learning
    this.performanceRecorder = options.performanceRecorder || null;
    this.enhancedRecorder = options.enhancedRecorder || null;
    
    // Dynamic discovery settings
    this.dynamicDiscovery = options.dynamicDiscovery !== false; // Default to true
    this.useDiamonds = options.useDiamonds !== false; // Default to true
    this.useScanner = options.useScanner !== false; // Default to true
    
    // Smart symbol tracking - stop wasting time on failures
    this.failedSymbols = new Map(); // symbol -> {lastCheck, failCount, lastScore}
    this.successfulSymbols = new Set();
    this.MAX_FAIL_COUNT = 10; // Increased to 10 fails before cooldown
    this.FAIL_COOLDOWN_MS = 15 * 60 * 1000; // Reduced to 15 minutes cooldown
  }

  start() {
    if (this.isRunning || !this.enabled) return;
    this.isRunning = true;
    this.status = 'RUNNING';
    console.log(`[AutoLoop] Starting with interval ${this.interval}ms for symbols: ${this.symbols.join(',')}`);
    this.timer = setInterval(() => this.runOnce(), this.interval);
    
    // Start brain integrator for position monitoring
    if (this.brainIntegrator) {
      this.brainIntegrator.start();
    }
  }

  stop() {
    if (!this.isRunning) return;
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
    this.isRunning = false;
    this.status = 'STOPPED';
    console.log('[AutoLoop] Stopped');
    
    // Stop brain integrator
    if (this.brainIntegrator) {
      this.brainIntegrator.stop();
    }
  }
  
  resetCircuitBreaker() {
    if (circuitBreaker) {
      circuitBreaker.close();
      console.log('[AutoLoop] Circuit breaker reset to CLOSED state');
      return true;
    }
    return false;
  }
  
  /**
   * Set or update the brain integrator
   */
  setBrainIntegrator(brainIntegrator) {
    // Stop old one if running
    if (this.brainIntegrator && this.isRunning) {
      this.brainIntegrator.stop();
    }
    
    this.brainIntegrator = brainIntegrator;
    
    // Start new one if we're running
    if (this.brainIntegrator && this.isRunning) {
      this.brainIntegrator.start();
    }
    
    console.log('[AutoLoop] Brain integrator updated');
  }
  
  async getOpenPositionCount() {
    try {
      const positionsResp = await fetch('http://localhost:4000/api/paper/positions');
      if (positionsResp.ok) {
        const positions = await positionsResp.json();
        // Only count positions with qty > 0 (not closed positions)
        const openPositions = Array.isArray(positions) 
          ? positions.filter(p => p.qty > 0).length 
          : 0;
        return openPositions;
      }
    } catch (error) {
      console.error('[AutoLoop] Error getting position count:', error.message);
    }
    return 0;
  }

  async cancelPendingOrders() {
    try {
      console.log('[AutoLoop] Checking for pending orders to cancel...');
      
      // Get all orders
      const ordersResp = await fetch('http://localhost:4000/api/paper/orders?limit=100');
      if (!ordersResp.ok) return;
      
      const ordersData = await ordersResp.json();
      const orders = ordersData.orders || ordersData.items || [];
      const pendingOrders = orders.filter(o => o.status === 'pending');
      
      if (pendingOrders.length === 0) {
        console.log('[AutoLoop] No pending orders to cancel');
        return;
      }
      
      console.log(`[AutoLoop] Found ${pendingOrders.length} pending orders to cancel`);
      
      // Cancel each pending order
      let cancelCount = 0;
      for (const order of pendingOrders) {
        try {
          const cancelResp = await fetch(`http://localhost:4000/api/paper/orders/${order.id}`, {
            method: 'DELETE'
          });
          
          if (cancelResp.ok) {
            cancelCount++;
            console.log(`[AutoLoop] Cancelled order ${order.id} for ${order.symbol}`);
          } else {
            console.error(`[AutoLoop] Failed to cancel order ${order.id}:`, await cancelResp.text());
          }
        } catch (error) {
          console.error(`[AutoLoop] Error cancelling order ${order.id}:`, error.message);
        }
        
        // Small delay to avoid rate limits
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      
      console.log(`[AutoLoop] Cancelled ${cancelCount} of ${pendingOrders.length} pending orders`);
      
    } catch (error) {
      console.error('[AutoLoop] Error in cancelPendingOrders:', error);
    }
  }

  async cancelStaleOrders() {
    try {
      const ordersResp = await fetch('http://localhost:4000/api/paper/orders?limit=50');
      if (!ordersResp.ok) return;
      
      const ordersData = await ordersResp.json();
      const orders = ordersData.orders || ordersData.items || [];
      const pendingOrders = orders.filter(o => o.status === 'pending');
      
      if (pendingOrders.length === 0) return;
      
      // Cancel orders older than 1 hour
      const oneHourAgo = Date.now() - (60 * 60 * 1000);
      const staleOrders = pendingOrders.filter(o => {
        const orderTime = new Date(o.create_date || o.timestamp).getTime();
        return orderTime < oneHourAgo;
      });
      
      if (staleOrders.length > 0) {
        console.log(`[AutoLoop] Found ${staleOrders.length} stale orders to cancel`);
        for (const order of staleOrders) {
          try {
            await fetch(`http://localhost:4000/api/paper/orders/${order.id}`, { method: 'DELETE' });
            console.log(`[AutoLoop] Cancelled stale order ${order.id} for ${order.symbol}`);
          } catch (error) {
            console.error(`[AutoLoop] Failed to cancel stale order ${order.id}:`, error.message);
          }
          await new Promise(resolve => setTimeout(resolve, 100));
        }
      }
    } catch (error) {
      console.error('[AutoLoop] Error checking stale orders:', error.message);
    }
  }

  async runOnce() {
    // Check circuit breaker first
    if (circuitBreaker) {
      const canRun = circuitBreaker.canExecute('autoloop');
      if (!canRun.allowed) {
        this.status = `BLOCKED: CIRCUIT_BREAKER - ${canRun.reason}`;
        console.log(`[AutoLoop] Circuit breaker preventing execution: ${canRun.reason}`);
        return;
      }
    }
    
    const health = currentHealth();
    if (!health.ok) {
      this.status = `BLOCKED: HEALTH ${health.breaker}`;
      console.log(`[AutoLoop] Skipping run: health is ${health.breaker}`);
      return;
    }
    
    // Check and cancel stale orders periodically
    if (Math.random() < 0.1) { // 10% chance each cycle
      await this.cancelStaleOrders();
    }
    
    // Monitor positions for quick exits (especially diamonds)
    await this._monitorPositionsForExits();
    
    // Check stop losses using risk manager
    if (this.riskManager) {
      const stopsToExecute = await this.riskManager.checkStopLosses();
      if (stopsToExecute.length > 0) {
        console.log(`[AutoLoop] Risk Manager found ${stopsToExecute.length} stops to execute`);
        for (const stop of stopsToExecute) {
          const stopSignal = {
            symbol: stop.symbol,
            side: 'SELL',
            quantity: stop.qty,
            strategy: stop.reason,
            confidence: 0.95,
            expectedValue: 0, // Stop loss, no expected profit
            price: stop.currentPrice,
            source: 'risk_manager'
          };
          
          console.log(`[AutoLoop] Executing stop loss for ${stop.symbol}: ${(stop.lossPct * 100).toFixed(2)}% loss`);
          this.generatedSignals.push(stopSignal);
        }
      }
    }
    
    // Check for too many pending orders AND prevent duplicates
    this.pendingOrderSymbols = new Set();
    try {
      const ordersResp = await fetch('http://localhost:4000/api/paper/orders?status=pending');
      if (ordersResp.ok) {
        const pendingOrders = await ordersResp.json();
        if (Array.isArray(pendingOrders)) {
          // Track all pending order symbols
          pendingOrders.forEach(order => {
            if (order.side.toLowerCase() === 'buy') {
              this.pendingOrderSymbols.add(order.symbol);
            }
          });
          
          console.log(`[AutoLoop] ${pendingOrders.length} pending orders for symbols: ${Array.from(this.pendingOrderSymbols).join(', ')}`);
          
          // Cancel old orders or duplicates
          if (pendingOrders.length > 10) {
            console.warn(`[AutoLoop] ⚠️ Too many pending orders! Cancelling duplicates and old ones...`);
            const symbolCounts = new Map();
            
            // Sort by timestamp (newest first)
            pendingOrders.sort((a, b) => new Date(b.created_at || b.timestamp) - new Date(a.created_at || a.timestamp));
            
            for (const order of pendingOrders) {
              const count = symbolCounts.get(order.symbol) || 0;
              symbolCounts.set(order.symbol, count + 1);
              
              // Cancel if duplicate (keep only the newest) or if older than 5 minutes
              const orderTime = new Date(order.created_at || order.timestamp).getTime();
              const fiveMinutesAgo = Date.now() - (5 * 60 * 1000);
              
              if (count > 0 || orderTime < fiveMinutesAgo) {
                await fetch(`http://localhost:4000/api/paper/orders/${order.id}/cancel`, { method: 'POST' });
                console.log(`[AutoLoop] Cancelled ${count > 0 ? 'duplicate' : 'stale'} order: ${order.symbol} ${order.side}`);
              }
            }
          }
        }
      }
    } catch (e) {
      console.error('[AutoLoop] Error checking pending orders:', e.message);
    }

    try {
      // 1. Get strategy allocations
      this.status = 'ALLOCATING';
      const strategyStatsResp = await fetch('http://localhost:4000/api/strategies');
      const strategyData = await strategyStatsResp.json();
      const strategyStats = {};
      if (strategyData.items) {
        for (const item of strategyData.items) {
          strategyStats[item.id] = item.performance || {};
        }
      }

      const paperAccountResp = await fetch('http://localhost:4000/api/paper/account');
      const paperAccountData = await paperAccountResp.json();
      
      // Extract values from the balances structure
      const paperAccount = {
        cash: paperAccountData.balances?.total_cash || paperAccountData.balances?.cash || paperAccountData.cash || 0,
        equity: paperAccountData.balances?.total_equity || paperAccountData.balances?.equity || paperAccountData.equity || 100000,
        market_value: paperAccountData.balances?.market_value || paperAccountData.market_value || 0
      };

      // Reset daily stats if new day
      const today = new Date().toDateString();
      if (this.dailyStats.lastReset !== today) {
        this.dailyStats = {
          tradesExecuted: 0,
          capitalDeployed: 0,
          lastReset: today
        };
      }

      // Check capital constraints
      const availableCash = paperAccount.cash || 0;
      const totalEquity = paperAccount.equity || 100000;
      const positionsValue = paperAccount.market_value || (totalEquity - availableCash);

      console.log(`[AutoLoop] Capital check - Cash: $${availableCash.toFixed(2)}, Positions: $${positionsValue.toFixed(2)}, Daily trades: ${this.dailyStats.tradesExecuted}`);

      // Enforce capital limits
      if (positionsValue >= this.capitalLimits.maxTotalCapital) {
        this.status = 'CAPITAL_LIMIT_REACHED';
        console.log(`[AutoLoop] Capital limit reached: $${positionsValue.toFixed(2)} >= $${this.capitalLimits.maxTotalCapital}`);
        
        // Cancel any pending orders to prevent further capital deployment
        await this.cancelPendingOrders();
        return;
      }

      if (availableCash < this.capitalLimits.minCashBuffer) {
        this.status = 'INSUFFICIENT_CASH';
        console.log(`[AutoLoop] Insufficient cash buffer: $${availableCash.toFixed(2)} < $${this.capitalLimits.minCashBuffer}`);
        await this.cancelPendingOrders();
        return;
      }

      if (this.dailyStats.tradesExecuted >= this.capitalLimits.maxDailyTrades) {
        this.status = 'DAILY_TRADE_LIMIT';
        console.log(`[AutoLoop] Daily trade limit reached: ${this.dailyStats.tradesExecuted} >= ${this.capitalLimits.maxDailyTrades}`);
        await this.cancelPendingOrders();
        return;
      }

      const allocationContext = {
        equity: paperAccount.equity || 100000,
        volatilityRegime: 'neutral_medium', // Could be fetched from market data
        marketConditions: {},
        availableCash: Math.max(0, availableCash - this.capitalLimits.minCashBuffer),
        maxPerTrade: this.capitalLimits.maxPerTrade
      };

      // For testing, create equal-weight allocations for all strategies
      let allocationResult;
      if (Object.keys(strategyStats).length === 0 || Object.values(strategyStats).every(s => (s.trades || 0) === 0)) {
        // No performance data - allocate equally to all active strategies
        const activeStrategies = this.strategyManager ? Object.values(this.strategyManager.getAllStrategies()).filter(s => s.enabled !== false) : [];
        const weight = activeStrategies.length > 0 ? 1.0 / activeStrategies.length : 0;
        const allocations = {};
        
        activeStrategies.forEach(strategy => {
          allocations[strategy.id || strategy.name] = {
            weight: weight,
            amount: (allocationContext.availableCash || 20000) * weight,
            reason: 'equal_weight_testing'
          };
        });
        
        allocationResult = { 
          allocations,
          summary: { selectedCount: activeStrategies.length, totalWeight: 1.0 }
        };
        console.log(`[AutoLoop] Using equal-weight allocation for ${activeStrategies.length} strategies (testing mode)`);
      } else {
        // Normal IR-weighted allocation
        allocationResult = allocator.allocateCapital(strategyStats, allocationContext);
      }
      
      console.log(`[AutoLoop] Strategy allocations:`, Object.entries(allocationResult.allocations).map(([id, alloc]) => 
        `${id}: weight=${alloc.weight?.toFixed(3) || 0}, amount=$${alloc.amount?.toFixed(2) || 0}`
      ).join(', '));

      // 2. Generate signals from active strategies
      this.status = 'GENERATING_SIGNALS';
      const rawSignals = await this._generateStrategySignals(allocationResult.allocations);

      if (rawSignals.length === 0) {
        this.status = 'IDLE: NO_SIGNALS';
        console.log('[AutoLoop] No signals generated from strategies.');
        return;
      }

      // 3. Coordinate conflicting signals
      this.status = 'COORDINATING';
      const winningIntents = coordinator.pickWinningIntents(rawSignals, strategyStats);

      if (winningIntents.length === 0) {
        this.status = 'IDLE: NO_WINNERS';
        console.log('[AutoLoop] No winning intents after coordination.');
        return;
      }

      // 3.5. Publish proposals to decisionsBus (for frontend visibility)
      // this.status = 'PUBLISHING_PROPOSALS';
      // for (const intent of winningIntents) {
      //   const proposal = {
      //     trace_id: `proposal-${Date.now()}-${intent.symbol}`,
      //     timestamp: new Date().toISOString(),
      //     as_of: new Date().toISOString(),
      //     symbol: intent.symbol,
      //     action: intent.side,
      //     strategy: intent.key.split(':')[2] || 'unknown',
      //     confidence: intent.meta?.winner_score || 0.5,
      //     note: `Proposal: ${intent.symbol} ${intent.side} (${intent.strategy_id})`,
      //     score: intent.meta?.score || 0,
      //     execution: { status: 'PROPOSED' },
      //     meta: intent.meta
      //   };
      //   decisionsBus.publish(proposal);
      // }

      // 4. Apply enhanced risk gates
      this.status = 'VALIDATING_RISKS';
      const validatedIntents = [];

      for (const intent of winningIntents) {
        console.log(`[AutoLoop] Processing intent:`, JSON.stringify({symbol: intent.symbol, price: intent.price, side: intent.side, size_hint: intent.size_hint}));
        const riskContext = {
          equity: paperAccount.equity,
          cash: paperAccount.cash,
          quote_age_s: health.quote_age_s,
          broker_age_s: health.broker_age_s,
          broker_ok: health.ok,
          last_price: intent.price,
          open_positions_count: paperAccount.positions?.filter(p => p.qty > 0).length || 0,
          current_daily_var: 0.005, // Could be calculated
          strategy_stats: strategyStats,
          strategy_heat: 0.1, // Could be tracked per strategy
          correlation_buckets: {},
          volatility_regime: allocationContext.volatilityRegime
        };
        console.log(`[AutoLoop] Risk context price:`, riskContext.last_price);

        const riskResult = await riskGate.validateTrade(intent, riskContext);

        if (riskResult.decision === 'ACCEPT') {
          validatedIntents.push({
            ...intent,
            validated_qty: riskResult.routed_qty,
            risk_audit: riskResult.audit
          });
        } else {
          console.log(`[AutoLoop] Intent rejected for ${intent.symbol}: ${riskResult.reason}`);
        }
      }

      if (validatedIntents.length === 0) {
        this.status = 'BLOCKED: RISK_GATES';
        console.log('[AutoLoop] All intents rejected by risk gates.');
        return;
      }

      // 5. Execute validated trades
      this.status = `EXECUTING ${validatedIntents.length} TRADES`;

      for (const intent of validatedIntents) {
        try {
          // First, emit the decision for recording
          const decisionData = {
            symbol: intent.symbol,
            side: intent.side,
            qty: intent.validated_qty,
            price: intent.price,
            strategy_id: intent.strategy_id,
            confidence: intent.confidence || intent.score || 0.7,
            brain_score: intent.brain_score,
            stage: 'intent',
            reason: intent.meta?.reason || 'autoloop_signal',
            analysis: {
              volatility: intent.volatility || 'unknown',
              spread_bps: intent.spread_bps,
              costs_est: intent.costs_est,
              position_value: intent.position_value || (intent.price * intent.validated_qty),
              scores: intent.meta?.scoring_breakdown || {},
              risk_audit: intent.risk_audit
            },
            market_context: {
              regime: allocationContext.volatilityRegime || 'neutral',
              volatility: allocationContext.volatilityRegime || 'medium',
              vix: null,
              news_sentiment: null
            },
            timestamp: new Date().toISOString()
          };
          
          // Emit decision via fetch to decisions endpoint
          try {
            await fetch('http://localhost:4000/api/decisions/record', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(decisionData)
            });
          } catch (e) {
            console.log('[AutoLoop] Could not record decision:', e.message);
          }
          
          const idempotencyKey = `autoloop-${nanoid(12)}`;
          const orderPayload = {
            symbol: intent.symbol,
            side: intent.side,
            qty: intent.validated_qty,
            type: 'market',
            idempotency_key: idempotencyKey,
            strategy_id: intent.strategy_id,
            coordination_meta: intent.meta
          };

          const orderResp = await fetch('http://localhost:4000/api/orders', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(orderPayload)
          });

          if (orderResp.ok) {
            const orderResult = await orderResp.json();
            console.log(`[AutoLoop] Order placed: ${intent.symbol} ${intent.side} ${intent.validated_qty} (ID: ${orderResult.id})`);
            
            // Set stop loss for buy orders
            if ((intent.side === 'buy' || intent.side === 'BUY') && this.riskManager) {
              const entryPrice = intent.price || orderResult.price || orderResult.avg_fill_price;
              if (entryPrice) {
                this.riskManager.setStopLoss(intent.symbol, entryPrice, 'buy');
              }
            }
            
            // Emit trade_executed event for SystemIntegrator
            this.emit('trade_executed', {
              symbol: intent.symbol,
              side: intent.side,
              quantity: intent.validated_qty,
              price: intent.price,
              strategy: intent.strategy_id,
              order_id: orderResult.id,
              timestamp: new Date().toISOString(),
              metadata: intent.meta
            });
            
            // Record success in circuit breaker
            if (circuitBreaker) {
              circuitBreaker.recordSuccess('order_placement');
            }
            
            // Update daily stats
            this.dailyStats.tradesExecuted++;
          } else {
            const errorText = await orderResp.text();
            console.error(`[AutoLoop] Order failed for ${intent.symbol}: ${orderResp.status} - ${errorText}`);
            
            // Record failure in circuit breaker
            if (circuitBreaker) {
              circuitBreaker.recordFailure('ORDER_FAILURE', new Error(`Order failed: ${orderResp.status}`), {
                symbol: intent.symbol,
                side: intent.side,
                qty: intent.validated_qty,
                status: orderResp.status,
                error: errorText
              });
            }
          }
        } catch (error) {
          console.error(`[AutoLoop] Error placing order for ${intent.symbol}:`, error.message);
        }
      }

      // Update daily stats
      this.dailyStats.tradesExecuted += validatedIntents.length;
      
      this.status = 'COMPLETED';
      this.lastRun = new Date();
      console.log(`[AutoLoop] Cycle completed: ${validatedIntents.length} trades executed (Daily total: ${this.dailyStats.tradesExecuted}/${this.capitalLimits.maxDailyTrades})`);

    } catch (error) {
      this.status = `ERROR: ${error.message}`;
      console.error('[AutoLoop] Error in runOnce:', error);
      
      // Record failure in circuit breaker
      if (circuitBreaker) {
        circuitBreaker.recordFailure('AUTOLOOP_ERROR', error, {
          status: this.status,
          phase: 'runOnce'
        });
      }
      
      // If this is a critical error, throw it to stop the loop
      if (error.message.includes('CRITICAL') || error.message.includes('FATAL')) {
        throw error;
      }
    }
  }

  /**
   * Generate signals from allocated strategies - PARALLEL VERSION
   */
  async _generateStrategySignals(allocations) {
    const signalsBySymbol = new Map(); // Track best signal per symbol
    
    // Get current positions to avoid buying what we already own
    const positionsResp = await fetch('http://localhost:4000/api/paper/positions');
    const positions = positionsResp.ok ? await positionsResp.json() : [];
    const ownedSymbols = new Set(positions.filter(p => p.qty > 0).map(p => p.symbol));
    
    // Get all candidates in parallel
    console.log('[AutoLoop] Fetching all candidates in parallel...');
    const [highValueCandidates, regularCandidates] = await Promise.all([
      this._getHighValueCandidates(),
      this._getRegularCandidates(allocations)
    ]);
    
    console.log(`[AutoLoop] Found ${highValueCandidates.length} high-value and ${regularCandidates.length} regular candidates`);
    
    // Combine all candidates
    const allCandidates = [...highValueCandidates, ...regularCandidates];
    
    // Filter out owned symbols and pending orders
    const eligibleCandidates = allCandidates.filter(candidate => {
      if (ownedSymbols.has(candidate.symbol)) {
        console.log(`[AutoLoop] Skipping ${candidate.symbol} - already have position`);
        return false;
      }
      if (this.pendingOrderSymbols && this.pendingOrderSymbols.has(candidate.symbol)) {
        console.log(`[AutoLoop] Skipping ${candidate.symbol} - already have pending order`);
        return false;
      }
      return true;
    });
    
    console.log(`[AutoLoop] Testing ${eligibleCandidates.length} eligible candidates against ${Object.keys(allocations).length} strategies in parallel...`);
    
    // Get active strategy IDs with positive allocations
    const activeStrategies = Object.entries(allocations)
      .filter(([_, allocation]) => allocation.weight > 0)
      .map(([strategyId, allocation]) => ({ strategyId, allocation }));
    
    // Create all test combinations (candidate × strategy)
    const testPromises = [];
    for (const candidate of eligibleCandidates) {
      for (const { strategyId, allocation } of activeStrategies) {
        testPromises.push(
          this._testCandidateWithStrategy(candidate, strategyId, allocation)
            .catch(e => {
              console.error(`[AutoLoop] Error testing ${candidate.symbol} with ${strategyId}:`, e.message);
              return null; // Return null on error to continue processing other candidates
            })
        );
      }
    }
    
    // Run all tests in parallel with concurrency limit to avoid overwhelming the system
    const BATCH_SIZE = 20; // Process 20 tests at a time
    const allSignals = [];
    
    for (let i = 0; i < testPromises.length; i += BATCH_SIZE) {
      const batch = testPromises.slice(i, i + BATCH_SIZE);
      const batchResults = await Promise.all(batch);
      
      // Filter out null results and add valid signals
      const validSignals = batchResults.filter(signal => signal !== null);
      allSignals.push(...validSignals);
      
      // Log progress
      if (i > 0 && i % 100 === 0) {
        console.log(`[AutoLoop] Progress: tested ${i}/${testPromises.length} combinations, found ${allSignals.length} signals so far`);
      }
      
      // Early exit if we have enough high-confidence signals
      if (allSignals.length >= 10) {
        console.log(`[AutoLoop] Found sufficient signals (${allSignals.length}), stopping early`);
        break;
      }
    }
    
    // Keep only the best signal per symbol
    for (const signal of allSignals) {
      if (signal) {
        const existing = signalsBySymbol.get(signal.symbol);
        if (!existing || signal.confidence > existing.confidence) {
          signalsBySymbol.set(signal.symbol, signal);
        }
      }
    }
    
    // Convert to array and sort by confidence
    const uniqueSignals = Array.from(signalsBySymbol.values())
      .sort((a, b) => b.confidence - a.confidence);
    
    console.log(`[AutoLoop] Parallel evaluation complete: ${uniqueSignals.length} unique signals from ${testPromises.length} tests`);
    
    return uniqueSignals;
  }
  
  /**
   * Get regular trading candidates from scanner and dynamic discovery
   */
  async _getRegularCandidates(allocations) {
    const candidates = new Map(); // Use Map to avoid duplicates
    
    try {
      // Don't limit by position count - let capital and risk management decide
      const openPositions = await this.getOpenPositionCount();
      console.log(`[AutoLoop] Current open positions: ${openPositions}`);
      
      const limit = 20; // Get plenty of candidates for testing
      
      // 1. Dynamic discovery - Re-enabled with timeout protection
      if (this.dynamicDiscovery) {
        try {
          const controller = new AbortController();
          const timeout = setTimeout(() => controller.abort(), 5000); // 5 second timeout
          
          const dynamicResp = await fetch(`http://localhost:4000/api/discovery/dynamic-symbols?limit=${limit}`, {
            signal: controller.signal
          });
          clearTimeout(timeout);
          
          if (dynamicResp.ok) {
            const dynamicData = await dynamicResp.json();
            // Add unique candidates to map
            (dynamicData.symbols || []).forEach(item => {
              if (!candidates.has(item.symbol)) {
                candidates.set(item.symbol, {
                  symbol: item.symbol,
                  confidence: item.score || 0.5,
                  source: item.source || 'dynamic',
                  last: item.last || item.price,
                  bid: item.bid,
                  ask: item.ask,
                  change_pct: item.change_pct || 0,
                  volume: item.volume || 0,
                  spread_bps: item.spread_bps || 50
                });
              }
            });
            console.log(`[AutoLoop] Dynamic discovery added ${dynamicData.symbols?.length || 0} candidates`);
          }
        } catch (e) {
          if (e.name === 'AbortError') {
            console.error(`[AutoLoop] Dynamic discovery timeout after 5 seconds`);
          } else {
            console.error(`[AutoLoop] Dynamic discovery failed:`, e.message);
          }
        }
      }
      
      // 2. Penny movers scanner (often has good opportunities)
      try {
        const pennyResp = await fetch(`http://localhost:4000/api/scanner/candidates?list=penny_movers&limit=${limit}`);
        if (pennyResp.ok) {
          const pennyData = await pennyResp.json();
          pennyData.forEach(item => {
            if (!candidates.has(item.symbol)) {
              // Add confidence score if not present
              item.confidence = item.confidence || item.score || 0.5;
              candidates.set(item.symbol, item);
            }
          });
        }
      } catch (e) {
        console.error(`[AutoLoop] Penny movers scanner failed:`, e.message);
      }
      
      // 3. Regular scanner for additional candidates
      try {
        const scannerResp = await fetch(`http://localhost:4000/api/scanner/candidates?list=small_caps_liquid&limit=${limit}`);
        if (scannerResp.ok) {
          const scannerData = await scannerResp.json();
          scannerData.forEach(item => {
            if (!candidates.has(item.symbol)) {
              candidates.set(item.symbol, item);
            }
          });
        }
      } catch (e) {
        console.error(`[AutoLoop] Scanner failed:`, e.message);
      }
      
      // 4. Include trending/volume movers
      try {
        const moversResp = await fetch(`http://localhost:4000/api/scanner/candidates?list=volume_movers&limit=${limit}`);
        if (moversResp.ok) {
          const movers = await moversResp.json();
          movers.forEach(item => {
            if (!candidates.has(item.symbol) && item.volume > 10000) { // Lower volume threshold for testing
              candidates.set(item.symbol, item);
            }
          });
        }
      } catch (e) {
        console.error(`[AutoLoop] Volume movers scan failed:`, e.message);
      }
      
      // Convert map to array and filter
      const candidateArray = Array.from(candidates.values())
        .filter(candidate => {
          // Skip failed symbols
          const failInfo = this.failedSymbols.get(candidate.symbol);
          if (failInfo && failInfo.failCount >= this.MAX_FAIL_COUNT) {
            const timeSinceLastCheck = Date.now() - failInfo.lastCheck;
            if (timeSinceLastCheck < this.FAIL_COOLDOWN_MS) {
              return false;
            }
          }
          
          // Apply poor capital mode filters
          if (POOR_CAPITAL_MODE.enabled) {
            const price = candidate.last || candidate.price || 0;
            const spread_bps = candidate.spread_bps || 0;
            
            if (price < POOR_CAPITAL_MODE.universe.minPrice || 
                price > POOR_CAPITAL_MODE.universe.maxPrice ||
                spread_bps > POOR_CAPITAL_MODE.universe.maxSpreadBps) {
              return false;
            }
          }
          
          return true;
        })
        .sort((a, b) => {
          // Prioritize by score/confidence
          return (b.confidence || b.score || 0.5) - (a.confidence || a.score || 0.5);
        });
      
      console.log(`[AutoLoop] Found ${candidateArray.length} regular candidates after filtering`);
      return candidateArray;
      
    } catch (error) {
      console.error(`[AutoLoop] Error getting regular candidates:`, error.message);
      return [];
    }
  }
  
  /**
   * Determine trade side (BUY/SELL) for a candidate
   */
  async _determineSide(candidate, strategyId) {
    // Check if we already have a position
    const hasPosition = await this.hasPosition(candidate.symbol);
    
    // Get performance boost from winning strategies
    let performanceBoost = 0;
    
    // Check strategy performance for boost
    if (this.performanceRecorder) {
      try {
        const strategyMetrics = await this.performanceRecorder.getStrategyPerformance(strategyId, 24);
        if (strategyMetrics && strategyMetrics.winRate > 0.6) {
          performanceBoost = 0.05; // 5% boost for proven winners
          console.log(`[AutoLoop] Strategy ${strategyId} has ${(strategyMetrics.winRate * 100).toFixed(1)}% win rate, adding performance boost`);
        }
      } catch (e) {
        // Ignore performance lookup errors
      }
    }
    
    // Use brain integrator for unified decision making
    if (!this.brainIntegrator) {
      console.log(`[AutoLoop] No brain integrator available for ${candidate.symbol}`);
      return null;
    }
    const decision = await this.brainIntegrator.scoreCandidate(candidate.symbol, {
      hasPosition: hasPosition,
      strategyId: strategyId,
      candidateData: candidate,
      newsNudge: candidate.newsNudge || 0,
      performanceBoost: performanceBoost,
      enhancedRecorder: this.enhancedRecorder
    });
    
    if (!decision) {
      console.log(`[AutoLoop] No signal from brain scorer for ${candidate.symbol}`);
      return null;
    }
    
    console.log(`[AutoLoop] Brain decision for ${candidate.symbol}: ${decision.side} (score: ${decision.score}, conf: ${decision.confidence})`);
    
    // Return the side from brain scoring
    return decision.side;
  }
  
  /**
   * Check if we have an open position for a symbol
   */
  async hasPosition(symbol) {
    try {
      const positionsResp = await fetch('http://localhost:4000/api/paper/positions');
      if (!positionsResp.ok) return false;
      
      const positions = await positionsResp.json();
      return positions.some(p => p.symbol === symbol && p.qty > 0);
    } catch (error) {
      console.error(`[AutoLoop] Error checking position for ${symbol}:`, error.message);
      return false;
    }
  }

  /**
   * Get coordination audit trail
   */
  getCoordinationAudit() {
    return coordinator.getLastCycleAudit();
  }

  /**
   * Get risk gate rejection statistics
   */
  getRiskRejections(limit = 10) {
    return riskGate.getRecentRejections(limit);
  }

  /**
   * Get allocation summary
   */
  getAllocationSummary() {
    return allocator.getCurrentAllocation();
  }

  /**
   * Get high-value candidates from diamonds, news, and significant movers
   */
  async _getHighValueCandidates() {
    const candidates = [];
    
    try {
      // 1. Get diamonds (high-impact news on penny stocks)
      const diamondsResp = await fetch('http://localhost:4000/api/lab/diamonds?limit=10');
      if (diamondsResp.ok) {
        const diamondsData = await diamondsResp.json();
        const diamonds = diamondsData.items || [];
        if (diamonds.length > 0) {
          console.log(`[AutoLoop] 💎 Found ${diamonds.length} diamonds`);
          for (const diamond of diamonds) {
            // Fetch quotes for diamonds
            const quoteResp = await fetch(`http://localhost:4000/api/quotes?symbols=${diamond.symbol}`);
            if (quoteResp.ok) {
              const quotes = await quoteResp.json();
              // Quotes API returns an array, not an object
              const quote = Array.isArray(quotes) ? quotes[0] : quotes[diamond.symbol];
              if (quote && quote.last) {
                candidates.push({
                  symbol: diamond.symbol,
                  source: 'diamonds',
                  confidence: diamond.impactScore || 0.8,
                is_diamond: true,
                news_impact: diamond.score || diamond.impactScore,
                last: quote.last,
                bid: quote.bid || quote.last * 0.995,
                ask: quote.ask || quote.last * 1.005,
                volume: quote.volume || 0,
                change_pct: quote.change_pct || 0,
                spread_bps: ((quote.ask - quote.bid) / quote.last * 10000) || 10,
                metadata: diamond.features || diamond.evidence || {}
                });
              }
            }
          }
        }
      }
    } catch (e) {
      console.error('[AutoLoop] Error fetching diamonds:', e.message);
    }
    
    // 2. Get significant movers (>5% change)
    try {
      const moversResp = await fetch('http://localhost:4000/api/scanner/candidates?list=volume_movers&limit=20');
      if (moversResp.ok) {
        const movers = await moversResp.json();
        // Apply poor capital mode constraints
        const bigMovers = movers.filter(m => {
          const meetsBasic = Math.abs(m.change_pct || 0) > 5 && m.volume > 2000000;
          if (!meetsBasic) return false;
          
          // Apply POOR_CAPITAL_MODE constraints if enabled
          if (POOR_CAPITAL_MODE.enabled) {
            const price = m.last || m.price || 0;
            const spread_bps = m.spread_bps || 0;
            
            return price >= POOR_CAPITAL_MODE.universe.minPrice &&
                   price <= POOR_CAPITAL_MODE.universe.maxPrice &&
                   spread_bps <= POOR_CAPITAL_MODE.universe.maxSpreadBps;
          }
          return true;
        });
        
        for (const mover of bigMovers.slice(0, 5)) {
          if (!candidates.find(c => c.symbol === mover.symbol)) {
            candidates.push({
              symbol: mover.symbol,
              source: 'big_mover',
              confidence: Math.min(0.8, Math.abs(mover.change_pct) / 10),
              is_diamond: Math.abs(mover.change_pct) > 10 && mover.last < 10, // Very big moves on cheap stocks are diamonds
              last: mover.last || mover.price,
              bid: mover.bid || mover.last * 0.995,
              ask: mover.ask || mover.last * 1.005,
              volume: mover.volume,
              change_pct: mover.change_pct,
              spread_bps: mover.spread_bps || 10
            });
          }
        }
      }
    } catch (e) {
      console.error('[AutoLoop] Error fetching movers:', e.message);
    }
    
    return candidates;
  }
  
  /**
   * Test a specific candidate with a specific strategy
   */
  async _testCandidateWithStrategy(candidate, strategyId, allocation) {
    console.log(`[AutoLoop] Testing ${candidate.symbol} (${candidate.source}) with strategy ${strategyId}`);
    
    // Skip if we've failed this symbol too many times
    const failInfo = this.failedSymbols.get(candidate.symbol);
    if (failInfo && failInfo.failCount >= this.MAX_FAIL_COUNT) {
      const timeSinceFail = Date.now() - failInfo.lastCheck;
      if (timeSinceFail < this.FAIL_COOLDOWN_MS) {
        return null;
      }
    }
    
    // Get account info for position sizing
    let accountEquity = 100000;
    try {
      const accountResp = await fetch('http://localhost:4000/api/paper/account');
      if (accountResp.ok) {
        const accountData = await accountResp.json();
        accountEquity = accountData.equity || accountData.balances?.total_equity || 100000;
      }
    } catch (e) {
      console.log('[AutoLoop] Using default equity for position sizing');
    }
    
    // For diamonds, boost confidence score instead of lowering threshold
    // This lets the brain and tournament system handle them naturally
    if (candidate.is_diamond) {
      candidate.confidence = Math.min(0.9, (candidate.confidence || 0.5) + 0.2);
      candidate.adjustedScore = candidate.confidence;
      candidate.newsNudge = (candidate.newsNudge || 0) + 0.1; // Add news boost for diamonds
      console.log(`[AutoLoop] 💎 Diamond boost applied to ${candidate.symbol}: confidence ${candidate.confidence}, newsNudge ${candidate.newsNudge}`);
    }
    
    const side = await this._determineSide(candidate, strategyId);
    
    if (!side) {
      return null;
    }
    
    // Get portfolio P&L to adjust position sizing
    let portfolioAdjustment = 1.0;
    if (this.performanceRecorder && this.performanceRecorder.getCurrentPortfolioStatus) {
      try {
        const portfolioStatus = await this.performanceRecorder.getCurrentPortfolioStatus();
        // Scale up when winning (max 1.5x), scale down when losing (min 0.5x)
        if (portfolioStatus.pnlPercent > 2) {
          portfolioAdjustment = Math.min(1.5, 1 + (portfolioStatus.pnlPercent / 10));
          console.log(`[AutoLoop] Winning! Scaling up position size by ${((portfolioAdjustment - 1) * 100).toFixed(0)}%`);
        } else if (portfolioStatus.pnlPercent < -2) {
          portfolioAdjustment = Math.max(0.5, 1 + (portfolioStatus.pnlPercent / 20));
          console.log(`[AutoLoop] Losing. Scaling down position size by ${((1 - portfolioAdjustment) * 100).toFixed(0)}%`);
        }
      } catch (e) {
        // Ignore and use default
      }
    }
    
    // Calculate position size with more reasonable risk levels
    let riskPerTrade = candidate.is_diamond ? 0.02 : 0.01; // 2% for high-confidence diamonds, 1% for others
    let maxPositionPct = candidate.is_diamond ? 0.05 : 0.10; // 5% for diamonds, 10% max for regular trades
    
    // Apply portfolio adjustment
    riskPerTrade *= portfolioAdjustment;
    
    // Apply poor capital mode constraints if enabled
    if (POOR_CAPITAL_MODE.enabled && accountEquity < 25000) {
      riskPerTrade = Math.min(riskPerTrade, POOR_CAPITAL_MODE.risk.maxRiskPerTrade);
      maxPositionPct = Math.min(maxPositionPct, POOR_CAPITAL_MODE.risk.maxPositionSize);
      console.log(`[AutoLoop] Poor capital mode active: risk ${(riskPerTrade*100).toFixed(1)}%, max position ${(maxPositionPct*100).toFixed(1)}%`);
    }
    
    const basePositionValue = accountEquity * riskPerTrade;
    const maxPositionValue = accountEquity * maxPositionPct;
    // Use allocation amount if available, otherwise use base position value
    const allocationAmount = allocation.amount || basePositionValue;
    // Use risk manager for optimal position sizing if available
    let shares;
    let targetPositionValue;
    const price = candidate.last || candidate.price || 10;
    
    if (this.riskManager) {
      // Use Kelly Criterion-based sizing
      const sizing = await this.riskManager.calculateOptimalSize(
        candidate.symbol, 
        price, 
        candidate.confidence || 0.6
      );
      
      if (sizing.shares > 0) {
        shares = sizing.shares;
        targetPositionValue = sizing.positionValue;
        console.log(`[AutoLoop] Risk-adjusted sizing for ${candidate.symbol}: ${shares} shares (${sizing.percentOfEquity.toFixed(1)}% of equity, Kelly: ${sizing.kellyPercent.toFixed(1)}%)`);
      } else {
        console.log(`[AutoLoop] Risk manager rejected ${candidate.symbol}: ${sizing.reason}`);
        return null;
      }
    } else {
      // Fallback to original sizing
      targetPositionValue = Math.min(basePositionValue, maxPositionValue, allocationAmount);
      shares = Math.max(1, Math.floor(targetPositionValue / price));
      console.log(`[AutoLoop] Fixed sizing for ${candidate.symbol}: $${targetPositionValue.toFixed(2)} / $${price} = ${shares} shares`);
    }
    
    // Calculate expected value
    const expectedMovePct = candidate.is_diamond ? 0.15 : 0.10; // 15% for diamonds, 10% for others
    const evCalc = this.evCalculator.calculateExpectedValue({
      symbol: candidate.symbol,
      side: side,
      quantity: shares,
      price: candidate.last,
      expectedMove: expectedMovePct,
      confidence: candidate.confidence || 0.6,
      spreadBps: candidate.spread_bps || 10,
      volatility: candidate.is_diamond ? 'high' : 'normal',
      marketCap: candidate.last < 5 ? 'small' : 'large'
    });
    
    // For diamonds with high confidence, always proceed
    // For testing, allow trades with slightly negative EV
    const evThreshold = -20; // Allow negative EV up to -$20 while bot learns
    if (candidate.is_diamond || evCalc.expectedValue > evThreshold) {
      console.log(`[AutoLoop] ✅ Signal generated for ${candidate.symbol}: ${side} ${shares} shares (EV: ${evCalc.expectedValue})`);
      return {
        symbol: candidate.symbol,
        side: side,
        quantity: shares,
        strategy: strategyId,
        confidence: candidate.confidence,
        expectedValue: Math.max(0.01, evCalc.expectedValue), // Ensure positive for diamonds
        afterCostEV: evCalc.afterCost,
        price: candidate.last,
        source: candidate.source,
        metadata: {
          ...candidate.metadata,
          is_diamond: candidate.is_diamond,
          news_impact: candidate.news_impact,
          change_pct: candidate.change_pct,
          exit_strategy: candidate.is_diamond ? 'quick' : 'normal'
        }
      };
    }
    
    console.log(`[AutoLoop] ❌ No signal for ${candidate.symbol}: EV ${evCalc.expectedValue} <= ${evThreshold}`);
    return null;
  }

  /**
   * Monitor positions for quick exits, especially diamonds - PARALLEL VERSION
   */
  async _monitorPositionsForExits() {
    try {
      // Get open positions
      const positionsResp = await fetch('http://localhost:4000/api/paper/positions');
      if (!positionsResp.ok) return;
      
      const positions = await positionsResp.json();
      if (!positions || positions.length === 0) return;
      
      // Initialize peak PnL tracking if not already set
      if (!this.peakPnLTracker) {
        this.peakPnLTracker = new Map();
      }
      
      // Get current quotes for positions
      const symbols = positions.map(p => p.symbol).join(',');
      const quotesResp = await fetch(`http://localhost:4000/api/quotes?symbols=${symbols}`);
      if (!quotesResp.ok) return;
      
      const quotes = await quotesResp.json();
      
      // Process all positions in parallel
      const exitPromises = positions.map(async (position) => {
        const quote = quotes[position.symbol];
        if (!quote) return null;
        
        const currentPrice = quote.last;
        const entryPrice = position.avg_price || position.price;
        const pnlPct = ((currentPrice - entryPrice) / entryPrice) * 100;
        
        // Check if this is a diamond position (check trades history)
        let isDiamond = false;
        try {
          const tradesResp = await fetch(`http://localhost:4000/api/trades?symbol=${position.symbol}&limit=1`);
          if (tradesResp.ok) {
            const trades = await tradesResp.json();
            const lastTrade = trades.items?.[0];
            isDiamond = lastTrade?.metadata?.is_diamond || lastTrade?.metadata?.exit_strategy === 'quick';
          }
        } catch (e) {
          // Ignore
        }
        
        // Track position holding time
        let holdingTimeHours = 0;
        try {
          const tradesResp = await fetch(`http://localhost:4000/api/trades?symbol=${position.symbol}&limit=1`);
          if (tradesResp.ok) {
            const trades = await tradesResp.json();
            const lastTrade = trades.items?.[0];
            if (lastTrade && lastTrade.timestamp) {
              holdingTimeHours = (Date.now() - new Date(lastTrade.timestamp).getTime()) / (1000 * 60 * 60);
            }
          }
        } catch (e) {
          // Ignore
        }
        
        // Smart exit logic for diamonds
        if (isDiamond) {
          let shouldExit = false;
          let exitReason = '';
          
          // Dynamic profit targets based on holding time
          if (pnlPct > 0) {
            // If we're up big quickly, take profits
            if (pnlPct > 15) {
              shouldExit = true;
              exitReason = 'strong_profit_15pct';
            }
            // Trailing stop: if we hit 10% and drop back to 7%, exit
            else if (pnlPct > 7 && position.peak_pnl && position.peak_pnl > 10 && pnlPct < position.peak_pnl - 3) {
              shouldExit = true;
              exitReason = 'trailing_stop_triggered';
            }
            // Standard 10% target
            else if (pnlPct > 10) {
              shouldExit = true;
              exitReason = 'profit_target_10pct';
            }
            // Time-based exit: lower targets for longer holds
            else if (holdingTimeHours > 2 && pnlPct > 5) {
              shouldExit = true;
              exitReason = 'time_based_exit_5pct';
            }
          }
          // Stop loss
          else if (pnlPct < -5) {
            shouldExit = true;
            exitReason = 'stop_loss_5pct';
          }
          
          // Track peak PnL for trailing stops
          const currentPeak = this.peakPnLTracker.get(position.symbol) || 0;
          if (pnlPct > currentPeak) {
            this.peakPnLTracker.set(position.symbol, pnlPct);
          }
          position.peak_pnl = this.peakPnLTracker.get(position.symbol);
          
          if (shouldExit) {
            console.log(`[AutoLoop] 💎 Diamond exit: ${position.symbol} @ ${pnlPct.toFixed(2)}% (${exitReason})`);
            
            const exitSignal = {
              symbol: position.symbol,
              side: 'SELL',
              quantity: position.qty,
              strategy: 'diamond_exit',
              confidence: 0.9,
              expectedValue: 1,
              price: currentPrice,
              source: 'smart_exit',
              metadata: {
                exit_reason: exitReason,
                pnl_pct: pnlPct,
                holding_hours: holdingTimeHours.toFixed(1),
                peak_pnl: position.peak_pnl,
                is_diamond: true
              }
            };
            
            await this._executeTrade(exitSignal);
          } else if (pnlPct > 5) {
            console.log(`[AutoLoop] 💎 Diamond holding: ${position.symbol} @ ${pnlPct.toFixed(2)}% (peak: ${(position.peak_pnl || pnlPct).toFixed(2)}%)`);
          }
        } else {
          // Normal positions: More sophisticated exit logic
          let shouldExit = false;
          let exitReason = '';
          
          if (pnlPct > 0) {
            // Take profits on big moves
            if (pnlPct > 30) {
              shouldExit = true;
              exitReason = 'strong_profit_30pct';
            }
            // Trailing stop after 20%
            else if (pnlPct > 15 && position.peak_pnl && position.peak_pnl > 20 && pnlPct < position.peak_pnl - 5) {
              shouldExit = true;
              exitReason = 'trailing_stop_from_20pct';
            }
            // Standard target
            else if (pnlPct > 20) {
              shouldExit = true;
              exitReason = 'profit_target_20pct';
            }
          }
          // Stop loss
          else if (pnlPct < -10) {
            shouldExit = true;
            exitReason = 'stop_loss_10pct';
          }
          
          // Track peak PnL
          if (!position.peak_pnl || pnlPct > position.peak_pnl) {
            position.peak_pnl = pnlPct;
          }
          
          if (shouldExit) {
            console.log(`[AutoLoop] Exit: ${position.symbol} @ ${pnlPct.toFixed(2)}% (${exitReason})`);
            
            const exitSignal = {
              symbol: position.symbol,
              side: 'SELL',
              quantity: position.qty,
              strategy: 'position_exit',
              confidence: 0.8,
              expectedValue: 1,
              price: currentPrice,
              source: 'smart_exit',
              metadata: {
                exit_reason: exitReason,
                pnl_pct: pnlPct,
                holding_hours: holdingTimeHours.toFixed(1),
                peak_pnl: position.peak_pnl
              }
            };
            
            await this._executeTrade(exitSignal);
          } else if (pnlPct > 10) {
            console.log(`[AutoLoop] Holding: ${position.symbol} @ ${pnlPct.toFixed(2)}% (peak: ${(position.peak_pnl || pnlPct).toFixed(2)}%)`);
          }
        }
      });
      
      // Execute all exit trades in parallel
      const exitSignals = await Promise.all(exitPromises);
      const validExitSignals = exitSignals.filter(signal => signal !== null);
      
      if (validExitSignals.length > 0) {
        console.log(`[AutoLoop] Executing ${validExitSignals.length} exit trades in parallel`);
        await Promise.all(validExitSignals.map(signal => this._executeTrade(signal)));
      }
    } catch (error) {
      console.error('[AutoLoop] Error monitoring positions:', error.message);
    }
  }
  
  /**
   * Execute a trade signal
   */
  async _executeTrade(signal) {
    try {
      const orderData = {
        symbol: signal.symbol,
        side: signal.side.toLowerCase(),
        qty: signal.quantity,
        type: 'market',
        strategy: signal.strategy,
        metadata: signal.metadata
      };
      
      const orderResp = await fetch('http://localhost:4000/api/paper/orders', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(orderData)
      });
      
      if (orderResp.ok) {
        const order = await orderResp.json();
        console.log(`[AutoLoop] Order placed: ${order.id} - ${signal.side} ${signal.quantity} ${signal.symbol}`);
        
        // Set stop loss for buy orders
        if ((signal.side === 'buy' || signal.side === 'BUY') && this.riskManager) {
          const entryPrice = signal.price || order.price || order.avg_fill_price;
          if (entryPrice) {
            this.riskManager.setStopLoss(signal.symbol, entryPrice, 'buy');
          }
        }
        
        // Record trade result for sell orders
        if ((signal.side === 'sell' || signal.side === 'SELL') && this.riskManager) {
          // Get position P&L from paper broker
          const position = this.paperBroker?.getPosition?.(signal.symbol);
          if (position && position.unrealized_pl !== undefined) {
            this.riskManager.recordTradeResult(signal.symbol, position.unrealized_pl);
          }
        }
        
        // Emit trade_executed event for SystemIntegrator
        this.emit('trade_executed', {
          symbol: signal.symbol,
          side: signal.side,
          quantity: signal.quantity,
          price: signal.price,
          strategy: signal.strategy,
          order_id: order.id,
          timestamp: new Date().toISOString(),
          metadata: signal.metadata
        });
        
        return order;
      }
    } catch (error) {
      console.error(`[AutoLoop] Error executing trade for ${signal.symbol}:`, error.message);
    }
  }

  /**
   * Trigger bot competition based on significant news sentiment
   */
  async triggerNewsBasedCompetition(symbol, nudgeValue) {
    try {
      const competitionResp = await fetch('http://localhost:4000/api/bot-competition/news-triggered', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          newsEvents: [{
            type: 'significant_sentiment',
            symbol: symbol,
            tickers: [symbol],
            magnitude: nudgeValue,
            source: 'news_nudge',
            timestamp: new Date().toISOString()
          }],
          marketContext: {
            regime: this.marketIndicators?.getMarketContext?.()?.regime || 'neutral',
            volatility: this.marketIndicators?.getMarketContext?.()?.volatility || 'medium'
          },
          nudge: nudgeValue
        })
      });

      const result = await competitionResp.json();
      
      if (result.success) {
        console.log(`[AutoLoop] ✅ Bot competition triggered successfully!`);
        console.log(`[AutoLoop] Competition ID: ${result.competition?.id}`);
        console.log(`[AutoLoop] ${result.message}`);
      } else {
        console.log(`[AutoLoop] ⚠️ Bot competition not triggered: ${result.reason}`);
      }
      
      return result;
    } catch (error) {
      console.error(`[AutoLoop] Error triggering bot competition:`, error);
      throw error;
    }
  }
}

module.exports = { AutoLoop };
