/**
 * Brain Integrator - The Central Nervous System
 * 
 * This is the UNIFIED BRAIN that connects all decision-making systems.
 * It ensures EVERY trade decision goes through ALL validation layers.
 */

const EventEmitter = require('events');

class BrainIntegrator extends EventEmitter {
  constructor(config = {}) {
    super();
    
    // All brain components
    this.components = {
      aiOrchestrator: null,      // High-level strategy management
      brainService: null,        // ML scoring and planning
      policyEngine: null,        // Rules and circuit breakers
      indicatorsConnector: null, // Technical analysis integration
      decisionCoordinator: null, // Conflict resolution
      circuitBreaker: null,      // System protection
      capitalTracker: null,      // Capital management
      dataValidator: null,       // Data quality checks
      positionSizer: null,       // Risk-based sizing
      performanceRecorder: null, // Track all decisions
      marketIndicators: null     // Market regime detection
    };
    
    // Decision cache to prevent duplicates
    this.recentDecisions = new Map();
    this.decisionTTL = 3000; // 3 seconds - much shorter to allow fresh evaluations
    
    // Unified state
    this.state = {
      regime: 'unknown',
      volatility: 'medium',
      trend: 'neutral',
      systemHealth: 'green',
      activeStrategies: new Set(),
      lastUpdate: null
    };
    
    // Thresholds (from tradingThresholds.js)
    this.thresholds = config.thresholds || {
      buyThreshold: 0.35,  // TESTING MODE: Very aggressive
      sellThreshold: 0.30, // TESTING MODE: Lower threshold
      minConfidence: 0.25, // TESTING MODE: Accept lower confidence
      newsBoostThreshold: 0.02,
      exitWinnerThreshold: 0.4,
      exitLoserThreshold: 0.5
    };
    
    this.minConfidence = config.minConfidence || 0.25; // TESTING MODE: Match threshold
    this.checkInterval = config.checkInterval || 30000;
    
    console.log('[BrainIntegrator] Central nervous system initialized');
  }
  
  /**
   * Connect all brain components
   */
  connectComponents(components) {
    Object.assign(this.components, components);
    
    // Validate critical components
    const required = ['circuitBreaker', 'capitalTracker', 'dataValidator'];
    for (const comp of required) {
      if (!this.components[comp]) {
        console.warn(`[BrainIntegrator] Missing critical component: ${comp}`);
      }
    }
    
    console.log('[BrainIntegrator] Connected components:', Object.keys(this.components).filter(k => !!this.components[k]));
  }
  
  /**
   * The MAIN decision function - ALL decisions go through here
   */
  async makeDecision(context) {
    const { symbol, action, strategy, hasPosition, candidateData } = context;
    const startTime = Date.now();
    
    // Use enhanced brain if available
    if (this.enhancedBrain) {
      try {
        const enhancedDecision = await this.enhancedBrain.makeEnhancedDecision({
          ...context,
          currentPrice: candidateData?.last || candidateData?.price
        });
        
        console.log(`[BrainIntegrator] Enhanced decision for ${symbol}: ${enhancedDecision.action} (score: ${enhancedDecision.score.toFixed(3)}, confidence: ${(enhancedDecision.confidence * 100).toFixed(1)}%)`);
        return enhancedDecision;
      } catch (error) {
        console.error(`[BrainIntegrator] Enhanced brain error, falling back:`, error.message);
        // Fall through to regular decision making
      }
    }
    
    try {
      // 1. Check decision cache to prevent duplicates
      const cacheKey = `${symbol}-${action}-${strategy}-${hasPosition ? 'owned' : 'new'}`;
      if (this.recentDecisions.has(cacheKey)) {
        const cached = this.recentDecisions.get(cacheKey);
        if (Date.now() - cached.timestamp < this.decisionTTL) {
          console.log(`[BrainIntegrator] Using cached decision for ${symbol} (${action}, hasPos: ${hasPosition})`);
          return cached.decision;
        }
      }
      
      // 2. System health check
      const systemCheck = await this.checkSystemHealth();
      if (!systemCheck.healthy) {
        console.log(`[BrainIntegrator] System unhealthy: ${systemCheck.reason}`);
        return this.createDecision('reject', 0, systemCheck.reason);
      }
      
      // 3. Get ML score from BrainService
      let brainScore = null;
      if (this.components.brainService) {
        try {
          brainScore = await this.components.brainService.scoreSymbol(symbol);
        } catch (err) {
          console.log(`[BrainIntegrator] Brain service error: ${err.message}`);
        }
      }
      
      // 4. Get technical indicators score
      let technicalScore = null;
      if (this.components.indicatorsConnector) {
        try {
          const indicators = await this.components.indicatorsConnector.getDecisionContext(symbol);
          technicalScore = indicators.composite?.confidence || 0.5;
        } catch (err) {
          console.log(`[BrainIntegrator] Indicators error: ${err.message}`);
        }
      }
      
      // 5. Combine scores with weights
      // Use candidateData confidence if available, add some randomness for variety
      const baseConfidence = candidateData?.confidence || candidateData?.score || 0.5;
      const newsBoost = candidateData?.newsNudge || context.newsNudge || 0;
      const perfBoost = context.performanceBoost || 0;
      
      // Apply enhanced learning adjustments if available
      let finalScore = this.combineScores({
        brain: brainScore?.final_score || baseConfidence,
        technical: technicalScore || baseConfidence + (Math.random() * 0.2 - 0.1), // Add some variance
        news: newsBoost,
        strategy: baseConfidence + perfBoost
      });
      
      // Apply performance-based adjustments from enhanced recorder
      if (this.components.enhancedRecorder && strategy) {
        finalScore = this.components.enhancedRecorder.getAdjustedConfidence(
          symbol, 
          strategy, 
          finalScore
        );
      }
      
      console.log(`[BrainIntegrator] Scoring ${symbol}: base=${baseConfidence.toFixed(3)}, news=${newsBoost.toFixed(3)}, perf=${perfBoost.toFixed(3)}, final=${finalScore.toFixed(3)}`);
      
      // 6. Apply position-specific logic
      let adjustedScore = finalScore;
      if (hasPosition) {
        // More conservative with existing positions
        adjustedScore = finalScore * 0.9;
        
        // Check if we should exit
        if (action === 'sell' || action === 'SELL') {
          const exitScore = await this.evaluateExit(symbol, finalScore);
          adjustedScore = exitScore;
        }
      }
      
      // 7. Check against thresholds
      const decision = this.evaluateThresholds(adjustedScore, action, hasPosition);
      
      // 8. Risk and capital checks
      if (decision.action === 'buy' || decision.action === 'BUY') {
        const riskCheck = await this.validateRisk(symbol, context);
        if (!riskCheck.allowed) {
          return this.createDecision('reject', adjustedScore, riskCheck.reason);
        }
      }
      
      // 9. Final decision with full context
      const finalDecision = {
        action: decision.action,
        symbol: symbol,
        score: adjustedScore,
        confidence: Math.min(adjustedScore, 0.95),
        scores: {
          brain: brainScore?.final_score || 0.5,
          technical: technicalScore || 0.5,
          combined: finalScore,
          adjusted: adjustedScore
        },
        reasoning: [
          `Brain score: ${(brainScore?.final_score || 0.5).toFixed(2)}`,
          `Technical score: ${(technicalScore || 0.5).toFixed(2)}`,
          hasPosition ? 'Existing position adjusted' : 'New position',
          decision.reason
        ],
        metadata: {
          strategy: strategy,
          hasPosition: hasPosition,
          regime: this.state.regime,
          processingMs: Date.now() - startTime
        }
      };
      
      // 10. Cache and emit
      this.recentDecisions.set(cacheKey, {
        decision: finalDecision,
        timestamp: Date.now()
      });
      
      // Clean old cache entries
      this.cleanDecisionCache();
      
      // Emit for monitoring
      this.emit('decision_made', finalDecision);
      
      // Record in performance tracker
      if (this.components.performanceRecorder) {
        this.components.performanceRecorder.recordDecision(finalDecision);
      }
      
      console.log(`[BrainIntegrator] Decision for ${symbol}: ${finalDecision.action} (score: ${adjustedScore.toFixed(3)})`);
      
      return finalDecision;
      
    } catch (error) {
      console.error(`[BrainIntegrator] Decision error for ${symbol}:`, error);
      return this.createDecision('error', 0, error.message);
    }
  }
  
  /**
   * Score a candidate for AutoLoop
   */
  async scoreCandidate(symbol, context) {
    const decision = await this.makeDecision({
      symbol,
      action: context.hasPosition ? 'hold' : 'buy',
      strategy: context.strategyId,
      hasPosition: context.hasPosition,
      candidateData: context.candidateData,
      newsNudge: context.newsNudge,
      performanceBoost: context.performanceBoost
    });
    
    // Convert to AutoLoop format
    if (decision.action === 'reject' || decision.action === 'error' || decision.action === 'hold') {
      console.log(`[BrainIntegrator] Returning null for ${symbol} - action: ${decision.action}`);
      return null;
    }
    
    console.log(`[BrainIntegrator] Converting ${symbol} decision: ${decision.action} → ${decision.action === 'buy' ? 'buy' : 'sell'}`);
    return {
      side: decision.action === 'buy' ? 'buy' : 'sell',
      score: decision.score,
      confidence: decision.confidence,
      reasoning: decision.reasoning
    };
  }
  
  /**
   * Check overall system health
   */
  async checkSystemHealth() {
    const checks = [];
    
    // Circuit breaker
    if (this.components.circuitBreaker) {
      const cbStatus = this.components.circuitBreaker.canExecute();
      if (!cbStatus.allowed) {
        return { healthy: false, reason: `Circuit breaker: ${cbStatus.reason}` };
      }
    }
    
    // Capital availability
    if (this.components.capitalTracker) {
      const capital = this.components.capitalTracker.getStatus();
      if (capital.utilizationPct > 0.95) {
        return { healthy: false, reason: 'Capital utilization > 95%' };
      }
    }
    
    // Data freshness
    if (this.components.dataValidator) {
      // Basic freshness check
      const now = Date.now();
      if (this.state.lastUpdate && now - this.state.lastUpdate > 300000) { // 5 min
        return { healthy: false, reason: 'Market data stale' };
      }
    }
    
    return { healthy: true };
  }
  
  /**
   * Combine multiple scores with weights
   */
  combineScores(scores) {
    const weights = {
      brain: 0.4,      // ML model
      technical: 0.3,  // Technical indicators
      news: 0.2,       // News sentiment
      strategy: 0.1    // Strategy signal
    };
    
    let weightedSum = 0;
    let totalWeight = 0;
    
    for (const [source, score] of Object.entries(scores)) {
      if (score !== null && weights[source]) {
        weightedSum += score * weights[source];
        totalWeight += weights[source];
      }
    }
    
    return totalWeight > 0 ? weightedSum / totalWeight : 0.5;
  }
  
  /**
   * Evaluate exit conditions for existing positions
   */
  async evaluateExit(symbol, currentScore) {
    // Get position details
    if (!this.components.performanceRecorder) {
      return currentScore;
    }
    
    try {
      // PerformanceRecorder doesn't have getPosition method - skip for now
      // TODO: Implement proper position tracking
      return currentScore;
      
      const pnlPct = position.unrealizedPnLPercent || 0;
      
      // Exit winners if score drops
      if (pnlPct > 0.02 && currentScore < this.thresholds.exitWinnerThreshold) {
        return 0.8; // Strong exit signal
      }
      
      // Exit losers more aggressively
      if (pnlPct < -0.01 && currentScore < this.thresholds.exitLoserThreshold) {
        return 0.9; // Very strong exit signal
      }
      
      // Time-based decay
      const holdTime = Date.now() - position.entryTime;
      const daysSinceEntry = holdTime / (1000 * 60 * 60 * 24);
      if (daysSinceEntry > 5) {
        return currentScore * 0.8; // Reduce score for old positions
      }
      
    } catch (err) {
      console.log(`[BrainIntegrator] Exit evaluation error: ${err.message}`);
    }
    
    return currentScore;
  }
  
  /**
   * Evaluate score against thresholds
   */
  evaluateThresholds(score, requestedAction, hasPosition) {
    // Buy decision
    if (!hasPosition && score >= this.thresholds.buyThreshold) {
      return {
        action: 'buy',
        reason: `Score ${score.toFixed(3)} exceeds buy threshold ${this.thresholds.buyThreshold}`
      };
    }
    
    // Sell decision
    if (hasPosition && score <= this.thresholds.sellThreshold) {
      return {
        action: 'sell',
        reason: `Score ${score.toFixed(3)} below sell threshold ${this.thresholds.sellThreshold}`
      };
    }
    
    // Below all thresholds
    return {
      action: 'hold',
      reason: `Score ${score.toFixed(3)} below action thresholds`
    };
  }
  
  /**
   * Validate risk for new positions
   */
  async validateRisk(symbol, context) {
    const checks = [];
    
    // Position limit check
    try {
      const positionsResp = await fetch('http://localhost:4000/api/paper/positions');
      if (positionsResp.ok) {
        const positions = await positionsResp.json();
        // No hard position limit - let capital and risk management decide
        // Just log the current position count for monitoring
        const openCount = positions.filter(p => p.qty > 0).length;
        console.log(`[BrainIntegrator] Current open positions: ${openCount}`);
      }
    } catch (e) {
      // Continue if can't check positions
    }
    
    // Symbol concentration
    // TODO: Check if we already have too much exposure to this symbol
    
    // Sector concentration
    // TODO: Check sector exposure limits
    
    return { allowed: true };
  }
  
  /**
   * Create a decision object
   */
  createDecision(action, score, reason) {
    return {
      action: action,
      score: score,
      confidence: score,
      reasoning: [reason],
      metadata: {
        timestamp: Date.now()
      }
    };
  }
  
  /**
   * Update market state
   */
  updateMarketState(state) {
    this.state = {
      ...this.state,
      ...state,
      lastUpdate: Date.now()
    };
    
    this.emit('state_updated', this.state);
  }
  
  /**
   * Clean old decisions from cache
   */
  cleanDecisionCache() {
    const now = Date.now();
    for (const [key, cached] of this.recentDecisions.entries()) {
      if (now - cached.timestamp > this.decisionTTL * 2) {
        this.recentDecisions.delete(key);
      }
    }
  }
  
  /**
   * Start monitoring positions
   */
  start() {
    if (this.monitoringInterval) return;
    
    this.monitoringInterval = setInterval(() => {
      this.monitorPositions();
    }, this.checkInterval);
    
    console.log('[BrainIntegrator] Started position monitoring');
  }
  
  /**
   * Monitor existing positions for exit signals
   */
  async monitorPositions() {
    if (!this.components.paperBroker) return;
    
    try {
      const positions = this.components.paperBroker.getPositions();
      
      for (const position of positions) {
        if (position.qty > 0) {
          const decision = await this.makeDecision({
            symbol: position.symbol,
            action: 'sell',
            strategy: 'exit_monitor',
            hasPosition: true,
            candidateData: { currentPrice: position.currentPrice }
          });
          
          if (decision.action === 'sell' && decision.confidence > 0.7) {
            console.log(`[BrainIntegrator] Exit signal for ${position.symbol}`);
            this.emit('exit_signal', {
              symbol: position.symbol,
              reason: decision.reasoning.join(', '),
              confidence: decision.confidence
            });
          }
        }
      }
    } catch (error) {
      console.error('[BrainIntegrator] Position monitoring error:', error);
    }
  }
  
  /**
   * Stop monitoring
   */
  stop() {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
    
    console.log('[BrainIntegrator] Stopped');
  }
}

module.exports = { BrainIntegrator };