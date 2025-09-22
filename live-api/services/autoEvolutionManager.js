/**
 * Auto Evolution Manager
 * 
 * Manages automatic bot competition cycling based on trade count,
 * time intervals, and market conditions.
 */

const EventEmitter = require('events');

class AutoEvolutionManager extends EventEmitter {
  constructor(botCompetitionService, performanceRecorder, geneticInheritance) {
    super();
    this.botCompetitionService = botCompetitionService;
    this.performanceRecorder = performanceRecorder;
    this.geneticInheritance = geneticInheritance;
    
    this.config = {
      tradesPerCycle: 50,  // Start new competition after 50 trades
      maxConcurrentCompetitions: 1,
      minTimeBetweenCompetitions: 1 * 60 * 60 * 1000, // 1 hour
      autoEvolutionEnabled: true,
      scheduleInterval: 4 * 60 * 60 * 1000  // Check every 4 hours
    };
    
    this.stats = {
      totalCycles: 0,
      lastCycleTime: null,
      totalTradesSinceLastCycle: 0,
      competitionsStarted: 0
    };
    
    this.scheduleTimer = null;
    this.isRunning = false;
  }
  
  /**
   * Start the auto evolution manager
   */
  start() {
    if (this.isRunning) return;
    
    console.log('[AutoEvolution] Starting automatic evolution cycling');
    this.isRunning = true;
    
    // Track trades for triggering new cycles
    this.performanceRecorder.on('trade_recorded', this.onTradeRecorded.bind(this));
    
    // Handle competition completions
    this.botCompetitionService.on('competition_complete', this.onCompetitionComplete.bind(this));
    
    // Start scheduled checks
    this.startScheduledChecks();
  }
  
  /**
   * Stop the auto evolution manager
   */
  stop() {
    console.log('[AutoEvolution] Stopping automatic evolution cycling');
    this.isRunning = false;
    
    if (this.scheduleTimer) {
      clearInterval(this.scheduleTimer);
      this.scheduleTimer = null;
    }
    
    this.performanceRecorder.removeAllListeners('trade_recorded');
    this.botCompetitionService.removeAllListeners('competition_complete');
  }
  
  /**
   * Handle trade recorded event
   */
  onTradeRecorded(trade) {
    if (!this.config.autoEvolutionEnabled) return;
    
    this.stats.totalTradesSinceLastCycle++;
    
    // Check if we should start a new cycle
    if (this.stats.totalTradesSinceLastCycle >= this.config.tradesPerCycle) {
      this.checkAndStartNewCycle('trade_threshold');
    }
  }
  
  /**
   * Handle competition complete event
   */
  async onCompetitionComplete(results) {
    console.log(`[AutoEvolution] Competition ${results.competitionId} completed`);
    
    try {
      // Extract genes from winners
      const topN = Math.min(10, results.bots.length);
      const genes = await this.geneticInheritance.extractTopPerformers(
        results.competitionId, 
        topN
      );
      
      // Store winning strategies for future reference
      this.emit('winning_genes_extracted', {
        competitionId: results.competitionId,
        genes: genes,
        timestamp: new Date().toISOString()
      });
      
      console.log(`[AutoEvolution] Extracted ${genes.length} winning gene sets`);
      
    } catch (error) {
      console.error('[AutoEvolution] Error extracting genes:', error);
    }
  }
  
  /**
   * Start scheduled checks for evolution triggers
   */
  startScheduledChecks() {
    // Check every N hours for evolution opportunities
    this.scheduleTimer = setInterval(() => {
      this.checkEvolutionTriggers();
    }, this.config.scheduleInterval);
    
    // Initial check
    this.checkEvolutionTriggers();
  }
  
  /**
   * Check various triggers for starting evolution
   */
  async checkEvolutionTriggers() {
    console.log('[AutoEvolution] Checking evolution triggers...');
    
    const triggers = [];
    
    // 1. Time-based trigger
    const timeSinceLastCycle = this.stats.lastCycleTime 
      ? Date.now() - this.stats.lastCycleTime 
      : Infinity;
      
    if (timeSinceLastCycle > 6 * 60 * 60 * 1000) { // 6 hours
      triggers.push('scheduled_time');
    }
    
    // 2. Performance stagnation trigger
    const recentPerformance = await this.checkPerformanceStagnation();
    if (recentPerformance.isStagnant) {
      triggers.push('performance_stagnation');
    }
    
    // 3. Market regime change trigger
    const marketRegimeChanged = await this.checkMarketRegimeChange();
    if (marketRegimeChanged) {
      triggers.push('market_regime_change');
    }
    
    // Start new cycle if any triggers are active
    if (triggers.length > 0) {
      console.log(`[AutoEvolution] Triggers active: ${triggers.join(', ')}`);
      this.checkAndStartNewCycle(triggers[0]);
    }
  }
  
  /**
   * Check if we should start a new evolution cycle
   */
  async checkAndStartNewCycle(trigger) {
    // Check if we can start a new competition
    const activeCompetitions = this.botCompetitionService.getActiveCompetitions();
    if (activeCompetitions.length >= this.config.maxConcurrentCompetitions) {
      console.log('[AutoEvolution] Max concurrent competitions reached, skipping cycle');
      return;
    }
    
    // Check minimum time between competitions
    const timeSinceLastCycle = this.stats.lastCycleTime 
      ? Date.now() - this.stats.lastCycleTime 
      : Infinity;
      
    if (timeSinceLastCycle < this.config.minTimeBetweenCompetitions) {
      console.log('[AutoEvolution] Too soon since last cycle, skipping');
      return;
    }
    
    // Start new evolution cycle
    await this.startEvolutionCycle(trigger);
  }
  
  /**
   * Start a new evolution cycle
   */
  async startEvolutionCycle(trigger) {
    console.log(`[AutoEvolution] Starting new evolution cycle, trigger: ${trigger}`);
    
    try {
      // Get previous competition results for genetic breeding
      const previousCompetitions = Array.from(this.botCompetitionService.competitions.values())
        .filter(c => c.status === 'completed')
        .slice(-3); // Last 3 competitions
      
      // Generate new strategies based on past winners
      const newStrategies = await this.generateEvolvedStrategies(previousCompetitions);
      
      // Start new competition
      const competitionConfig = {
        durationDays: 3, // Shorter cycles for faster evolution
        initialCapitalMin: 50,
        initialCapitalMax: 50,
        totalPoolCapital: 5000,  // 100 bots * $50
        winnerBonus: 0.3,  // 30% bonus for winners
        loserPenalty: 0.6, // 60% penalty for losers
        reallocationIntervalHours: 0.5, // Every 30 minutes
        metadata: {
          trigger: trigger,
          generation: this.stats.totalCycles + 1,
          autoEvolution: true
        }
      };
      
      const competition = this.botCompetitionService.startCompetition(competitionConfig);
      
      // Add evolved bots
      for (let i = 0; i < newStrategies.length; i++) {
        this.botCompetitionService.addBot(competition.id, newStrategies[i]);
      }
      
      // Update stats
      this.stats.totalCycles++;
      this.stats.lastCycleTime = Date.now();
      this.stats.totalTradesSinceLastCycle = 0;
      this.stats.competitionsStarted++;
      
      console.log(`[AutoEvolution] Started competition ${competition.id} with ${newStrategies.length} evolved bots`);
      
      this.emit('cycle_started', {
        competitionId: competition.id,
        trigger: trigger,
        generation: this.stats.totalCycles,
        botCount: newStrategies.length
      });
      
    } catch (error) {
      console.error('[AutoEvolution] Error starting evolution cycle:', error);
    }
  }
  
  /**
   * Generate evolved strategies from past winners
   */
  async generateEvolvedStrategies(previousCompetitions) {
    const strategies = [];
    
    // If no previous competitions, use diverse random strategies
    if (previousCompetitions.length === 0) {
      return this.generateDiverseStrategies(100);
    }
    
    // Extract winning genes from previous competitions
    const winningGenes = [];
    for (const comp of previousCompetitions) {
      if (comp.bots) {
        const topBots = Array.from(comp.bots.values())
          .sort((a, b) => (b.performance?.totalReturn || 0) - (a.performance?.totalReturn || 0))
          .slice(0, 10);
        
        winningGenes.push(...topBots);
      }
    }
    
    // Breed new generation
    const evolvedStrategies = this.breedNewGeneration(winningGenes);
    
    // Add some random mutations for diversity
    const mutations = this.generateDiverseStrategies(20);
    
    return [...evolvedStrategies, ...mutations];
  }
  
  /**
   * Generate diverse random strategies
   */
  generateDiverseStrategies(count) {
    const strategies = [];
    const types = ['rsi_reversion', 'volatility_breakout', 'mean_reversion', 'ma_crossover', 'momentum', 'breakout'];
    const symbols = [
      'SNDL', 'TLRY', 'BB', 'NOK', 'PLTR', 'RIOT', 'MARA', 'OCGN',
      'PROG', 'ATER', 'CEI', 'FAMI', 'XELA', 'GNUS', 'ZOM', 'NAKD',
      'CLOV', 'WISH', 'RIG', 'WKHS', 'GOEV', 'RIDE', 'NKLA', 'SPCE',
      'F', 'GE', 'SNAP', 'LYFT', 'UBER', 'DKNG', 'PENN', 'FUBO'
    ];
    
    for (let i = 0; i < count; i++) {
      const type = types[i % types.length];
      const symbol = symbols[i % symbols.length];
      
      strategies.push({
        name: `AutoEvo-${type}-Gen${this.stats.totalCycles}-${i}`,
        type: type,
        symbol: symbol,
        generation: this.stats.totalCycles + 1,
        metadata: {
          autoEvolved: true,
          parentGeneration: this.stats.totalCycles
        }
      });
    }
    
    return strategies;
  }
  
  /**
   * Breed new generation from winning genes
   */
  breedNewGeneration(parents) {
    const offspring = [];
    const targetCount = 80; // 80 bred + 20 mutations = 100 total
    
    if (parents.length < 2) {
      return this.generateDiverseStrategies(targetCount);
    }
    
    // Cross-breed top performers
    for (let i = 0; i < targetCount; i++) {
      const parent1 = parents[i % parents.length];
      const parent2 = parents[(i + 1) % parents.length];
      
      const child = {
        name: `AutoEvo-${parent1.type}-X-${parent2.type}-Gen${this.stats.totalCycles}-${i}`,
        type: Math.random() > 0.5 ? parent1.type : parent2.type,
        symbol: Math.random() > 0.5 ? parent1.symbol : parent2.symbol,
        generation: this.stats.totalCycles + 1,
        metadata: {
          autoEvolved: true,
          parentGeneration: this.stats.totalCycles,
          parent1: parent1.name,
          parent2: parent2.name,
          inheritedTraits: {
            riskTolerance: (parent1.riskTolerance || 0.5) * 0.6 + (parent2.riskTolerance || 0.5) * 0.4,
            timeframe: Math.random() > 0.5 ? parent1.timeframe : parent2.timeframe
          }
        }
      };
      
      offspring.push(child);
    }
    
    return offspring;
  }
  
  /**
   * Check for performance stagnation
   */
  async checkPerformanceStagnation() {
    // Get recent performance metrics
    const recentTrades = await this.performanceRecorder.getRecentTrades(50);
    
    if (recentTrades.length < 20) {
      return { isStagnant: false };
    }
    
    // Calculate rolling sharpe ratios
    const firstHalf = recentTrades.slice(0, 25);
    const secondHalf = recentTrades.slice(25);
    
    const firstHalfReturn = firstHalf.reduce((sum, t) => sum + (t.profit || 0), 0);
    const secondHalfReturn = secondHalf.reduce((sum, t) => sum + (t.profit || 0), 0);
    
    // If performance hasn't improved, consider it stagnant
    const isStagnant = secondHalfReturn <= firstHalfReturn * 0.9;
    
    return { 
      isStagnant,
      firstHalfReturn,
      secondHalfReturn
    };
  }
  
  /**
   * Check for market regime change
   */
  async checkMarketRegimeChange() {
    // This would integrate with market indicators service
    // For now, return false
    return false;
  }
  
  /**
   * Get current status
   */
  getStatus() {
    return {
      isRunning: this.isRunning,
      config: this.config,
      stats: this.stats,
      activeCompetitions: this.botCompetitionService.getActiveCompetitions().length
    };
  }
}

module.exports = AutoEvolutionManager;
