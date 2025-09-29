/**
 * System Integrator - Connects all isolated components
 * The "nervous system" of the trading brain
 */

class SystemIntegrator {
  constructor(config) {
    this.components = {
      autoLoop: config.autoLoop,
      botCompetition: config.botCompetition,
      evoTester: config.evoTester,
      strategyManager: config.strategyManager,
      aiOrchestrator: config.aiOrchestrator,
      tournament: config.tournament,
      performanceRecorder: config.performanceRecorder,
      newsSystem: config.newsSystem,
      geneticInheritance: config.geneticInheritance
    };
    
    this.memory = new Map(); // Until we add a real database
    this.setupConnections();
  }

  /**
   * Wire all the connections between systems
   */
  setupConnections() {
    // TODO: Most components need to extend EventEmitter for this to work properly
    // For now, we'll skip event-based connections and rely on direct method calls
    
    console.log('[SystemIntegrator] Components connected (event listeners disabled temporarily)');
    
    // Components are connected but event-based communication needs to be implemented
    // The AI components can still communicate through direct method calls
  }

  /**
   * Feed trading results back to bot competition
   */
  feedbackToBotCompetition(trade) {
    // Find which bot's strategy this trade belongs to
    const bot = this.findBotByStrategy(trade.strategy);
    if (bot) {
      this.components.botCompetition.recordTrade(bot.id, trade);
    }
  }

  /**
   * Trigger experiments based on news
   */
  async triggerTargetedExperiments(newsEvent) {
    // Start bot competition focused on affected symbols
    const config = {
      symbols: newsEvent.affectedSymbols,
      initialCapital: 50,
      botCount: 100,
      metadata: {
        trigger: 'news',
        event: newsEvent
      }
    };
    
    await this.components.botCompetition.startCompetition(config);
  }

  /**
   * Evolve strategies based on performance data
   */
  async evolveStrategiesBasedOnPerformance(performanceData) {
    // Feed performance data to EvoTester
    const evolutionRequest = {
      baseStrategies: performanceData.topStrategies,
      marketConditions: await this.getCurrentMarketConditions(),
      objective: 'maximize_sharpe_minimize_drawdown'
    };
    
    await this.components.evoTester.evolve(evolutionRequest);
  }

  /**
   * Store successful strategies with context
   */
  storeSuccessfulStrategy(strategy) {
    const context = {
      strategy: strategy,
      marketConditions: this.getCurrentMarketConditions(),
      performance: strategy.performance,
      timestamp: new Date().toISOString()
    };
    
    // Store in memory (should be database)
    const key = `${context.marketConditions.regime}-${strategy.type}`;
    if (!this.memory.has(key)) {
      this.memory.set(key, []);
    }
    this.memory.get(key).push(context);
  }

  /**
   * Get current market conditions
   */
  async getCurrentMarketConditions() {
    // This would integrate with market indicators
    return {
      regime: 'neutral',
      volatility: 'medium',
      trend: 'sideways',
      vix: 17
    };
  }

  /**
   * Find bot by strategy
   */
  findBotByStrategy(strategyId) {
    // This would query active bots
    return this.components.botCompetition.getBotByStrategy(strategyId);
  }

  /**
   * Start new competition with evolved bots
   */
  async startNewCompetition(bots) {
    const config = {
      durationDays: 7,
      initialCapitalMin: 50,
      initialCapitalMax: 50,
      totalPoolCapital: 5000
    };
    
    const competition = await this.components.botCompetition.startCompetition(config);
    
    // Add evolved bots
    for (const bot of bots) {
      await this.components.botCompetition.addBot(competition.id, bot);
    }
  }

  /**
   * Master learning cycle
   */
  async runLearningCycle() {
    console.log('[SystemIntegrator] Running learning cycle...');
    
    // 1. Analyze recent performance
    const recentPerformance = await this.components.performanceRecorder.getRecentMetrics();
    
    // 2. Extract lessons
    const lessons = this.extractLessons(recentPerformance);
    
    // 3. Update all systems
    await this.propagateLearning(lessons);
    
    // 4. Start new experiments
    await this.startNewExperiments(lessons);
    
    console.log('[SystemIntegrator] Learning cycle complete');
  }

  /**
   * Extract lessons from performance
   */
  extractLessons(performance) {
    return {
      successfulStrategies: performance.filter(p => p.sharpe > 1.5),
      failedStrategies: performance.filter(p => p.sharpe < 0),
      marketConditions: this.getCurrentMarketConditions(),
      timestamp: new Date()
    };
  }

  /**
   * Propagate learning to all systems
   */
  async propagateLearning(lessons) {
    // Update AI Orchestrator
    await this.components.aiOrchestrator.updatePolicy(lessons);
    
    // Update Strategy Manager
    await this.components.strategyManager.adjustStrategies(lessons);
    
    // Update Evolution parameters
    await this.components.evoTester.updateEvolutionParams(lessons);
  }

  /**
   * Start new experiments based on lessons
   */
  async startNewExperiments(lessons) {
    // Bot competition with new insights
    const newBots = await this.components.geneticInheritance.generateFromLessons(lessons);
    await this.startNewCompetition(newBots);
  }
}

module.exports = { SystemIntegrator };
