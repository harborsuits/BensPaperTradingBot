const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const { EventEmitter } = require('events');

class EvolutionBridge extends EventEmitter {
  constructor(strategyManager, performanceRecorder) {
    super();
    this.strategyManager = strategyManager;
    this.performanceRecorder = performanceRecorder;
    
    // Configuration
    this.pythonPath = process.env.PYTHON_PATH || 'python3';
    this.evolutionScript = path.join(__dirname, '../../trading_bot/core/strategy_evolution.py');
    this.evotraderPath = path.join(__dirname, '../../Evotrader');
    
    // State
    this.isRunning = false;
    this.currentGeneration = 0;
    this.population = [];
    this.bestStrategies = [];
    
    // Evolution parameters (from ai_policy.yaml)
    this.config = {
      populationSize: 50,
      generations: 20,
      mutationRate: 0.1,
      crossoverRate: 0.7,
      elitePercent: 0.2,
      tournamentSize: 5,
      testCapital: 2000, // $2k for testing
      minFitness: 0.6,
      maxConcurrentTests: 5
    };
    
    // Test results storage
    this.testResults = new Map();
    
    console.log('[EvolutionBridge] Initialized');
  }
  
  async startEvolution(options = {}) {
    if (this.isRunning) {
      console.log('[EvolutionBridge] Evolution already running');
      return false;
    }
    
    this.isRunning = true;
    this.currentGeneration = 0;
    
    const config = { ...this.config, ...options };
    
    try {
      // Initialize population with existing strategies
      await this.initializePopulation(config);
      
      // Run evolution cycles
      for (let gen = 0; gen < config.generations; gen++) {
        this.currentGeneration = gen;
        console.log(`[EvolutionBridge] Starting generation ${gen + 1}/${config.generations}`);
        
        // Evaluate fitness
        await this.evaluatePopulation();
        
        // Select and breed
        if (gen < config.generations - 1) {
          await this.evolveNextGeneration(config);
        }
        
        // Report progress
        this.reportGenerationStats();
        
        // Check for early stopping
        if (this.shouldStopEarly()) {
          console.log('[EvolutionBridge] Early stopping - fitness plateau reached');
          break;
        }
      }
      
      // Promote best strategies
      await this.promoteBestStrategies();
      
    } catch (error) {
      console.error('[EvolutionBridge] Evolution error:', error);
    } finally {
      this.isRunning = false;
    }
    
    return true;
  }
  
  async initializePopulation(config) {
    // Get existing strategies as seeds
    const baseStrategies = this.strategyManager.getAllStrategies();
    
    this.population = [];
    
    // Create variants of existing strategies
    for (const strategy of baseStrategies) {
      // Original
      this.population.push({
        id: `${strategy.name}_gen0_original`,
        baseStrategy: strategy.name,
        parameters: this.extractParameters(strategy),
        generation: 0,
        fitness: 0,
        trades: 0,
        pnl: 0
      });
      
      // Create mutations
      for (let i = 0; i < 3; i++) {
        this.population.push({
          id: `${strategy.name}_gen0_mut${i}`,
          baseStrategy: strategy.name,
          parameters: this.mutateParameters(this.extractParameters(strategy)),
          generation: 0,
          fitness: 0,
          trades: 0,
          pnl: 0
        });
      }
    }
    
    // Fill remaining population with random strategies
    while (this.population.length < config.populationSize) {
      const baseIdx = Math.floor(Math.random() * baseStrategies.length);
      const base = baseStrategies[baseIdx];
      
      this.population.push({
        id: `random_gen0_${this.population.length}`,
        baseStrategy: base.name,
        parameters: this.randomizeParameters(this.extractParameters(base)),
        generation: 0,
        fitness: 0,
        trades: 0,
        pnl: 0
      });
    }
    
    console.log(`[EvolutionBridge] Initialized population with ${this.population.length} strategies`);
  }
  
  extractParameters(strategy) {
    // Extract configurable parameters from strategy
    const params = {};
    
    if (strategy.config) {
      // Common trading parameters
      params.stopLoss = strategy.config.stopLoss || 0.02;
      params.takeProfit = strategy.config.takeProfit || 0.05;
      params.entryThreshold = strategy.config.entryThreshold || 0.7;
      params.exitThreshold = strategy.config.exitThreshold || 0.3;
      params.positionSize = strategy.config.positionSize || 0.01;
      params.maxPositions = strategy.config.maxPositions || 3;
      
      // Strategy-specific parameters
      if (strategy.name.includes('MA')) {
        params.fastPeriod = strategy.config.fastPeriod || 10;
        params.slowPeriod = strategy.config.slowPeriod || 20;
      }
      
      if (strategy.name.includes('RSI')) {
        params.rsiPeriod = strategy.config.rsiPeriod || 14;
        params.rsiOverbought = strategy.config.rsiOverbought || 70;
        params.rsiOversold = strategy.config.rsiOversold || 30;
      }
    }
    
    return params;
  }
  
  mutateParameters(params) {
    const mutated = { ...params };
    
    for (const key in mutated) {
      if (Math.random() < this.config.mutationRate) {
        const value = mutated[key];
        
        if (typeof value === 'number') {
          // Mutate by +/- 10-30%
          const factor = 1 + (Math.random() - 0.5) * 0.6;
          mutated[key] = value * factor;
          
          // Ensure reasonable bounds
          if (key.includes('Loss') || key.includes('Profit')) {
            mutated[key] = Math.max(0.001, Math.min(0.2, mutated[key]));
          } else if (key.includes('Period')) {
            mutated[key] = Math.round(Math.max(2, Math.min(200, mutated[key])));
          } else if (key.includes('Threshold')) {
            mutated[key] = Math.max(0, Math.min(1, mutated[key]));
          }
        }
      }
    }
    
    return mutated;
  }
  
  randomizeParameters(params) {
    const randomized = { ...params };
    
    for (const key in randomized) {
      const value = randomized[key];
      
      if (typeof value === 'number') {
        // Randomize within reasonable ranges
        if (key.includes('Loss') || key.includes('Profit')) {
          randomized[key] = 0.005 + Math.random() * 0.095; // 0.5% to 10%
        } else if (key.includes('Period')) {
          randomized[key] = Math.round(5 + Math.random() * 95); // 5 to 100
        } else if (key.includes('Threshold')) {
          randomized[key] = 0.2 + Math.random() * 0.6; // 0.2 to 0.8
        } else if (key.includes('Size')) {
          randomized[key] = 0.005 + Math.random() * 0.045; // 0.5% to 5%
        }
      }
    }
    
    return randomized;
  }
  
  async evaluatePopulation() {
    console.log('[EvolutionBridge] Evaluating population fitness...');
    
    // Batch evaluate strategies
    const batchSize = this.config.maxConcurrentTests;
    
    for (let i = 0; i < this.population.length; i += batchSize) {
      const batch = this.population.slice(i, i + batchSize);
      
      await Promise.all(batch.map(strategy => this.evaluateStrategy(strategy)));
    }
    
    // Sort by fitness
    this.population.sort((a, b) => b.fitness - a.fitness);
    
    // Store best performers
    this.bestStrategies = this.population.slice(0, Math.ceil(this.population.length * 0.1));
  }
  
  async evaluateStrategy(strategy) {
    // Run paper trading test with isolated capital
    const testCapital = this.config.testCapital;
    const testDuration = 24 * 60 * 60 * 1000; // 24 hours
    
    try {
      // Create temporary strategy instance
      const tempStrategy = this.createTempStrategy(strategy);
      
      // Run backtest or paper trade
      const results = await this.runIsolatedTest(tempStrategy, testCapital, testDuration);
      
      // Calculate fitness
      strategy.trades = results.trades;
      strategy.pnl = results.pnl;
      strategy.fitness = this.calculateFitness(results);
      
      // Store detailed results
      this.testResults.set(strategy.id, results);
      
    } catch (error) {
      console.error(`[EvolutionBridge] Error evaluating ${strategy.id}:`, error.message);
      strategy.fitness = 0;
    }
  }
  
  createTempStrategy(strategyConfig) {
    // Create a temporary strategy instance with evolved parameters
    const base = this.strategyManager.strategies.get(strategyConfig.baseStrategy);
    
    if (!base) {
      throw new Error(`Base strategy ${strategyConfig.baseStrategy} not found`);
    }
    
    // Clone and modify
    return {
      name: strategyConfig.id,
      instance: {
        ...base.instance,
        config: { ...base.instance.config, ...strategyConfig.parameters }
      },
      status: 'testing',
      config: strategyConfig.parameters
    };
  }
  
  async runIsolatedTest(strategy, capital, duration) {
    // This would integrate with backtesting engine or paper broker
    // For now, simulate results based on parameters
    
    const results = {
      trades: Math.floor(Math.random() * 50) + 10,
      wins: 0,
      losses: 0,
      pnl: 0,
      maxDrawdown: 0,
      sharpe: 0
    };
    
    // Simulate trades based on strategy parameters
    for (let i = 0; i < results.trades; i++) {
      const isWin = Math.random() < (0.4 + strategy.config.entryThreshold * 0.3);
      
      if (isWin) {
        results.wins++;
        const profit = capital * strategy.config.takeProfit * Math.random();
        results.pnl += profit;
      } else {
        results.losses++;
        const loss = capital * strategy.config.stopLoss * Math.random();
        results.pnl -= loss;
      }
    }
    
    // Calculate metrics
    results.winRate = results.wins / results.trades;
    results.profitFactor = results.wins > 0 ? 
      (results.pnl + results.losses * capital * strategy.config.stopLoss) / 
      (results.losses * capital * strategy.config.stopLoss) : 0;
    
    results.sharpe = results.pnl / (capital * 0.2); // Simplified Sharpe
    results.maxDrawdown = Math.max(0.05, Math.random() * 0.3);
    
    return results;
  }
  
  calculateFitness(results) {
    // Multi-objective fitness function
    let fitness = 0;
    
    // Profit factor (40%)
    fitness += Math.min(2, results.profitFactor || 0) * 0.2;
    
    // Win rate (20%)
    fitness += (results.winRate || 0) * 0.2;
    
    // Sharpe ratio (20%)
    fitness += Math.min(2, Math.max(0, results.sharpe || 0)) * 0.1;
    
    // Max drawdown (20%)
    fitness += Math.max(0, 1 - (results.maxDrawdown || 0.5)) * 0.2;
    
    // Trade frequency bonus (up to 10%)
    const tradeFreq = Math.min(1, results.trades / 50);
    fitness += tradeFreq * 0.1;
    
    // PnL bonus (up to 20%)
    const pnlRatio = results.pnl / this.config.testCapital;
    fitness += Math.min(0.2, Math.max(-0.1, pnlRatio));
    
    return Math.max(0, Math.min(1, fitness));
  }
  
  async evolveNextGeneration(config) {
    const nextGen = [];
    
    // Elite selection
    const eliteCount = Math.ceil(this.population.length * config.elitePercent);
    for (let i = 0; i < eliteCount; i++) {
      const elite = { ...this.population[i] };
      elite.generation++;
      elite.id = elite.id.replace(/_gen\d+_/, `_gen${elite.generation}_`);
      nextGen.push(elite);
    }
    
    // Tournament selection and breeding
    while (nextGen.length < this.population.length) {
      const parent1 = this.tournamentSelect(config.tournamentSize);
      const parent2 = this.tournamentSelect(config.tournamentSize);
      
      let child;
      if (Math.random() < config.crossoverRate) {
        child = this.crossover(parent1, parent2);
      } else {
        child = { ...parent1 };
      }
      
      if (Math.random() < config.mutationRate) {
        child.parameters = this.mutateParameters(child.parameters);
      }
      
      child.generation++;
      child.id = `${child.baseStrategy}_gen${child.generation}_${nextGen.length}`;
      child.fitness = 0;
      child.trades = 0;
      child.pnl = 0;
      
      nextGen.push(child);
    }
    
    this.population = nextGen;
  }
  
  tournamentSelect(tournamentSize) {
    let best = null;
    
    for (let i = 0; i < tournamentSize; i++) {
      const idx = Math.floor(Math.random() * this.population.length);
      const candidate = this.population[idx];
      
      if (!best || candidate.fitness > best.fitness) {
        best = candidate;
      }
    }
    
    return best;
  }
  
  crossover(parent1, parent2) {
    const child = {
      baseStrategy: parent1.baseStrategy,
      parameters: {},
      generation: Math.max(parent1.generation, parent2.generation),
      fitness: 0,
      trades: 0,
      pnl: 0
    };
    
    // Uniform crossover
    for (const key in parent1.parameters) {
      if (Math.random() < 0.5) {
        child.parameters[key] = parent1.parameters[key];
      } else {
        child.parameters[key] = parent2.parameters[key] || parent1.parameters[key];
      }
    }
    
    return child;
  }
  
  reportGenerationStats() {
    const stats = {
      generation: this.currentGeneration,
      avgFitness: this.population.reduce((sum, s) => sum + s.fitness, 0) / this.population.length,
      bestFitness: this.population[0].fitness,
      worstFitness: this.population[this.population.length - 1].fitness,
      avgTrades: this.population.reduce((sum, s) => sum + s.trades, 0) / this.population.length,
      totalPnL: this.population.reduce((sum, s) => sum + s.pnl, 0)
    };
    
    console.log(`[EvolutionBridge] Generation ${stats.generation} stats:`, stats);
    this.emit('generationComplete', stats);
  }
  
  shouldStopEarly() {
    if (this.currentGeneration < 5) return false;
    
    // Check if fitness has plateaued
    const recentBest = this.bestStrategies.map(s => s.fitness);
    const variance = this.calculateVariance(recentBest);
    
    return variance < 0.001; // Very little improvement
  }
  
  calculateVariance(values) {
    const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
    const squaredDiffs = values.map(v => Math.pow(v - mean, 2));
    return squaredDiffs.reduce((sum, v) => sum + v, 0) / values.length;
  }
  
  async promoteBestStrategies() {
    console.log('[EvolutionBridge] Promoting best strategies...');
    
    const toPromote = this.population
      .filter(s => s.fitness >= this.config.minFitness)
      .slice(0, 5); // Top 5
    
    for (const strategy of toPromote) {
      try {
        // Register evolved strategy
        const evolved = this.createTempStrategy(strategy);
        evolved.name = `${strategy.baseStrategy}_evolved_${Date.now()}`;
        evolved.status = 'promoted';
        
        // Add to strategy manager
        this.strategyManager.registerStrategy(evolved.name, evolved.instance);
        
        console.log(`[EvolutionBridge] Promoted ${evolved.name} with fitness ${strategy.fitness.toFixed(3)}`);
        
        // Record promotion
        if (this.performanceRecorder) {
          this.performanceRecorder.recordDecision({
            symbol: 'EVOLUTION',
            strategy_id: 'evolution_bridge',
            side: 'promote',
            confidence: strategy.fitness,
            meta: {
              promoted_strategy: evolved.name,
              base_strategy: strategy.baseStrategy,
              generation: strategy.generation,
              parameters: strategy.parameters,
              test_results: this.testResults.get(strategy.id)
            }
          });
        }
        
      } catch (error) {
        console.error(`[EvolutionBridge] Error promoting ${strategy.id}:`, error);
      }
    }
    
    this.emit('evolutionComplete', {
      generations: this.currentGeneration + 1,
      promoted: toPromote.length,
      bestFitness: toPromote[0]?.fitness || 0
    });
  }
  
  getEvolutionStatus() {
    return {
      isRunning: this.isRunning,
      currentGeneration: this.currentGeneration,
      populationSize: this.population.length,
      bestFitness: this.population[0]?.fitness || 0,
      avgFitness: this.population.reduce((sum, s) => sum + s.fitness, 0) / this.population.length || 0,
      bestStrategies: this.bestStrategies.slice(0, 5).map(s => ({
        id: s.id,
        fitness: s.fitness,
        trades: s.trades,
        pnl: s.pnl
      }))
    };
  }
}

module.exports = EvolutionBridge;
