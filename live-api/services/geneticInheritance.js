/**
 * Genetic Inheritance Service
 * Extracts successful traits from winning bots and breeds new generations
 */

class GeneticInheritanceService {
  constructor(botCompetitionService, evolutionBridge) {
    this.botCompetitionService = botCompetitionService;
    this.evolutionBridge = evolutionBridge;
    
    // Gene pool storage
    this.genePool = new Map(); // symbol -> winning strategies
    this.marketMemory = new Map(); // market condition -> successful genes
  }

  /**
   * Extract genetic traits from winning bots
   */
  extractWinnerGenes(competitionId, topN = 10) {
    const status = this.botCompetitionService.getCompetitionStatus(competitionId);
    if (!status) return [];
    
    const winners = status.leaderboard.slice(0, topN);
    const genes = [];
    
    winners.forEach(winner => {
      const bot = this.botCompetitionService.bots.get(winner.id);
      if (!bot) return;
      
      const gene = {
        strategyType: bot.strategy.type,
        symbol: bot.strategy.symbol,
        generation: bot.strategy.generation,
        fitness: winner.returnPct,
        winRate: winner.winRate,
        capitalGrowth: winner.currentCapital / 50, // Growth multiple
        parameters: this.extractParameters(bot),
        marketConditions: this.captureMarketConditions(),
        metadata: {
          ...bot.strategy.metadata,
          competitionId,
          extractedAt: new Date().toISOString()
        }
      };
      
      genes.push(gene);
      
      // Store in gene pool by symbol
      if (!this.genePool.has(gene.symbol)) {
        this.genePool.set(gene.symbol, []);
      }
      this.genePool.get(gene.symbol).push(gene);
    });
    
    return genes;
  }

  /**
   * Breed new generation from winner genes
   */
  breedNewGeneration(parentGenes, count = 100) {
    const newBots = [];
    
    // Elite cloning (20% are direct copies of winners)
    const eliteCount = Math.floor(count * 0.2);
    for (let i = 0; i < eliteCount; i++) {
      const elite = parentGenes[i % parentGenes.length];
      newBots.push(this.cloneWithMutation(elite, 0.05)); // 5% mutation
    }
    
    // Crossover breeding (60% are bred from two parents)
    const crossoverCount = Math.floor(count * 0.6);
    for (let i = 0; i < crossoverCount; i++) {
      const parent1 = this.tournamentSelect(parentGenes);
      const parent2 = this.tournamentSelect(parentGenes);
      newBots.push(this.crossover(parent1, parent2));
    }
    
    // Wild cards (20% are random for exploration)
    const wildcardCount = count - eliteCount - crossoverCount;
    for (let i = 0; i < wildcardCount; i++) {
      newBots.push(this.generateWildcard(parentGenes));
    }
    
    return newBots;
  }

  /**
   * Crossover two parent strategies
   */
  crossover(parent1, parent2) {
    const child = {
      name: `Gen${Math.max(parent1.generation, parent2.generation) + 1}-X`,
      type: Math.random() > 0.5 ? parent1.strategyType : parent2.strategyType,
      symbol: Math.random() > 0.7 ? parent1.symbol : parent2.symbol, // 70% inherit symbol
      generation: Math.max(parent1.generation, parent2.generation) + 1,
      parameters: this.blendParameters(parent1.parameters, parent2.parameters),
      metadata: {
        parent1Id: parent1.metadata.competitionId,
        parent2Id: parent2.metadata.competitionId,
        breedingMethod: 'crossover',
        inheritedFitness: (parent1.fitness + parent2.fitness) / 2
      }
    };
    
    // Apply mutation
    if (Math.random() < 0.1) { // 10% mutation rate
      child.parameters = this.mutateParameters(child.parameters);
      child.metadata.mutated = true;
    }
    
    return child;
  }

  /**
   * Tournament selection (fitness-based)
   */
  tournamentSelect(population, tournamentSize = 3) {
    let best = null;
    let bestFitness = -Infinity;
    
    for (let i = 0; i < tournamentSize; i++) {
      const contestant = population[Math.floor(Math.random() * population.length)];
      const fitness = this.calculateCompositeFitness(contestant);
      
      if (fitness > bestFitness) {
        best = contestant;
        bestFitness = fitness;
      }
    }
    
    return best;
  }

  /**
   * Calculate composite fitness score
   */
  calculateCompositeFitness(gene) {
    return (
      gene.fitness * 0.4 +           // Return percentage
      gene.winRate * 0.3 +           // Win rate
      Math.log(gene.capitalGrowth) * 0.2 + // Capital growth (log scale)
      (1 / (gene.generation + 1)) * 0.1   // Favor newer generations slightly
    );
  }

  /**
   * Extract strategy parameters based on type
   */
  extractParameters(bot) {
    const baseParams = {
      entryThreshold: 0.7 + (Math.random() - 0.5) * 0.2,
      exitThreshold: 0.3 + (Math.random() - 0.5) * 0.2,
      stopLoss: 0.02 + Math.random() * 0.03,
      takeProfit: 0.03 + Math.random() * 0.07,
      positionSize: 0.25 // For $50 accounts
    };
    
    // Add strategy-specific parameters
    switch (bot.strategy.type) {
      case 'rsi_reversion':
        return {
          ...baseParams,
          rsiPeriod: 14,
          rsiBuyThreshold: 30,
          rsiSellThreshold: 70
        };
      
      case 'volatility':
        return {
          ...baseParams,
          volPeriod: 20,
          volThreshold: 1.5,
          volMultiplier: 2.0
        };
      
      case 'mean_reversion':
        return {
          ...baseParams,
          maPeriod: 20,
          stdDevMultiplier: 2.0,
          meanReversionSpeed: 0.1
        };
      
      case 'ma_crossover':
        return {
          ...baseParams,
          fastMA: 9,
          slowMA: 21,
          signalStrength: 0.02
        };
      
      default:
        return baseParams;
    }
  }

  /**
   * Blend parameters from two parents
   */
  blendParameters(params1, params2) {
    const blended = {};
    
    // Average numeric parameters
    for (const key in params1) {
      if (typeof params1[key] === 'number' && params2[key] !== undefined) {
        blended[key] = (params1[key] + params2[key]) / 2;
      } else {
        blended[key] = Math.random() > 0.5 ? params1[key] : params2[key];
      }
    }
    
    return blended;
  }

  /**
   * Mutate parameters slightly
   */
  mutateParameters(params) {
    const mutated = { ...params };
    
    for (const key in mutated) {
      if (typeof mutated[key] === 'number' && Math.random() < 0.3) { // 30% chance per param
        mutated[key] *= (0.8 + Math.random() * 0.4); // ±20% change
      }
    }
    
    return mutated;
  }

  /**
   * Clone with slight mutation
   */
  cloneWithMutation(gene, mutationRate) {
    const clone = {
      ...gene,
      generation: gene.generation + 1,
      parameters: { ...gene.parameters },
      metadata: {
        ...gene.metadata,
        clonedFrom: gene.metadata.competitionId,
        clonedAt: new Date().toISOString()
      }
    };
    
    if (Math.random() < mutationRate) {
      clone.parameters = this.mutateParameters(clone.parameters);
      clone.metadata.mutated = true;
    }
    
    return clone;
  }

  /**
   * Generate wildcard for exploration
   */
  generateWildcard(genePool) {
    const symbols = [...new Set(genePool.map(g => g.symbol))];
    const types = [...new Set(genePool.map(g => g.strategyType))];
    
    return {
      name: `Wildcard-${Date.now()}`,
      type: types[Math.floor(Math.random() * types.length)],
      symbol: symbols[Math.floor(Math.random() * symbols.length)],
      generation: 1,
      parameters: this.extractParameters({ strategy: { type: types[0] } }),
      metadata: {
        wildcard: true,
        inspired_by: genePool.length
      }
    };
  }

  /**
   * Capture current market conditions for memory
   */
  captureMarketConditions() {
    // This would integrate with market indicators
    return {
      regime: 'neutral', // Would get from indicators
      volatility: 'medium',
      trend: 'sideways',
      vix: 17,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Get best genes for current market conditions
   */
  getBestGenesForConditions(marketConditions) {
    const key = `${marketConditions.regime}-${marketConditions.volatility}`;
    return this.marketMemory.get(key) || [];
  }

  /**
   * Store successful genes with market context
   */
  storeMarketMemory(genes, marketConditions) {
    const key = `${marketConditions.regime}-${marketConditions.volatility}`;
    
    if (!this.marketMemory.has(key)) {
      this.marketMemory.set(key, []);
    }
    
    const memory = this.marketMemory.get(key);
    genes.forEach(gene => {
      if (gene.fitness > 0.1) { // Only store profitable genes
        memory.push({
          ...gene,
          marketConditions,
          storedAt: new Date().toISOString()
        });
      }
    });
    
    // Keep only top 50 per market condition
    memory.sort((a, b) => b.fitness - a.fitness);
    if (memory.length > 50) {
      memory.splice(50);
    }
  }
}

module.exports = { GeneticInheritanceService };
