import { evoTesterApi } from './api';
import segregatedCapitalService from './segregatedCapitalService';

export interface StrategyCandidate {
  id: string;
  name: string;
  fitness: number;
  generation: number;
  experimentId: string;
  createdAt: Date;
  lastUpdated: Date;
  performance: {
    totalTrades: number;
    winRate: number;
    profitFactor: number;
    maxDrawdown: number;
    sharpeRatio: number;
  };
  metadata: {
    riskLevel: 'low' | 'medium' | 'high';
    marketConditions: string[];
    strategyType: string;
    parameters: Record<string, any>;
  };
}

export interface PromotionCriteria {
  minGenerations: number;
  minFitness: number;
  minWinRate: number;
  maxDrawdown: number;
  minTrades: number;
  consistencyScore: number;
  validationPeriod: number; // days
}

export interface PromotionPipeline {
  id: string;
  name: string;
  criteria: PromotionCriteria;
  active: boolean;
  candidates: StrategyCandidate[];
  promoted: StrategyCandidate[];
  rejected: StrategyCandidate[];
  createdAt: Date;
}

export interface ValidationResult {
  candidateId: string;
  passed: boolean;
  score: number;
  validationPeriod: number;
  performance: {
    pnl: number;
    winRate: number;
    drawdown: number;
  };
  feedback: string[];
  validatedAt: Date;
}

class PromotionService {
  private pipelines: Map<string, PromotionPipeline> = new Map();
  private candidates: Map<string, StrategyCandidate> = new Map();
  private validationResults: Map<string, ValidationResult> = new Map();

  constructor() {
    this.initializeDefaultPipelines();
  }

  private initializeDefaultPipelines() {
    // Conservative Promotion Pipeline
    this.createPipeline({
      id: 'conservative_promotion',
      name: 'Conservative Promotion',
      criteria: {
        minGenerations: 10,
        minFitness: 2.0,
        minWinRate: 0.55,
        maxDrawdown: 0.15,
        minTrades: 100,
        consistencyScore: 0.7,
        validationPeriod: 7
      },
      active: true,
      candidates: [],
      promoted: [],
      rejected: [],
      createdAt: new Date()
    });

    // Aggressive Promotion Pipeline
    this.createPipeline({
      id: 'aggressive_promotion',
      name: 'Aggressive Promotion',
      criteria: {
        minGenerations: 5,
        minFitness: 1.5,
        minWinRate: 0.60,
        maxDrawdown: 0.20,
        minTrades: 50,
        consistencyScore: 0.6,
        validationPeriod: 3
      },
      active: true,
      candidates: [],
      promoted: [],
      rejected: [],
      createdAt: new Date()
    });

    // High-Frequency Promotion Pipeline
    this.createPipeline({
      id: 'high_freq_promotion',
      name: 'High-Frequency Promotion',
      criteria: {
        minGenerations: 3,
        minFitness: 1.2,
        minWinRate: 0.65,
        maxDrawdown: 0.10,
        minTrades: 200,
        consistencyScore: 0.8,
        validationPeriod: 1
      },
      active: false,
      candidates: [],
      promoted: [],
      rejected: [],
      createdAt: new Date()
    });
  }

  createPipeline(pipeline: PromotionPipeline) {
    this.pipelines.set(pipeline.id, pipeline);
  }

  addCandidate(candidate: StrategyCandidate) {
    this.candidates.set(candidate.id, candidate);

    // Evaluate against all active pipelines
    for (const [pipelineId, pipeline] of this.pipelines) {
      if (pipeline.active && this.evaluateCandidate(candidate, pipeline.criteria)) {
        pipeline.candidates.push(candidate);
        console.log(`Candidate ${candidate.name} added to pipeline ${pipeline.name}`);
      }
    }
  }

  private evaluateCandidate(candidate: StrategyCandidate, criteria: PromotionCriteria): boolean {
    // Check basic criteria
    if (candidate.generation < criteria.minGenerations) return false;
    if (candidate.fitness < criteria.minFitness) return false;
    if (candidate.performance.winRate < criteria.minWinRate) return false;
    if (candidate.performance.maxDrawdown > criteria.maxDrawdown) return false;
    if (candidate.performance.totalTrades < criteria.minTrades) return false;

    // Calculate consistency score (simplified)
    const consistencyScore = this.calculateConsistencyScore(candidate);
    if (consistencyScore < criteria.consistencyScore) return false;

    return true;
  }

  private calculateConsistencyScore(candidate: StrategyCandidate): number {
    // Simplified consistency calculation based on win rate stability
    // In a real implementation, this would analyze performance across different market conditions
    const winRate = candidate.performance.winRate;
    const profitFactor = candidate.performance.profitFactor;
    const sharpeRatio = candidate.performance.sharpeRatio;

    // Weighted average of key metrics
    return (winRate * 0.4) + (Math.min(profitFactor / 2, 1) * 0.3) + (Math.min(sharpeRatio / 3, 1) * 0.3);
  }

  async promoteCandidate(candidateId: string, pipelineId: string): Promise<boolean> {
    const candidate = this.candidates.get(candidateId);
    const pipeline = this.pipelines.get(pipelineId);

    if (!candidate || !pipeline) {
      throw new Error('Candidate or pipeline not found');
    }

    try {
      // Start validation process
      const validationResult = await this.validateCandidate(candidate, pipeline.criteria);

      if (validationResult.passed) {
        // Move to promoted list
        pipeline.promoted.push(candidate);
        pipeline.candidates = pipeline.candidates.filter(c => c.id !== candidateId);

        // Deploy to main competition
        await this.deployToCompetition(candidate);

        console.log(`Strategy ${candidate.name} promoted to main competition`);
        return true;
      } else {
        // Move to rejected list
        pipeline.rejected.push(candidate);
        pipeline.candidates = pipeline.candidates.filter(c => c.id !== candidateId);

        console.log(`Strategy ${candidate.name} rejected during validation`);
        return false;
      }
    } catch (error) {
      console.error(`Failed to promote candidate ${candidateId}:`, error);
      return false;
    }
  }

  private async validateCandidate(
    candidate: StrategyCandidate,
    criteria: PromotionCriteria
  ): Promise<ValidationResult> {
    // Simulate validation process
    // In a real implementation, this would run the strategy in a validation environment
    const validationDays = criteria.validationPeriod;

    // Simulate validation results
    const pnl = Math.random() * 1000 - 200; // -200 to +800
    const winRate = 0.5 + Math.random() * 0.3; // 0.5 to 0.8
    const drawdown = Math.random() * 0.15; // 0 to 0.15

    const passed = pnl > 0 && winRate > criteria.minWinRate && drawdown < criteria.maxDrawdown;
    const score = (pnl / 1000 * 0.4) + (winRate * 0.4) + ((1 - drawdown / 0.15) * 0.2);

    const result: ValidationResult = {
      candidateId: candidate.id,
      passed,
      score,
      validationPeriod: validationDays,
      performance: {
        pnl,
        winRate,
        drawdown
      },
      feedback: this.generateValidationFeedback(passed, pnl, winRate, drawdown),
      validatedAt: new Date()
    };

    this.validationResults.set(candidate.id, result);
    return result;
  }

  private generateValidationFeedback(
    passed: boolean,
    pnl: number,
    winRate: number,
    drawdown: number
  ): string[] {
    const feedback: string[] = [];

    if (passed) {
      feedback.push('✅ Passed all validation criteria');
      if (pnl > 500) feedback.push('💰 Strong profit performance');
      if (winRate > 0.7) feedback.push('🎯 High win rate maintained');
      if (drawdown < 0.05) feedback.push('🛡️ Excellent risk management');
    } else {
      feedback.push('❌ Failed validation criteria');
      if (pnl <= 0) feedback.push('📉 Negative P&L during validation');
      if (winRate < 0.55) feedback.push('⚠️ Win rate below threshold');
      if (drawdown > 0.12) feedback.push('🚨 Excessive drawdown');
    }

    return feedback;
  }

  private async deployToCompetition(candidate: StrategyCandidate): Promise<void> {
    try {
      // Allocate capital from competition pool
      const allocation = segregatedCapitalService.allocateCapital(
        'competition_pool',
        candidate.id,
        5000, // Standard allocation for new strategies
        candidate.metadata.riskLevel
      );

      // Register strategy with main trading system
      await evoTesterApi.promoteStrategy({
        id: candidate.id,
        name: candidate.name,
        fitness: candidate.fitness,
        parameters: candidate.metadata.parameters,
        allocationId: allocation.id,
        deployedAt: new Date().toISOString()
      });

      console.log(`Strategy ${candidate.name} deployed with allocation ${allocation.id}`);
    } catch (error) {
      console.error(`Failed to deploy strategy ${candidate.id} to competition:`, error);
      throw error;
    }
  }

  // Public getters
  getPipelines(): PromotionPipeline[] {
    return Array.from(this.pipelines.values());
  }

  getCandidates(): StrategyCandidate[] {
    return Array.from(this.candidates.values());
  }

  getValidationResult(candidateId: string): ValidationResult | undefined {
    return this.validationResults.get(candidateId);
  }

  getPipeline(pipelineId: string): PromotionPipeline | undefined {
    return this.pipelines.get(pipelineId);
  }

  // Statistics
  getPromotionStats() {
    const pipelines = this.getPipelines();
    const totalCandidates = pipelines.reduce((sum, p) => sum + p.candidates.length, 0);
    const totalPromoted = pipelines.reduce((sum, p) => sum + p.promoted.length, 0);
    const totalRejected = pipelines.reduce((sum, p) => sum + p.rejected.length, 0);

    const successRate = totalCandidates > 0 ? totalPromoted / (totalPromoted + totalRejected) : 0;

    return {
      totalCandidates,
      totalPromoted,
      totalRejected,
      successRate,
      activePipelines: pipelines.filter(p => p.active).length
    };
  }

  // Auto-promotion check (to be called periodically)
  async checkForPromotions() {
    for (const [pipelineId, pipeline] of this.pipelines) {
      if (!pipeline.active) continue;

      for (const candidate of pipeline.candidates) {
        // Check if candidate has been in pipeline long enough
        const daysInPipeline = (Date.now() - candidate.createdAt.getTime()) / (1000 * 60 * 60 * 24);

        if (daysInPipeline >= pipeline.criteria.validationPeriod) {
          try {
            await this.promoteCandidate(candidate.id, pipelineId);
          } catch (error) {
            console.error(`Auto-promotion failed for candidate ${candidate.id}:`, error);
          }
        }
      }
    }
  }

  // Cleanup
  destroy() {
    this.pipelines.clear();
    this.candidates.clear();
    this.validationResults.clear();
  }
}

// Export singleton instance
export const promotionService = new PromotionService();
export default promotionService;
