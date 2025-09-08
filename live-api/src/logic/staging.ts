/**
 * Pipeline stage management with hysteresis to prevent flip-flopping
 * Implements minimum hold times and threshold bands for stable transitions
 */

export type PipelineStage = 'CONTEXT' | 'CANDIDATES' | 'GATES' | 'PLAN' | 'ROUTE';

export interface StageTransition {
  from: PipelineStage;
  to: PipelineStage;
  score: number;
  timestamp: number;
  reason: string;
}

export class StageManager {
  private lastTransitions: Map<string, StageTransition> = new Map();
  private readonly MIN_HOLD_TIME_MS = 2 * 60 * 1000; // 2 minutes minimum

  nextStage(
    symbol: string,
    current: PipelineStage,
    score: number,
    routeThreshold: number
  ): { stage: PipelineStage; shouldTransition: boolean; reason: string } {
    const now = Date.now();
    const lastTransition = this.lastTransitions.get(symbol);

    // Check minimum hold time
    if (lastTransition && (now - lastTransition.timestamp) < this.MIN_HOLD_TIME_MS) {
      return {
        stage: current,
        shouldTransition: false,
        reason: `Holding ${current} (min hold time: ${(this.MIN_HOLD_TIME_MS / 1000).toFixed(0)}s)`
      };
    }

    const upThreshold = routeThreshold;           // promote if >= threshold
    const downThreshold = routeThreshold - 0.6;   // demote if < (threshold - 0.6)

    let newStage = current;
    let reason = `Holding ${current}`;

    // ROUTE transitions
    if (current !== 'ROUTE' && score >= upThreshold) {
      newStage = 'ROUTE';
      reason = `Promoting to ROUTE (score ${score.toFixed(1)} >= ${upThreshold.toFixed(1)})`;
    } else if (current === 'ROUTE' && score < downThreshold) {
      newStage = 'PLAN';
      reason = `Demoting to PLAN (score ${score.toFixed(1)} < ${downThreshold.toFixed(1)})`;
    }
    // PLAN transitions
    else if (current === 'PLAN' && score >= upThreshold) {
      newStage = 'ROUTE';
      reason = `Promoting to ROUTE (score ${score.toFixed(1)} >= ${upThreshold.toFixed(1)})`;
    } else if (current === 'PLAN' && score < routeThreshold - 1.2) {
      newStage = 'GATES';
      reason = `Demoting to GATES (score ${score.toFixed(1)} < ${(routeThreshold - 1.2).toFixed(1)})`;
    }
    // GATES transitions
    else if (current === 'GATES' && score >= routeThreshold - 0.8) {
      newStage = 'PLAN';
      reason = `Promoting to PLAN (score ${score.toFixed(1)} >= ${(routeThreshold - 0.8).toFixed(1)})`;
    } else if (current === 'GATES' && score < routeThreshold - 2.0) {
      newStage = 'CANDIDATES';
      reason = `Demoting to CANDIDATES (score ${score.toFixed(1)} < ${(routeThreshold - 2.0).toFixed(1)})`;
    }

    const shouldTransition = newStage !== current;

    if (shouldTransition) {
      const transition: StageTransition = {
        from: current,
        to: newStage,
        score,
        timestamp: now,
        reason
      };
      this.lastTransitions.set(symbol, transition);
    }

    return { stage: newStage, shouldTransition, reason };
  }

  getLastTransition(symbol: string): StageTransition | undefined {
    return this.lastTransitions.get(symbol);
  }

  getAllTransitions(): StageTransition[] {
    return Array.from(this.lastTransitions.values());
  }

  clearOldTransitions(maxAgeMs: number = 24 * 60 * 60 * 1000) { // 24 hours
    const cutoff = Date.now() - maxAgeMs;
    for (const [symbol, transition] of this.lastTransitions) {
      if (transition.timestamp < cutoff) {
        this.lastTransitions.delete(symbol);
      }
    }
  }
}

// Singleton instance
export const stageManager = new StageManager();

// Clean up old transitions every hour
setInterval(() => stageManager.clearOldTransitions(), 60 * 60 * 1000);
