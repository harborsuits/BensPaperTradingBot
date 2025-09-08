/**
 * Single canonical payload for all decision events
 * Used by Brain Flow, Decisions, Storage, and UI components
 */

export type Stage = 'INGEST' | 'CONTEXT' | 'CANDIDATES' | 'GATES' | 'PLAN' | 'ROUTE' | 'MANAGE' | 'LEARN';

export interface Evidence {
  source: 'news' | 'price' | 'econ' | 'alt';
  title: string;
  url?: string;
  quote?: string;
  timestamp?: string;
}

export interface DecisionEvent {
  id: string;
  symbol: string;
  stage: Stage;
  score: number;
  factors: Record<string, number>;     // e.g., { momentum: 2.3, trend: 2.2, volume: 2.0 }
  action?: 'long' | 'short' | 'wait';
  rationale: string;                   // 1–2 sentence layman summary
  evidence: Evidence[];                // breadcrumbs (links/quotes)
  asOf: string;
  routeThreshold?: number;            // dynamic threshold used for this decision
  marketContext?: {
    vix: number;
    regime: string;
    ddPct: number;
    slippageBps: number;
  };
}

// Helper functions
export function createDecisionEvent(partial: Partial<DecisionEvent>): DecisionEvent {
  return {
    id: partial.id || `decision_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    symbol: partial.symbol || 'UNKNOWN',
    stage: partial.stage || 'CONTEXT',
    score: partial.score || 0,
    factors: partial.factors || {},
    action: partial.action,
    rationale: partial.rationale || 'Analysis in progress',
    evidence: partial.evidence || [],
    asOf: partial.asOf || new Date().toISOString(),
    routeThreshold: partial.routeThreshold,
    marketContext: partial.marketContext,
  };
}

export function shouldPublishToWS(prev: DecisionEvent | undefined, current: DecisionEvent): boolean {
  if (!prev) return true; // First event
  if (prev.stage !== current.stage) return true; // Stage change
  if (Math.abs(prev.score - current.score) >= 0.3) return true; // Significant score change
  return false; // No significant change
}
