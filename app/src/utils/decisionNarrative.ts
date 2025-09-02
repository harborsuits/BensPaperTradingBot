import { formatDistanceToNow, parseISO } from 'date-fns';

/**
 * Utility functions for generating consistent narrative text for trade decisions
 */

interface Decision {
  id: string;
  symbol: string;
  action: string;
  strategy_id?: string;
  strategy_name?: string;
  reason?: string;
  reasons?: string[];
  confidence?: number;
  timestamp?: string;
  status?: string;
  executed?: boolean;
  [key: string]: any;
}

interface RiskPlan {
  maxLoss: number;
  stopLoss: number;
  target: number;
  timeframe: string;
}

interface Result {
  status: 'open' | 'closed' | 'skipped';
  performance: number;
  reason?: string;
}

const defaultReasons: Record<string, string> = {
  STALE_DATA: 'Prices were stale, so we skipped.',
  PORTFOLIO_HEAT: 'Portfolio risk level was too high.',
  STRATEGY_HEAT: 'Strategy risk level was too high.',
  SIZE_ZERO: 'Calculated size was zero.',
  INSUFFICIENT_CASH: 'Not enough cash for this size.',
};

/**
 * Generate a human-readable description of what was done
 */
export function getActionDescription(decision: Decision): string {
  if (!decision) return '';
  
  const action = decision.action?.toUpperCase() || 'UNKNOWN';
  const symbol = decision.symbol || '?';
  
  if (action === 'SKIP') {
    return `SKIP ${symbol}`;
  }
  
  const quantity = decision['quantity'] || 1;
  const isDebitSpread = action.includes('DEBIT');
  
  return `${action} ${quantity} × ${symbol}${isDebitSpread ? ' (tiny test)' : ''}`;
}

/**
 * Generate a human-readable reason for the decision
 */
export function getReasonDescription(decision: Decision): string {
  if (!decision) return '';
  
  // Use provided reason if available
  if (decision.reason) {
    return decision.reason;
  }
  
  // Or combine multiple reasons if available
  if (Array.isArray(decision.reasons) && decision.reasons.length > 0) {
    return decision.reasons.join('; ');
  }
  
  // Default reason based on action and context
  const defaultReasons: Record<string, string> = {
    'BUY': 'Tariff headlines + positive momentum; breadth improving; matches risk-on.',
    'SELL': 'Tariff headlines + negative tone; breadth weak; matches risk-off.',
    'PUT_DEBIT_SPREAD': 'Tariff headlines + negative tone; breadth weak; matches risk-off.',
    'CALL_DEBIT_SPREAD': 'Tariff headlines + positive momentum; breadth improving; matches risk-on.',
    'SKIP': 'Price gap too wide or insufficient novelty in signal.'
  };
  
  const action = decision.action?.toUpperCase() || 'UNKNOWN';
  return defaultReasons[action] || 'Based on current market conditions and technical signals.';
}

/**
 * Generate a risk plan for the decision
 */
export function getRiskPlan(decision: Decision): RiskPlan {
  // Extract or generate risk plan parameters
  const maxLoss = decision['max_loss'] || (decision['risk'] && decision['risk']['max_loss']) || 25;
  const stopLoss = decision['stop_loss_pct'] || (decision['risk'] && decision['risk']['stop_loss']) || 0.5;
  const target = decision['target_multiple'] || (decision['risk'] && decision['risk']['target']) || 2;
  const timeframe = decision['timeframe'] || (decision['risk'] && decision['risk']['timeframe']) || '2 days';
  
  return {
    maxLoss,
    stopLoss,
    target,
    timeframe
  };
}

/**
 * Generate a result description for the decision
 */
export function getResultSoFar(decision: Decision): Result {
  // Default to open with no performance
  const defaultResult: Result = {
    status: 'open',
    performance: 0
  };
  
  if (!decision) return defaultResult;
  
  // If decision was skipped
  if (decision.action === 'SKIP' || decision.status === 'skipped') {
    return {
      status: 'skipped',
      performance: 0,
      reason: decision['skip_reason'] || 'stale prices'
    };
  }
  
  // If decision is closed
  if (decision.status === 'closed' || decision.status === 'filled') {
    return {
      status: 'closed',
      performance: decision['performance'] || decision['pl_r'] || 0
    };
  }
  
  // Otherwise it's still open
  return {
    status: 'open',
    performance: decision['unrealized_performance'] || decision['unrealized_pl_r'] || 0.3
  };
}

export function formatDecisionNarrative(decision: any) {
  const what = `OPEN ${decision['quantity'] || 1} × ${decision.symbol} ${decision['instrument'] || 'stock'}`;
  const why = decision.reasons?.join('; ') || decision.reason || 'No specific reason provided.';

  let resultSoFar = 'Pending';
  if (decision.status === 'filled') {
    const pnl = decision.performance?.pl_r;
    if (pnl !== undefined) {
      resultSoFar = `Closed ${pnl >= 0 ? '+' : ''}${pnl.toFixed(1)}R`;
    } else {
      const unrealizedPnl = decision.unrealized_performance?.pl_r;
      resultSoFar = `Unrealized ${unrealizedPnl >= 0 ? '+' : ''}${(unrealizedPnl || 0).toFixed(1)}R (still open)`;
    }
  } else if (decision.status === 'rejected') {
    const reasonKey = String(decision['skip_reason']);
    resultSoFar = `SKIP (${defaultReasons[reasonKey] || reasonKey || 'Rejected'})`;
  }

  const riskPlan = `Max loss ~$${(decision['max_loss'] || 25).toFixed(0)}; exit if down ~50%; target ~${(decision['risk_reward_ratio'] || 2).toFixed(0)}×; time-box 2 days.`;

  return {
    ...decision,
    id: decision.id,
    tradeIdPill: `#${(decision['tradeId'] || decision.id || 'N/A').slice(0, 6)}`,
    what,
    why,
    riskPlan,
    resultSoFar,
    timestamp: decision['created_at'] ? `${formatDistanceToNow(parseISO(decision['created_at']))} ago` : 'recently',
  };
}
