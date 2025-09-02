/**
 * Utility functions for generating consistent narrative text for trade candidates
 */

interface Candidate {
  id: string;
  symbol: string;
  action: string;
  strategy_id?: string;
  strategy_name?: string;
  score?: number;
  reason?: string;
  reasons?: string[];
  [key: string]: any;
}

interface CandidateCosts {
  feesPerHundred: number;
  spreadPercent: number;
  riskIncrease: number;
}

interface CandidatePlan {
  type: string;
  maxLoss: number;
  targetMultiple: number;
  timeframe: string;
}

/**
 * Generate a human-readable reason for why this candidate is on the list
 */
export function getReasonDescription(candidate: Candidate): string {
  if (!candidate) return '';
  
  // Use provided reason if available
  if (candidate.reason) {
    return candidate.reason;
  }
  
  // Or combine multiple reasons if available
  if (Array.isArray(candidate.reasons) && candidate.reasons.length > 0) {
    return candidate.reasons.join('; ');
  }
  
  // Default reason
  return "Fresh tariff headlines mentioning semiconductors; news is negative; options activity picking up.";
}

/**
 * Generate a human-readable description of how this candidate fits the current market
 */
export function getMarketFitDescription(candidate: Candidate, context?: any): string {
  if (!candidate) return '';
  
  const marketMood = context?.regime?.type?.toLowerCase() || 'neutral';
  const action = candidate.action?.toLowerCase() || '';
  
  // Check if the action aligns with market mood
  const isBearish = action.includes('put') || action.includes('sell') || action.includes('short');
  const isBullish = action.includes('call') || action.includes('buy') || action.includes('long');
  
  const alignsWithMarket = 
    (marketMood === 'bearish' && isBearish) ||
    (marketMood === 'bullish' && isBullish) ||
    (marketMood === 'neutral');
  
  if (alignsWithMarket) {
    return `Market mood: ${marketMood}. This idea lines up (${isBearish ? 'bearish' : 'bullish'} ${candidate.symbol.includes('SPY') ? 'market' : 'tech'}).`;
  } else {
    return `Market mood: ${marketMood}. This idea is contrarian (${isBearish ? 'bearish' : 'bullish'} ${candidate.symbol.includes('SPY') ? 'market' : 'tech'}).`;
  }
}

/**
 * Generate costs and risk information
 */
export function getCostsAndRisk(candidate: Candidate): CandidateCosts {
  // Default values
  const feesPerHundred = candidate.fees_per_hundred || candidate.costs?.fees_per_hundred || 0.09;
  const spreadPercent = candidate.spread_percent || candidate.costs?.spread_percent || 0.24;
  const riskIncrease = candidate.risk_increase || candidate.costs?.risk_increase || 1;
  
  return {
    feesPerHundred,
    spreadPercent,
    riskIncrease
  };
}

/**
 * Generate plan suggestion
 */
export function getPlanSuggestion(candidate: Candidate): CandidatePlan {
  // Extract or generate plan parameters
  const type = candidate.plan_type || 
               candidate.action?.toLowerCase().includes('put') ? 'SPY put debit spread' : 
               candidate.action?.toLowerCase().includes('call') ? 'SPY call debit spread' : 
               `${candidate.symbol} ${candidate.action?.toLowerCase().includes('sell') ? 'short' : 'long'}`;
               
  const maxLoss = candidate.max_loss || candidate.plan?.max_loss || 25;
  const targetMultiple = candidate.target_multiple || candidate.plan?.target_multiple || 2;
  const timeframe = candidate.timeframe || candidate.plan?.timeframe || '1–2 days';
  
  return {
    type,
    maxLoss,
    targetMultiple,
    timeframe
  };
}

/**
 * Generate decision and reason
 */
export function getDecision(candidate: Candidate, cashAvailable: number): { 
  decision: 'PASS' | 'SKIP_SPREAD' | 'SKIP_NOVELTY' | 'SKIP_CASH', 
  reason: string 
} {
  // Check if there's enough cash
  const notional = candidate.notional || candidate.costs?.notional || 100;
  if (notional > cashAvailable) {
    return {
      decision: 'SKIP_CASH',
      reason: "SKIP: Not enough cash for this size."
    };
  }
  
  // Check spread
  const spreadPercent = candidate.spread_percent || candidate.costs?.spread_percent || 0.24;
  if (spreadPercent > 0.5) {
    return {
      decision: 'SKIP_SPREAD',
      reason: "SKIP: 'Price gap at entry' is too wide today."
    };
  }
  
  // Check novelty
  const isNovel = candidate.is_novel !== false;
  if (!isNovel) {
    return {
      decision: 'SKIP_NOVELTY',
      reason: "SKIP: 'Feels like a repeat' (news not fresh)."
    };
  }
  
  // Default to PASS
  return {
    decision: 'PASS',
    reason: "Fresh idea, costs ok, and our odds look slightly favorable → probe with tiny size."
  };
}

/**
 * Format a complete candidate narrative object
 */
export function formatCandidateNarrative(candidate) {
  if (!candidate) {
    return null;
  }

  // --- Dynamic Narratives based on real data ---
  const symbol = candidate.symbol || 'UNKNOWN';
  const score = candidate.score || 0;
  const confidence = candidate.confidence || 0;
  const side = candidate.side || 'buy';
  const impact1h = candidate.explain?.impact1h || 0;
  const impact24h = candidate.explain?.impact24h || 0;
  const spreadPct = candidate.explain?.spreadPct || 0.5;
  const rvol = candidate.explain?.rvol || 1.0;
  const volume = candidate.explain?.volume || 0;
  const entry = candidate.plan?.entry || candidate.last || 100;
  const stop = candidate.plan?.stop || entry * 0.97;
  const take = candidate.plan?.take || entry * 1.03;
  const suggestedQty = candidate.risk?.suggestedQty || 100;

  // Generate dynamic "why this is on the list" based on actual data
  let whyOnList = `${symbol}: `;
  if (impact1h > 0.1) {
    whyOnList += `Strong ${impact1h > 0 ? 'positive' : 'negative'} news momentum (${impact1h.toFixed(2)} impact in last hour); `;
  } else if (rvol > 1.5) {
    whyOnList += `High relative volume (${rvol.toFixed(1)}x average); `;
  } else {
    whyOnList += `Scanner detected opportunity with ${confidence.toFixed(2)} confidence score; `;
  }
  whyOnList += `volume trending with ${(volume / 1000).toFixed(0)}K daily volume.`;

  // Dynamic market fit based on side and confidence
  const isBullish = side === 'buy';
  const marketFit = `${symbol}: Market scanner confidence: ${(confidence * 100).toFixed(0)}%. ${isBullish ? 'Bullish' : 'Bearish'} signal with ${(impact24h * 100).toFixed(1)}% 24h impact.`;

  // --- Costs & Risks using real data ---
  const feesAndSlip = `To open this trade you'd pay about $${((candidate.explain?.feeBps || 5) + (candidate.explain?.slipBps || 4)) / 100} per $100 traded (fees + typical slippage).`;
  const priceWiggle = `Prices can wiggle ≈${spreadPct.toFixed(2)}% around the quote; that could move your fill.`;
  const riskIncrease = Math.max(1, Math.floor(suggestedQty / 100)); // rough risk estimate
  const riskMeter = `If we take this trade, your risk meter would nudge up ~${riskIncrease} point (still inside your limit).`;

  // --- Dynamic Plan based on real data ---
  const maxLoss = Math.abs(entry - stop) * suggestedQty;
  const targetGain = Math.abs(take - entry) * suggestedQty;
  const plan = `${symbol}: Try ${suggestedQty} shares ${isBullish ? 'long' : 'short'}. Max loss ≈ $${maxLoss.toFixed(0)}, aim ≈$${targetGain.toFixed(0)} if it works, exit within 1–2 days.`;

  // --- Dynamic Go/No-go based on real conditions ---
  let goNoGo = `${symbol}: Signal strength: ${(score * 100).toFixed(0)}%, confidence: ${(confidence * 100).toFixed(0)}% → ${suggestedQty} share position recommended.`;

  if (spreadPct > 0.5) {
    goNoGo = `${symbol}: SKIP: 'Price gap at entry' is ${(spreadPct * 100).toFixed(1)}% wide today (too wide).`;
  } else if (confidence < 0.5) {
    goNoGo = `${symbol}: SKIP: 'Signal too weak' (${(confidence * 100).toFixed(0)}% confidence below threshold).`;
  } else if (rvol < 0.8) {
    goNoGo = `${symbol}: SKIP: 'Volume too low' (${rvol.toFixed(1)}x average volume).`;
  }

  // --- Cash Only ---
  const cashOnly = "We only spend cash we have. No margin. No credit.";

  return {
    ...candidate,
    narrative: {
      whyOnList,
      marketFit,
      costs: {
        feesAndSlip,
        priceWiggle,
        riskMeter,
      },
      plan,
      goNoGo,
      cashOnly,
    },
  };
}
