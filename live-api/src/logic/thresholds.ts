/**
 * Dynamic threshold calculations for pipeline stage transitions
 * Adjusts ROUTE threshold based on market regime, volatility, and portfolio health
 */

export interface MarketContext {
  vix: number;
  regime: 'bull' | 'bear' | 'chop' | 'neutral';
  ddPct: number;  // portfolio drawdown percentage
  slippageBps: number;  // expected round-trip slippage in basis points
}

export function routeThreshold({
  base = 9.2,          // start a bit higher than 9.0
  vix,
  regime,
  ddPct,
  slippageBps,
}: MarketContext & { base?: number }) {
  let t = base;

  // More volatility → require stronger edge
  t += Math.max(0, (vix - 18) * 0.05);      // +0.05 per VIX pt > 18

  // Bear/chop regimes are tougher
  if (regime === 'bear') t += 0.4;
  else if (regime === 'chop') t += 0.2;

  // Protect while in drawdown
  t += Math.min(0.8, Math.max(0, ddPct) * 0.05); // +0.05 per 1% DD (cap +0.8)

  // Pricing friction
  t += Math.min(0.3, slippageBps / 50 * 0.1);    // +0.1 per 50bps

  return Math.min(9.9, Math.max(base, t)); // Never go below base, cap at 9.9
}

export function planThreshold({
  base = 7.5,
  vix,
  regime,
}: Pick<MarketContext, 'vix' | 'regime'> & { base?: number }) {
  let t = base;

  // Adjust for market conditions
  if (regime === 'bull') t -= 0.2;
  else if (regime === 'bear') t += 0.3;

  t += Math.max(0, (vix - 20) * 0.02); // Slight adjustment for volatility

  return Math.max(6.5, Math.min(8.5, t));
}

export function gatesThreshold({
  base = 6.0,
  vix,
  regime,
}: Pick<MarketContext, 'vix' | 'regime'> & { base?: number }) {
  let t = base;

  // More permissive in bull markets
  if (regime === 'bull') t -= 0.3;
  else if (regime === 'bear') t += 0.2;

  return Math.max(5.0, Math.min(7.0, t));
}

// Mock market context for now - in production this would come from real data
export function getMockMarketContext(): MarketContext {
  return {
    vix: 18 + Math.random() * 10,  // 18-28 range
    regime: ['bull', 'bear', 'chop', 'neutral'][Math.floor(Math.random() * 4)] as any,
    ddPct: Math.random() * 5,  // 0-5% drawdown
    slippageBps: 20 + Math.random() * 40,  // 20-60 bps
  };
}
