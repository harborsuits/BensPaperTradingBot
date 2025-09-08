/**
 * Position sizing calculations for ROUTE stage opportunities
 * Provides actionable dollar amounts based on risk management
 */

export interface PositionSizeParams {
  equity: number;          // portfolio equity in dollars
  riskPct?: number;        // risk per trade as % of equity (default 0.5%)
  stopPct: number;         // distance to stop loss as percentage (e.g., 1.2%)
  confidence: number;      // confidence score 0.0-1.0
  maxPositionPct?: number; // maximum position size as % of equity (default 5%)
}

export function sizeDollars({
  equity,
  riskPct = 0.5,              // risk per trade (portfolio %)
  stopPct,
  confidence,
  maxPositionPct = 5,
}: PositionSizeParams): number {
  // Maximum dollars at risk for this trade
  const maxRisk = equity * (riskPct / 100);

  // Position size based on stop distance
  const positionSize = maxRisk / (stopPct / 100);

  // Adjust by confidence (0.6 = base, up to 1.0 for high confidence)
  const confidenceMultiplier = 0.6 + (0.4 * confidence);

  // Apply confidence adjustment
  let sizedPosition = positionSize * confidenceMultiplier;

  // Cap at maximum position percentage
  const maxPositionSize = equity * (maxPositionPct / 100);
  sizedPosition = Math.min(sizedPosition, maxPositionSize);

  return Math.round(sizedPosition);
}

export function sizeShares({
  positionDollars,
  price,
  minShares = 1,
}: {
  positionDollars: number;
  price: number;
  minShares?: number;
}): number {
  const shares = Math.floor(positionDollars / price);
  return Math.max(shares, minShares);
}

export function formatPositionSize(params: PositionSizeParams & { price: number }): string {
  const dollars = sizeDollars(params);
  const shares = sizeShares({ positionDollars: dollars, price: params.price });

  return `$${dollars.toLocaleString()} (${shares} shares)`;
}

// Example usage:
// const size = sizeDollars({
//   equity: 100000,
//   riskPct: 0.5,
//   stopPct: 1.2,
//   confidence: 0.8
// });
// console.log(formatPositionSize({ ...params, price: 150 }));
// Output: "$3,333 (22 shares)"
