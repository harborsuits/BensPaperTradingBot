// Formatters for consistent shorthand display across the UI
// Used by CandidatesQuickList, ActivityTicker, and other components

export function fmtSpread(value: number, isPct?: boolean): string {
  if (isPct) {
    return `${value.toFixed(2)}%`;
  }
  return `${(value / 100).toFixed(2)}%`;
}

export function fmtRV(mult: number): string {
  return `${mult.toFixed(1)}×`;
}

export function fmtTone(tone: string): string {
  switch (tone?.toLowerCase()) {
    case 'positive':
    case 'bullish':
      return 'pos';
    case 'negative':
    case 'bearish':
      return 'neg';
    case 'neutral':
      return 'neu';
    default:
      return tone?.slice(0, 3) || 'unk';
  }
}

export function fmtPass(pass: boolean, reason?: string): string {
  const result = pass ? 'PASS' : 'SKIP';
  return reason ? `${result} (${reason})` : result;
}

export function fmtShortHand(candidate: any): string {
  const symbol = candidate.symbol || 'UNKNOWN';
  const score = candidate.score || 0;
  const confidence = candidate.confidence || 0;
  const rv = candidate.explain?.relativeVolume || 1;
  const spread = candidate.explain?.spreadPct || 0;
  const tone = candidate.explain?.tone || 'unk';

  const scoreStr = `${Math.round(score * 100)}%`;
  const confStr = `${Math.round(confidence * 100)}%`;
  const rvStr = fmtRV(rv);
  const spreadStr = fmtSpread(spread, true);
  const toneStr = fmtTone(tone);

  return `${symbol} • ${scoreStr}/${confStr} • rv=${rvStr} • spr=${spreadStr} • tone=${toneStr}`;
}

export function fmtContextBar(symbol: string, lastDecision?: any, openOrders?: any[]): string {
  const decisionStr = lastDecision ? `last: ${lastDecision.side?.toUpperCase()} ${lastDecision.quantity}` : 'no recent';
  const ordersStr = openOrders?.length ? `${openOrders.length} open` : 'none open';
  return `${symbol} • ${decisionStr} • orders: ${ordersStr}`;
}

export function fmtTimeAgo(timestamp: string): string {
  try {
    const diff = Date.now() - new Date(timestamp).getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (days > 0) return `${days}d ago`;
    if (hours > 0) return `${hours}h ago`;
    if (minutes > 0) return `${minutes}m ago`;
    return 'now';
  } catch {
    return 'recent';
  }
}
