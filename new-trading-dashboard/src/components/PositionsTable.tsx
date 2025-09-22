import React from 'react';
import ProvenanceChip from './ProvenanceChip';
import StalePill from './StalePill';
import { useSyncedPositions } from '@/hooks/useSyncedData';

type Position = {
  symbol: string;
  qty: number;
  avg_price: number;
  market_price?: number | null;
  unrealized_pnl?: number | null;
  quote?: { stale?: boolean; cache_age_ms?: number; ttl_ms?: number };
};

export default function PositionsTable() {
  const { data: positions, isLoading } = useSyncedPositions();
  
  if (isLoading || !positions) return null;
  
  const rows = positions || [];
  const meta = { source: 'tradier', provider: 'tradier' };

  return (
    <div className="p-4 rounded-xl border bg-card text-foreground border-border">
      <div className="flex items-center justify-between">
        <h3 className="m-0">Positions</h3>
        <ProvenanceChip source={meta.source} provider={meta.provider} asof_ts={meta.asof_ts} latency_ms={meta.latency_ms} />
      </div>

      <table className="w-full mt-3 text-sm">
        <thead>
          <tr>
            <th align="left">Symbol</th>
            <th align="right">Qty</th>
            <th align="right">Avg</th>
            <th align="right">Mkt</th>
            <th align="right">Unrlzd</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((p) => {
            const q = p.quote || {};
            const stale = Boolean(q.stale);
            return (
              <tr key={p.symbol}>
                <td>{p.symbol}</td>
                <td align="right">{p.qty}</td>
                <td align="right">{Number(p.avg_price).toFixed(2)}</td>
                <td align="right" style={{ opacity: stale ? 0.6 : 1 }}>
                  {p.market_price != null ? Number(p.market_price).toFixed(2) : '—'}
                  <StalePill stale={stale} ageMs={q.cache_age_ms} ttlMs={q.ttl_ms} />
                </td>
                <td align="right" style={{ color: (p.unrealized_pnl ?? 0) >= 0 ? '#166534' : '#991b1b' }}>
                  {p.unrealized_pnl != null ? Number(p.unrealized_pnl).toFixed(2) : '—'}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

 
