import React, { useEffect, useState } from 'react';
import ProvenanceChip from './ProvenanceChip';
import StalePill from './StalePill';
import { getJSON } from '@/lib/fetchProvenance';

type Position = {
  symbol: string;
  qty: number;
  avg_price: number;
  market_price?: number | null;
  unrealized_pnl?: number | null;
  quote?: { stale?: boolean; cache_age_ms?: number; ttl_ms?: number };
};

export default function PositionsTable() {
  const [rows, setRows] = useState<Position[]>([]);
  const [meta, setMeta] = useState<any>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      const { data, meta } = await getJSON<{ positions: Position[] }>('/api/positions');
      if (!alive) return;
      setRows(data.positions || []);
      setMeta(meta || {});
    })();
    return () => { alive = false; };
  }, []);

  if (!meta) return null;

  return (
    <div style={{ padding: 16, borderRadius: 12, border: '1px solid #e5e7eb', background: '#fff' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 style={{ margin: 0 }}>Positions</h3>
        <ProvenanceChip source={meta.source} provider={meta.provider} asof_ts={meta.asof_ts} latency_ms={meta.latency_ms} />
      </div>

      <table style={{ width: '100%', marginTop: 12, fontSize: 14 }}>
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

 
