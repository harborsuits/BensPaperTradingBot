import React, { useEffect, useState } from 'react';
import ProvenanceChip from './ProvenanceChip';
import { getJSON } from '@/lib/fetchProvenance';

type Trade = {
  trade_id?: string;
  mode: 'live' | 'paper';
  strategy?: string;
  symbol: string;
  side: 'buy' | 'sell' | 'short' | 'cover';
  qty: number;
  price?: number | null;
  ts_exec: string;
  broker_order_id?: string | null;
  venue?: string | null;
};

export default function TradesTable() {
  const [rows, setRows] = useState<Trade[]>([]);
  const [meta, setMeta] = useState<any>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      const { data, meta } = await getJSON<{ trades: Trade[] }>('/api/trades');
      if (!alive) return;
      setRows(data.trades || []);
      setMeta(meta || {});
    })();
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    const es = new EventSource('/api/paper/orders/stream');
    const onUpdate = (evt: MessageEvent) => {
      try {
        const data = JSON.parse(evt.data);
        setRows(prev => [data, ...prev].slice(0, 500));
      } catch {}
    };
    es.addEventListener('order_update', onUpdate as any);
    return () => es.close();
  }, []);

  if (!meta) return null;

  return (
    <div style={{ padding: 16, borderRadius: 12, border: '1px solid #e5e7eb', background: '#fff' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 style={{ margin: 0 }}>Trades</h3>
        <ProvenanceChip source={meta.source} provider={meta.provider} asof_ts={meta.asof_ts} latency_ms={meta.latency_ms} />
      </div>

      <table style={{ width: '100%', marginTop: 12, fontSize: 14 }}>
        <thead>
          <tr>
            <th>Time</th><th>Mode</th><th>Symbol</th><th>Side</th>
            <th align="right">Qty</th><th align="right">Price</th>
            <th>Broker ID</th><th>Venue</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((t, i) => (
            <tr key={t.trade_id || t.broker_order_id || `${t.symbol}-${t.ts_exec}-${i}`}>
              <td>{new Date(t.ts_exec).toLocaleTimeString()}</td>
              <td>{t.mode === 'live' ? 'Live' : 'Paper'}</td>
              <td>{t.symbol}</td>
              <td>{t.side}</td>
              <td align="right">{t.qty}</td>
              <td align="right">{t.price != null ? Number(t.price).toFixed(2) : '—'}</td>
              <td>{t.broker_order_id || '—'}</td>
              <td>{t.venue || '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}


