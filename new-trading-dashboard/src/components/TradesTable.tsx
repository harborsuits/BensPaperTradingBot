import React, { useEffect, useState } from 'react';
import ProvenanceChip from './ProvenanceChip';
import { getJSON } from '@/lib/fetchProvenance';
import { sseManager } from '@/services/sseManager';

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
    const sseService = sseManager.getConnection(`${window.location.origin}/api/paper/orders/stream`);
    
    const handleOrderUpdate = (message: any) => {
      try {
        setRows(prev => [message.data, ...prev].slice(0, 500));
      } catch {}
    };
    
    sseService.on('order_update', handleOrderUpdate);
    
    return () => {
      sseService.off('order_update', handleOrderUpdate);
    };
  }, []);

  if (!meta) return null;

  return (
    <div className="p-4 rounded-xl border bg-card text-foreground border-border">
      <div className="flex items-center justify-between">
        <h3 className="m-0">Trades</h3>
        <ProvenanceChip source={meta.source} provider={meta.provider} asof_ts={meta.asof_ts} latency_ms={meta.latency_ms} />
      </div>

      <table className="w-full mt-3 text-sm">
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


