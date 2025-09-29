import React, { useState, useEffect } from 'react';
import ProvenanceChip from './ProvenanceChip';
import StalePill from './StalePill';
import { useSyncedPortfolio, useSyncedQuotes } from '@/hooks/useSyncedData';

type Position = {
  symbol: string;
  quantity: number;
  cost_basis: number;
  market_price?: number;
  current_value?: number;
  unrealized_pnl?: number;
  pnl_percent?: number;
  quote?: { stale?: boolean; cache_age_ms?: number; ttl_ms?: number };
};

export default function PositionsTable() {
  const { data: portfolio, isLoading } = useSyncedPortfolio();
  const [positionsWithPnL, setPositionsWithPnL] = useState<Position[]>([]);
  
  // Get positions from portfolio
  const positions = Array.isArray(portfolio?.positions) ? portfolio.positions : [];
  
  // Extract position symbols for quotes
  const positionSymbols = positions.map((pos: any) => pos.symbol).filter(Boolean);
  const { data: quotesData } = useSyncedQuotes(positionSymbols);
  
  // Calculate P&L for each position
  useEffect(() => {
    if (positions.length > 0 && quotesData && Array.isArray(quotesData)) {
      const positionsWithCalcs = positions.map((pos: any) => {
        const quote = quotesData.find((q: any) => q.symbol === pos.symbol);
        const currentPrice = quote?.last || quote?.price || 0;
        const costBasis = Number(pos.cost_basis || 0);
        const quantity = Number(pos.quantity || 0);
        const totalCost = costBasis * quantity;
        const currentValue = currentPrice * quantity;
        const pnl = currentValue - totalCost;
        const pnlPercent = totalCost > 0 ? (pnl / totalCost) * 100 : 0;
        
        return {
          ...pos,
          market_price: currentPrice,
          current_value: currentValue,
          unrealized_pnl: pnl,
          pnl_percent: pnlPercent,
          quote: quote
        };
      });
      
      setPositionsWithPnL(positionsWithCalcs);
    }
  }, [positions, quotesData]);
  
  if (isLoading || !portfolio) return null;
  
  const rows = positionsWithPnL;
  const meta = portfolio as any;

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
            const pnlColor = (p.unrealized_pnl ?? 0) >= 0 ? '#10b981' : '#ef4444';
            
            return (
              <tr key={p.symbol}>
                <td className="font-medium">{p.symbol}</td>
                <td align="right">{p.quantity}</td>
                <td align="right">${Number(p.cost_basis).toFixed(2)}</td>
                <td align="right" style={{ opacity: stale ? 0.6 : 1 }}>
                  ${p.market_price != null ? Number(p.market_price).toFixed(2) : '—'}
                  <StalePill stale={stale} ageMs={q.cache_age_ms} ttlMs={q.ttl_ms} />
                </td>
                <td align="right" style={{ color: pnlColor, fontWeight: 'medium' }}>
                  {p.unrealized_pnl != null ? (
                    <>
                      {p.unrealized_pnl >= 0 ? '+' : ''}${Number(p.unrealized_pnl).toFixed(2)}
                      <span className="text-xs ml-1">
                        ({p.unrealized_pnl >= 0 ? '+' : ''}{p.pnl_percent?.toFixed(2)}%)
                      </span>
                    </>
                  ) : '—'}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

 
