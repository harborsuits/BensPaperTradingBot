import { useState } from "react";
import { TrendingUp, TrendingDown, DollarSign, Target, AlertCircle, CheckCircle } from "lucide-react";
import { z } from "zod";
import ProvenanceChip from "@/components/ProvenanceChip";
import ErrorState from "@/components/ErrorState";
import { useSyncedPortfolio } from "@/hooks/useSyncedData";

interface PortfolioData {
  cash: number;
  equity: number;
  day_pnl: number;
  open_pnl: number;
  positions: Array<{
    symbol: string;
    qty: number;
    avg_cost: number;
    last: number;
    pnl: number;
  }>;
  asOf: string;
  broker: string;
  mode: string;
}

// Zod schema for runtime validation - made more permissive
const PortfolioSchema = z.object({
  cash: z.number().default(0),
  equity: z.number().default(0),
  day_pnl: z.number().default(0),
  open_pnl: z.number().default(0),
  total_value: z.number().optional(), // Allow total_value if present
  positions: z.array(z.object({
    symbol: z.string().optional(),
    qty: z.number().optional(),
    avg_cost: z.number().optional(),
    last: z.number().optional(),
    pnl: z.number().optional(),
  })).default([]),
  asOf: z.string().default(new Date().toISOString()),
  broker: z.string().default('unknown'),
  mode: z.string().default('paper'),
}).passthrough(); // Allow additional fields

export default function PortfolioCard() {
  const { data: portfolio, isLoading, error } = useSyncedPortfolio();
  
  // Metadata is part of the portfolio object from Tradier
  const meta = portfolio as any;
  const positions = Array.isArray(portfolio?.positions) ? portfolio.positions : [];
  // Handle various portfolio data structures from different endpoints
  const equity = Number(portfolio?.equity ?? portfolio?.total_equity ?? 0);
  const cash = Number(portfolio?.cash ?? portfolio?.cash_balance ?? 0);
  const dayPnl = Number(portfolio?.day_pnl ?? portfolio?.daily_pl ?? 0);
  const openPnl = Number(portfolio?.open_pnl ?? portfolio?.unrealized_pl ?? 0);
  const broker = portfolio?.broker ?? 'tradier';
  const mode = portfolio?.mode ?? 'paper';
  const asOf = portfolio?.asOf ? new Date(portfolio.asOf).toLocaleTimeString() : 'N/A';
  // Accept provenance fields if backend stamps them
  const source = (portfolio as any)?.source as string | undefined;
  const provider = (portfolio as any)?.provider as string | undefined;
  const asof_ts = (portfolio as any)?.asof_ts as string | undefined;
  const latency_ms = (portfolio as any)?.latency_ms as number | undefined;
  const redFlag = Boolean((portfolio as any)?.reality_red_flag);
  const metaAny = portfolio as any;

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  if (isLoading) {
    return (
      <div className="border rounded-2xl p-4">
        <div className="flex items-baseline justify-between">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <DollarSign className="w-5 h-5" />
            Portfolio
          </h3>
        </div>
        <div className="mt-4 space-y-4">
          {/* Total Value skeleton */}
          <div className="animate-pulse">
            <div className="h-4 bg-muted rounded w-24 mb-1"></div>
            <div className="h-8 bg-muted rounded w-32"></div>
          </div>

          {/* Cash and Equity skeleton */}
          <div className="grid grid-cols-2 gap-4">
            <div className="animate-pulse">
              <div className="h-4 bg-muted rounded w-16 mb-1"></div>
              <div className="h-6 bg-muted rounded w-20"></div>
            </div>
            <div className="animate-pulse">
              <div className="h-4 bg-muted rounded w-16 mb-1"></div>
              <div className="h-6 bg-muted rounded w-20"></div>
            </div>
          </div>

          {/* P&L skeleton */}
          <div className="grid grid-cols-2 gap-4">
            <div className="animate-pulse">
              <div className="h-4 bg-muted rounded w-16 mb-1"></div>
              <div className="h-6 bg-muted rounded w-20"></div>
            </div>
            <div className="animate-pulse">
              <div className="h-4 bg-muted rounded w-16 mb-1"></div>
              <div className="h-6 bg-muted rounded w-20"></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error || !portfolio) {
    return (
      <div className="border rounded-2xl p-4">
        <div className="flex items-baseline justify-between">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <DollarSign className="w-5 h-5" />
            Portfolio
          </h3>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1 text-xs text-red-600">
              <AlertCircle className="w-3 h-3" />
              Disconnected
            </div>
            <ProvenanceChip source={meta?.source} provider={meta?.provider} asof_ts={meta?.asof_ts} latency_ms={meta?.latency_ms} />
          </div>
        </div>
        <ErrorState text={(error as any)?.data?.error || 'Portfolio data unavailable'} />
      </div>
    );
  }

  // Equity already includes cash, so total value is just equity
  const totalValue = equity;
  const dayPnlPercent = totalValue > 0 ? (dayPnl / totalValue) * 100 : 0;
  const openPnlPercent = totalValue > 0 ? (openPnl / totalValue) * 100 : 0;

  return (
    <div className="border rounded-2xl p-4">
      <div className="flex items-baseline justify-between">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <DollarSign className="w-5 h-5" />
          Portfolio
        </h3>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 text-xs text-green-600">
            <CheckCircle className="w-3 h-3" />
            {broker} {mode}
          </div>
          <span className="text-xs text-muted-foreground">{asOf}</span>
          <ProvenanceChip source={source || meta?.source} provider={provider || meta?.provider} asof_ts={asof_ts || meta?.asof_ts} latency_ms={latency_ms || meta?.latency_ms} />
        </div>
      </div>

      <div className="mt-4 space-y-4">
        {/* Summary Row */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-muted/50 rounded-lg p-3">
            <div className="text-sm text-muted-foreground">Total Value</div>
            <div className="text-xl font-bold">{formatCurrency(totalValue)}</div>
          </div>
          <div className="bg-muted/50 rounded-lg p-3">
            <div className="text-sm text-muted-foreground">Cash</div>
            <div className="text-xl font-bold">{formatCurrency(cash)}</div>
          </div>
        </div>

        {/* P&L Row */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-muted/50 rounded-lg p-3">
            <div className="text-sm text-muted-foreground">Day P&L</div>
            <div className={`text-lg font-bold flex items-center gap-1 ${
              dayPnl >= 0 ? 'text-green-600' : 'text-red-600'
            }`}>
              {dayPnl >= 0 ? (
                <TrendingUp className="w-4 h-4" />
              ) : (
                <TrendingDown className="w-4 h-4" />
              )}
              {formatCurrency(dayPnl)}
              <span className="text-sm">({formatPercent(dayPnlPercent)})</span>
            </div>
          </div>
          <div className="bg-muted/50 rounded-lg p-3">
            <div className="text-sm text-muted-foreground">Open P&L</div>
            <div className={`text-lg font-bold flex items-center gap-1 ${
              openPnl >= 0 ? 'text-green-600' : 'text-red-600'
            }`}>
              {openPnl >= 0 ? (
                <TrendingUp className="w-4 h-4" />
              ) : (
                <TrendingDown className="w-4 h-4" />
              )}
              {formatCurrency(openPnl)}
              <span className="text-sm">({formatPercent(openPnlPercent)})</span>
            </div>
          </div>
        </div>

        {redFlag && (
          <div className="text-xs text-red-600">Equity mismatch detected. Check broker vs derived valuation.</div>
        )}

        {(metaAny?.quotes_meta?.missing > 0 || metaAny?.stale_quotes_count > 0) && (
          <div className="text-xs text-amber-600 mt-1">
            {metaAny?.quotes_meta?.missing > 0 && <span>{metaAny.quotes_meta.missing} symbols valued via avg_price.</span>}
            {metaAny?.quotes_meta?.missing > 0 && metaAny?.stale_quotes_count > 0 && <span> · </span>}
            {metaAny?.stale_quotes_count > 0 && <span>{metaAny.stale_quotes_count} symbols valued with stale quotes.</span>}
          </div>
        )}

        {/* Positions */}
        {positions.length > 0 && (
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium">Top Performers</span>
              <span className="text-xs text-muted-foreground">
                {positions.length} total
              </span>
            </div>
            <div className="space-y-2">
              {positions
                .sort((a, b) => (b.pnl || 0) - (a.pnl || 0)) // Sort by P&L descending (best first)
                .slice(0, 3)
                .map((position) => (
                <div key={position.symbol} className="flex items-center justify-between bg-muted/30 rounded-lg p-2">
                  <div className="flex items-center gap-2">
                    <Target className="w-4 h-4 text-muted-foreground" />
                    <span className="font-medium">{position.symbol}</span>
                    <span className="text-xs text-muted-foreground">
                      {position.qty} @ {formatCurrency(position.avg_cost)}
                    </span>
                  </div>
                  <div className="text-right">
                    <div className={`text-sm font-medium ${
                      position.pnl >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {formatCurrency(position.pnl)}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {formatCurrency(position.last)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {positions.length === 0 && (
          <div className="text-center py-6 text-muted-foreground">
            <Target className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <div className="text-sm font-medium mb-1">No Open Positions</div>
            <div className="text-xs">All cash is available for trading</div>
          </div>
        )}
      </div>
    </div>
  );
}
