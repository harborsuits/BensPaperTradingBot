import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { TrendingUp, TrendingDown, DollarSign, Target, AlertCircle, CheckCircle } from "lucide-react";

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

export default function PortfolioCard() {
  const { data: portfolio, isLoading, error } = useQuery<PortfolioData>({
    queryKey: ["portfolio", "summary"],
    queryFn: async () => {
      const response = await fetch("/api/portfolio/summary");
      if (!response.ok) {
        throw new Error("Failed to fetch portfolio");
      }
      return response.json();
    },
    refetchInterval: 5000, // Update every 5 seconds
    staleTime: 2000,
  });

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
          {[1, 2, 3].map(i => (
            <div key={i} className="animate-pulse">
              <div className="h-12 bg-gray-200 rounded"></div>
            </div>
          ))}
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
          <div className="flex items-center gap-1 text-xs text-red-600">
            <AlertCircle className="w-3 h-3" />
            Disconnected
          </div>
        </div>
        <div className="mt-4 text-center py-8 text-muted-foreground">
          <AlertCircle className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p className="text-sm">Portfolio data unavailable</p>
          <p className="text-xs mt-1">Broker connection required</p>
        </div>
      </div>
    );
  }

  const totalValue = portfolio.cash + portfolio.equity;
  const dayPnlPercent = totalValue > 0 ? (portfolio.day_pnl / totalValue) * 100 : 0;
  const openPnlPercent = totalValue > 0 ? (portfolio.open_pnl / totalValue) * 100 : 0;

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
            {portfolio.broker} {portfolio.mode}
          </div>
          <span className="text-xs text-muted-foreground">
            {new Date(portfolio.asOf).toLocaleTimeString()}
          </span>
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
            <div className="text-xl font-bold">{formatCurrency(portfolio.cash)}</div>
          </div>
        </div>

        {/* P&L Row */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-muted/50 rounded-lg p-3">
            <div className="text-sm text-muted-foreground">Day P&L</div>
            <div className={`text-lg font-bold flex items-center gap-1 ${
              portfolio.day_pnl >= 0 ? 'text-green-600' : 'text-red-600'
            }`}>
              {portfolio.day_pnl >= 0 ? (
                <TrendingUp className="w-4 h-4" />
              ) : (
                <TrendingDown className="w-4 h-4" />
              )}
              {formatCurrency(portfolio.day_pnl)}
              <span className="text-sm">({formatPercent(dayPnlPercent)})</span>
            </div>
          </div>
          <div className="bg-muted/50 rounded-lg p-3">
            <div className="text-sm text-muted-foreground">Open P&L</div>
            <div className={`text-lg font-bold flex items-center gap-1 ${
              portfolio.open_pnl >= 0 ? 'text-green-600' : 'text-red-600'
            }`}>
              {portfolio.open_pnl >= 0 ? (
                <TrendingUp className="w-4 h-4" />
              ) : (
                <TrendingDown className="w-4 h-4" />
              )}
              {formatCurrency(portfolio.open_pnl)}
              <span className="text-sm">({formatPercent(openPnlPercent)})</span>
            </div>
          </div>
        </div>

        {/* Positions */}
        {portfolio.positions.length > 0 && (
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium">Top Positions</span>
              <span className="text-xs text-muted-foreground">
                {portfolio.positions.length} total
              </span>
            </div>
            <div className="space-y-2">
              {portfolio.positions.slice(0, 3).map((position) => (
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

        {portfolio.positions.length === 0 && (
          <div className="text-center py-4 text-muted-foreground">
            <div className="text-sm">No open positions</div>
          </div>
        )}
      </div>
    </div>
  );
}
