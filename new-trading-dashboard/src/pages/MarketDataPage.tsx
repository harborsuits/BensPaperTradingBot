import { useState, useEffect } from "react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { useQuotesQuery } from "@/hooks/useQuotes";
import { useUniverse } from "@/hooks/useUniverse";
import axios from "axios";
import { useQueryClient } from "@tanstack/react-query";
import { useBars } from "@/hooks/useBars";
import { useMarketOverview } from "@/hooks/useMarketOverview";
import {
  TrendingUp,
  TrendingDown,
  BarChart3,
  Plus,
  Search,
  X,
  RefreshCw,
  Clock,
  DollarSign,
  Activity
} from "lucide-react";

type Timeframe = "1Min" | "5Min" | "15Min" | "1Hour" | "1Day";
type ChartType = "candlestick" | "line" | "area";

interface WatchlistItem {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  lastUpdate: Date;
}

export default function MarketDataPage(){
  const qc = useQueryClient();
  const { watchlists } = useUniverse();
  const watchlistItems = watchlists.data?.items || watchlists.data || [];
  const currentId = watchlists.data?.currentId || watchlistItems[0]?.id;
  const current = watchlistItems.find((w:any) => w.id === currentId) || watchlistItems[0];
  const symbols = current?.symbols || [];
  const [selectedSymbol, setSelectedSymbol] = useState<string>("SPY");
  const [timeframe, setTimeframe] = useState<Timeframe>("1Day");
  const [chartType, setChartType] = useState<ChartType>("candlestick");
  const [newSymbol, setNewSymbol] = useState<string>("");
  const [showAddSymbol, setShowAddSymbol] = useState<boolean>(false);

  const overview = useMarketOverview();

  // Get quotes for watchlist
  const { data: quotes, isLoading: quotesLoading, refetch: refetchQuotes } = useQuotesQuery(symbols);

  // Get chart data for selected symbol
  const { data: chartData, isLoading: chartLoading, refetch: refetchChart } = useBars(
    selectedSymbol,
    timeframe,
    timeframe === "1Day" ? 180 : timeframe === "1Hour" ? 240 : 300
  );

  const timeframes: Timeframe[] = ["1Min", "5Min", "15Min", "1Hour", "1Day"];

  // Simple candlestick chart renderer (simplified for demo)
  const renderChart = () => {
    if (!chartData?.bars || chartData.bars.length === 0) {
      return (
        <div className="flex items-center justify-center h-64 text-muted-foreground">
          <div className="text-center">
            <BarChart3 className="w-12 h-12 mx-auto mb-2 opacity-50" />
            <p>No chart data available</p>
          </div>
        </div>
      );
    }

    const bars = chartData.bars.slice(-60);
    const maxPrice = Math.max(...bars.map(b => b.h));
    const minPrice = Math.min(...bars.map(b => b.l));
    const priceRange = maxPrice - minPrice || 1;

    // Layman-friendly summary
    const lastBar = bars[bars.length - 1];
    const refBar = bars.length > 1 ? bars[bars.length - 2] : bars[0];
    const lastPrice = Number(lastBar?.c ?? 0);
    const refPrice = Number(refBar?.c ?? refBar?.o ?? 0);
    const change = refPrice ? lastPrice - refPrice : 0;
    const pct = refPrice ? (change / refPrice) * 100 : 0;
    const dayLow = Math.min(...bars.map(b => b.l));
    const dayHigh = Math.max(...bars.map(b => b.h));
    const dayRange = dayHigh - dayLow || 1;
    const pos = Math.min(1, Math.max(0, (lastPrice - dayLow) / dayRange));

    // For intraday, render a simple line chart from closes
    if (timeframe !== "1Day") {
      const points = bars.map(b => Number(b.c));
      const max = Math.max(...points);
      const min = Math.min(...points);
      const range = (max - min) || 1;
      return (
        <div className="h-64 border rounded-lg bg-card p-4 overflow-hidden">
          <div className="flex justify-between items-center mb-3">
            <div>
              <h3 className="font-semibold">{selectedSymbol} · {timeframe}</h3>
              <div className="text-xs text-muted-foreground">{bars.length} bars</div>
            </div>
            <div className="text-right">
              <div className="text-xl font-bold">${lastPrice.toFixed(2)}</div>
              <div className={`${change >= 0 ? 'text-green-500' : 'text-red-500'} text-sm`}>
                {change >= 0 ? '+' : ''}{change.toFixed(2)} ({pct.toFixed(2)}%)
              </div>
            </div>
          </div>
          <div className="relative h-48">
            <svg width="100%" height="100%" viewBox={`0 0 ${bars.length - 1} 100`} preserveAspectRatio="none">
              <polyline
                fill="none"
                stroke={change >= 0 ? "#22c55e" : "#ef4444"}
                strokeWidth="1.5"
                points={points.map((p, i) => `${i},${100 - ((p - min) / range) * 100}`).join(' ')}
              />
            </svg>
          </div>
          <div className="flex justify-between text-xs text-muted-foreground mt-2">
            <span>${min.toFixed(2)}</span>
            <span>${max.toFixed(2)}</span>
          </div>
        </div>
      );
    }

    return (
      <div className="h-64 border rounded-lg bg-card p-4 overflow-hidden">
        <div className="flex justify-between items-center mb-3">
          <div>
            <h3 className="font-semibold">{selectedSymbol} · {timeframe}</h3>
            <div className="text-xs text-muted-foreground">{bars.length} bars</div>
          </div>
          <div className="text-right">
            <div className="text-xl font-bold">${lastPrice.toFixed(2)}</div>
            <div className={`text-sm ${change >= 0 ? 'text-green-500' : 'text-red-500'}`}>
              {change >= 0 ? '+' : ''}{change.toFixed(2)} ({(pct).toFixed(2)}%)
            </div>
          </div>
        </div>

        {/* Day range meter */}
        <div className="mb-3">
          <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
            <span>Low ${dayLow.toFixed(2)}</span>
            <span>High ${dayHigh.toFixed(2)}</span>
          </div>
          <div className="relative h-2 rounded bg-muted">
            <div className="absolute top-0 bottom-0 left-0 right-0 rounded border border-border" />
            <div
              className="absolute -top-1 w-1.5 h-4 rounded bg-primary"
              style={{ left: `${pos * 100}%`, transform: 'translateX(-50%)' }}
              title={`Current ${selectedSymbol}: $${lastPrice.toFixed(2)}`}
            />
          </div>
        </div>

        {/* Daily candlesticks */}
        <div className="flex items-end h-48 gap-1">
          {bars.map((bar, i) => {
            const height = ((bar.h - bar.l) / priceRange) * 180;
            const bodyHeight = Math.abs(bar.o - bar.c) / priceRange * 180;
            const bodyTop = ((Math.max(bar.o, bar.c) - bar.l) / priceRange) * 180;
            const wickTop = ((bar.h - bar.l) / priceRange) * 180;

            const isGreen = bar.c >= bar.o;
            const color = isGreen ? "bg-green-500" : "bg-red-500";

            return (
              <div key={i} className="flex flex-col items-center flex-1 min-w-0">
                {/* High-low wick */}
                <div
                  className="w-0.5 bg-foreground/60"
                  style={{ height: `${wickTop}px` }}
                />

                {/* Open-close body */}
                <div
                  className={`w-3 ${color} border border-foreground/20`}
                  style={{
                    height: `${Math.max(bodyHeight, 2)}px`,
                    marginTop: `${bodyTop}px`
                  }}
                />
              </div>
            );
          })}
        </div>

        {/* Price labels */}
        <div className="flex justify-between text-xs text-muted-foreground mt-2">
          <span>${minPrice.toFixed(2)}</span>
          <span>${maxPrice.toFixed(2)}</span>
        </div>

        {/* Legend */}
        <div className="mt-2 text-[11px] text-muted-foreground">
          <span className="mr-3"><span className="inline-block w-2 h-2 bg-green-500 rounded mr-1 align-middle" />Up</span>
          <span><span className="inline-block w-2 h-2 bg-red-500 rounded mr-1 align-middle" />Down</span>
        </div>
      </div>
    );
  };

  const addSymbol = async () => {
    const s = newSymbol.trim().toUpperCase();
    if (!s) return;
    try {
      await axios.post(`/api/watchlists/${current?.id || 'default'}/symbols`, { symbol: s });
      setNewSymbol("");
      setShowAddSymbol(false);
      refetchQuotes();
      qc.invalidateQueries({ queryKey: ["watchlists"] });
    } catch {}
  };

  const removeSymbol = async (symbol: string) => {
    try {
      await axios.delete(`/api/watchlists/${current?.id || 'default'}/symbols/${encodeURIComponent(symbol)}`);
      if (selectedSymbol === symbol && symbols.length > 1) {
        setSelectedSymbol(symbols.find(s => s !== symbol) || 'SPY');
      }
      refetchQuotes();
      qc.invalidateQueries({ queryKey: ["watchlists"] });
    } catch {}
  };

  return (
    <div className="w-full min-h-screen bg-background">
      <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Header Section */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <h1 className="text-3xl font-bold text-foreground">Market Data</h1>
            <Button
              onClick={() => {
                refetchQuotes();
                refetchChart();
              }}
              className="flex items-center gap-2"
            >
              <RefreshCw size={16} />
              Refresh Data
            </Button>
          </div>
          <p className="text-sm text-muted-foreground mt-2">
            Real-time charts, quotes, and market analysis
          </p>
        </div>

        {/* Main Dashboard Content - Vertical Flow */}
        <div className="dashboard-container">
          {/* Watchlist Section */}
          <div className="dashboard-section">
            <Card className="card">
              <div className="card-header">
                <div className="flex items-center justify-between">
                  <h3 className="card-title">Watchlist</h3>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => setShowAddSymbol(true)}
                    className="flex items-center gap-1"
                  >
                    <Plus size={14} />
                    Add Symbol
                  </Button>
                </div>
                <div className="card-subtle">{symbols.length} symbols</div>
              </div>
              <div className="card-content">
                {/* Add Symbol Input */}
                {showAddSymbol && (
                  <div className="mb-6 p-4 border border-border rounded-lg bg-card/30">
                    <div className="flex gap-2">
                      <input
                        type="text"
                        placeholder="Enter symbol (e.g., MSFT)"
                        value={newSymbol}
                        onChange={(e) => setNewSymbol(e.target.value.toUpperCase())}
                        className="flex-1 px-3 py-2 text-sm border border-border rounded focus:ring-2 focus:ring-primary focus:border-transparent"
                        onKeyPress={(e) => e.key === 'Enter' && addSymbol()}
                      />
                      <Button size="sm" onClick={addSymbol}>
                        Add
                      </Button>
                      <Button size="sm" variant="outline" onClick={() => setShowAddSymbol(false)}>
                        <X size={14} />
                      </Button>
                    </div>
                  </div>
                )}

                {/* Watchlist Items */}
                <div className="space-y-3">
                  {quotesLoading ? (
                    <div className="text-center py-8">
                      <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-3 text-primary" />
                      <p className="text-sm text-muted-foreground">Loading quotes...</p>
                    </div>
                  ) : !quotes || quotes.length === 0 ? (
                    <div className="text-center py-8 text-muted-foreground">
                      <Activity className="w-8 h-8 mx-auto mb-3 opacity-50" />
                      <p className="text-lg font-medium">No symbols in watchlist</p>
                      <p className="text-sm mt-2">Add symbols to start tracking market data.</p>
                    </div>
                  ) : (
                    quotes.map((quote: any) => (
                      <div
                        key={quote.symbol}
                        className={`flex items-center justify-between p-4 rounded-lg border cursor-pointer transition-all duration-200 ${
                          selectedSymbol === quote.symbol
                            ? 'bg-primary/10 border-primary/30 shadow-sm'
                            : 'hover:bg-muted/50 border-border hover:shadow-sm'
                        }`}
                        onClick={() => setSelectedSymbol(quote.symbol)}
                      >
                        <div className="flex-1">
                          <div className="flex items-center gap-3">
                            <span className="font-semibold text-lg">{quote.symbol}</span>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={(e) => {
                                e.stopPropagation();
                                removeSymbol(quote.symbol);
                              }}
                              className="h-7 w-7 p-0 opacity-50 hover:opacity-100 hover:bg-destructive/20 hover:text-destructive"
                            >
                              <X size={14} />
                            </Button>
                          </div>
                          <div className="text-sm text-muted-foreground mt-1">
                            ${Number(quote.last || quote.price || 0).toFixed(2)}
                          </div>
                        </div>
                        <div className="text-right">
                          <div className={`text-lg font-semibold ${
                            (quote.pct || 0) >= 0 ? 'text-green-400' : 'text-red-400'
                          }`}>
                            {(quote.pct || 0) >= 0 ? '+' : ''}{Number(quote.pct || 0).toFixed(2)}%
                          </div>
                          <div className={`flex items-center gap-1 text-sm mt-1 ${
                            (quote.pct || 0) >= 0 ? 'text-green-400' : 'text-red-400'
                          }`}>
                            {(quote.pct || 0) >= 0 ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                            ${Math.abs(Number(quote.change || 0)).toFixed(2)}
                          </div>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </Card>
          </div>

          {/* Chart Section */}
          <div className="dashboard-section">
            <Card className="card">
              <div className="card-header">
                <div className="flex items-center justify-between">
                  <h3 className="card-title">Chart - {selectedSymbol}</h3>
                  <div className="flex items-center gap-3">
                    <div className="card-subtle">{timeframe}</div>
                    {/* Timeframe Selector */}
                    <div className="flex gap-1">
                      {timeframes.map((tf) => (
                        <Button
                          key={tf}
                          size="sm"
                          variant={timeframe === tf ? "default" : "outline"}
                          onClick={() => setTimeframe(tf)}
                          className="text-xs px-3"
                        >
                          {tf.replace('Min', 'm').replace('Hour', 'h').replace('Day', 'd')}
                        </Button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
              <div className="card-content">
                {renderChart()}
              </div>
            </Card>
          </div>

          {/* Market Overview Section */}
          <div className="dashboard-section">
            <Card className="card">
              <div className="card-header">
                <h3 className="card-title">Market Overview</h3>
                <div className="card-subtle">Updated {overview.data?.asOf ? new Date(overview.data.asOf).toLocaleTimeString() : ""}</div>
              </div>
              <div className="card-content">
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                  <div className="flex items-center gap-4 p-4 bg-card/30 rounded-lg border border-border/50">
                    <div className="p-3 bg-blue-500/20 rounded-lg">
                      <Activity className="w-6 h-6 text-blue-400" />
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Market Status</p>
                      <p className="font-semibold">{overview.data?.marketStatus || "Unknown"}</p>
                    </div>
                  </div>

                  <div className="flex items-center gap-4 p-4 bg-card/30 rounded-lg border border-border/50">
                    <div className="p-3 bg-green-500/20 rounded-lg">
                      <TrendingUp className="w-6 h-6 text-green-400" />
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">S&P 500 (SPY)</p>
                      <p className={`font-semibold ${((overview.data?.spx?.pct || 0) >= 0) ? 'text-green-400' : 'text-red-400'}`}>
                        {overview.data?.spx ? `${overview.data.spx.pct >= 0 ? '+' : ''}${overview.data.spx.pct.toFixed(2)}%` : '—'}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-4 p-4 bg-card/30 rounded-lg border border-border/50">
                    <div className="p-3 bg-purple-500/20 rounded-lg">
                      <div className="w-6 h-6 text-purple-400 flex items-center justify-center font-bold text-xs">
                        VIX
                      </div>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Volatility Index</p>
                      <p className="font-semibold">{overview.data?.vix ? overview.data.vix.last.toFixed(2) : '—'}</p>
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
