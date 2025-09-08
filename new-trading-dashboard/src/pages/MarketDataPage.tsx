import { useState, useEffect } from "react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { useQuotesQuery } from "@/hooks/useQuotes";
import { useBars } from "@/hooks/useBars";
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
  const [symbols, setSymbols] = useState<string[]>(["SPY", "QQQ", "AAPL", "NVDA", "TSLA"]);
  const [selectedSymbol, setSelectedSymbol] = useState<string>("SPY");
  const [timeframe, setTimeframe] = useState<Timeframe>("1Day");
  const [chartType, setChartType] = useState<ChartType>("candlestick");
  const [newSymbol, setNewSymbol] = useState<string>("");
  const [showAddSymbol, setShowAddSymbol] = useState<boolean>(false);

  // Get quotes for watchlist
  const { data: quotes, isLoading: quotesLoading, refetch: refetchQuotes } = useQuotesQuery(symbols);

  // Get chart data for selected symbol
  const { data: chartData, isLoading: chartLoading, refetch: refetchChart } = useBars(
    selectedSymbol,
    timeframe,
    timeframe === "1Day" ? 90 : timeframe === "1Hour" ? 168 : 100
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

    const bars = chartData.bars.slice(-50); // Show last 50 bars
    const maxPrice = Math.max(...bars.map(b => b.h));
    const minPrice = Math.min(...bars.map(b => b.l));
    const priceRange = maxPrice - minPrice || 1;

    return (
      <div className="h-64 border rounded-lg bg-card p-4 overflow-hidden">
        <div className="flex justify-between items-center mb-4">
          <h3 className="font-semibold">{selectedSymbol} - {timeframe}</h3>
          <div className="text-sm text-muted-foreground">
            {bars.length} bars
          </div>
        </div>

        {/* Simplified chart visualization */}
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
      </div>
    );
  };

  const addSymbol = () => {
    if (newSymbol && !symbols.includes(newSymbol.toUpperCase())) {
      setSymbols([...symbols, newSymbol.toUpperCase()]);
      setNewSymbol("");
      setShowAddSymbol(false);
    }
  };

  const removeSymbol = (symbol: string) => {
    if (symbols.length > 1) {
      setSymbols(symbols.filter(s => s !== symbol));
      if (selectedSymbol === symbol) {
        setSelectedSymbol(symbols[0] === symbol ? symbols[1] : symbols[0]);
      }
    }
  };

  return (
    <div className="w-full py-6">
      <div className="w-full max-w-7xl mx-auto px-4 xl:px-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold">Market Data</h1>
            <p className="text-sm text-muted-foreground">
              Real-time charts, quotes, and market analysis
            </p>
          </div>
          <Button
            onClick={() => {
              refetchQuotes();
              refetchChart();
            }}
            className="flex items-center gap-2"
          >
            <RefreshCw size={16} />
            Refresh
          </Button>
        </div>

        {/* Watchlist and Chart Section */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 mb-6">
          {/* Watchlist */}
          <Card className="xl:col-span-1">
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
                  Add
                </Button>
              </div>
            </div>
            <div className="card-content">
              {/* Add Symbol Input */}
              {showAddSymbol && (
                <div className="mb-4 p-3 border rounded-lg bg-muted/50">
                  <div className="flex gap-2">
                    <input
                      type="text"
                      placeholder="Enter symbol (e.g., MSFT)"
                      value={newSymbol}
                      onChange={(e) => setNewSymbol(e.target.value.toUpperCase())}
                      className="flex-1 px-3 py-2 text-sm border rounded"
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
              <div className="space-y-2">
                {quotesLoading ? (
                  <div className="text-center py-4">
                    <RefreshCw className="w-6 h-6 animate-spin mx-auto mb-2" />
                    <p className="text-sm text-muted-foreground">Loading quotes...</p>
                  </div>
                ) : !quotes || quotes.length === 0 ? (
                  <div className="text-center py-4 text-muted-foreground">
                    No symbols in watchlist
                  </div>
                ) : (
                  quotes.map((quote: any) => (
                    <div
                      key={quote.symbol}
                      className={`flex items-center justify-between p-3 rounded-lg border cursor-pointer transition-colors ${
                        selectedSymbol === quote.symbol
                          ? 'bg-primary/10 border-primary/30'
                          : 'hover:bg-muted/50 border-border'
                      }`}
                      onClick={() => setSelectedSymbol(quote.symbol)}
                    >
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{quote.symbol}</span>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={(e) => {
                              e.stopPropagation();
                              removeSymbol(quote.symbol);
                            }}
                            className="h-6 w-6 p-0 opacity-50 hover:opacity-100"
                          >
                            <X size={12} />
                          </Button>
                        </div>
                        <div className="text-sm text-muted-foreground">
                          ${Number(quote.last || quote.price || 0).toFixed(2)}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className={`text-sm font-medium ${
                          (quote.pct || 0) >= 0 ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {(quote.pct || 0) >= 0 ? '+' : ''}{Number(quote.pct || 0).toFixed(2)}%
                        </div>
                        <div className={`flex items-center gap-1 text-xs ${
                          (quote.pct || 0) >= 0 ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {(quote.pct || 0) >= 0 ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                          ${Math.abs(Number(quote.change || 0)).toFixed(2)}
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </Card>

          {/* Chart */}
          <Card className="xl:col-span-2">
            <div className="card-header">
              <div className="flex items-center justify-between">
                <h3 className="card-title">Chart</h3>
                <div className="flex items-center gap-2">
                  {/* Timeframe Selector */}
                  <div className="flex gap-1">
                    {timeframes.map((tf) => (
                      <Button
                        key={tf}
                        size="sm"
                        variant={timeframe === tf ? "default" : "outline"}
                        onClick={() => setTimeframe(tf)}
                        className="text-xs"
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

        {/* Market Overview */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card>
            <div className="card-content">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-blue-500/20 rounded-lg">
                  <Activity className="w-6 h-6 text-blue-400" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Market Status</p>
                  <p className="font-semibold text-green-400">Open</p>
                </div>
              </div>
            </div>
          </Card>

          <Card>
            <div className="card-content">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-green-500/20 rounded-lg">
                  <TrendingUp className="w-6 h-6 text-green-400" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">S&P 500</p>
                  <p className="font-semibold">+0.45%</p>
                </div>
              </div>
            </div>
          </Card>

          <Card>
            <div className="card-content">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-purple-500/20 rounded-lg">
                  <div className="w-6 h-6 text-purple-400 flex items-center justify-center font-bold text-xs">
                    VIX
                  </div>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Volatility Index</p>
                  <p className="font-semibold">17.2</p>
                </div>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
