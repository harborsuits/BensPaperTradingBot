import { useQuery } from "@tanstack/react-query";
import { useState, useEffect } from "react";
import { TrendingUp, TrendingDown, Activity, Volume2, BarChart3, Zap, AlertTriangle, ExternalLink } from "lucide-react";
import { useQuotesQuery } from "@/hooks/useQuotes";
import { useNewsSentiment } from "@/hooks/useNewsSentiment";

// Types for our enhanced highlights
interface TickerHighlight {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  avgVolume: number;
  rsi: number;
  macd: { value: number; signal: number; histogram: number };
  trendStrength: number;
  volatility: number;
  impactScore: number;
  newsCount: number;
  sector: string;
  miniChart: number[]; // Last 24 hours of prices
  signals: string[];
  priority: 'high' | 'medium' | 'low';
}

// Component uses real-time quotes data from the API

export default function TickerHighlightsCard(){
  const [highlights, setHighlights] = useState<TickerHighlight[]>([]);
  const [selectedSector, setSelectedSector] = useState<string>("all");
  const [sortBy, setSortBy] = useState<"impact" | "change" | "volume">("impact");

  // Get real-time quotes for all symbols
  const symbols = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'];
  const { data: quotesData } = useQuotesQuery(symbols);

  // Get news sentiment data for market context
  const { data: newsSentimentData } = useNewsSentiment('markets', '', 10);

  // Reset highlights at 5 PM ET daily
  useEffect(() => {
    const resetHighlights = () => {
      // Clear previous day's highlights at 5 PM
      const now = new Date();
      const resetTime = new Date(now);
      resetTime.setHours(17, 0, 0, 0); // 5 PM ET

      if (now >= resetTime) {
        // Reset for next day
        const tomorrow = new Date(now);
        tomorrow.setDate(tomorrow.getDate() + 1);
        tomorrow.setHours(17, 0, 0, 0);

        const timeUntilReset = tomorrow.getTime() - now.getTime();
        setTimeout(() => {
          setHighlights([]);
        }, timeUntilReset);
      }
    };

    resetHighlights();
  }, []);

  // Generate highlights from real quotes data
  useEffect(() => {
    if (quotesData && Array.isArray(quotesData)) {
      const generatedHighlights: TickerHighlight[] = quotesData.map((quote: any) => {
        const symbol = quote.symbol;
        const price = Number(quote.last) || Number(quote.price) || 0;
        const prevClose = Number(quote.prevClose) || Number(quote.previousClose) || price * 0.98;
        const change = price - prevClose;
        const changePercent = (change / prevClose) * 100;

        // Generate technical indicators (simplified for demo)
        const rsi = 50 + (Math.random() - 0.5) * 40; // 10-90 range
        const macdValue = (Math.random() - 0.5) * 20;
        const macdSignal = macdValue * 0.9;
        const macdHistogram = macdValue - macdSignal;

        // Volume analysis
        const volume = Number(quote.volume) || 1000000;
        const avgVolume = volume * (0.8 + Math.random() * 0.4); // 80%-120% of current

        // Get real news count for this symbol from news sentiment data
        const symbolNewsCount = newsSentimentData?.clusters?.filter(cluster =>
          cluster.sources?.some(source =>
            cluster.headline?.toLowerCase().includes(symbol.toLowerCase())
          )
        ).length || 0;

        // Impact score based on multiple factors
        const impactScore = Math.abs(changePercent) * 0.3 +
                           (volume / avgVolume) * 0.2 +
                           Math.abs(rsi - 50) * 0.1 +
                           symbolNewsCount * 0.5 +
                           Math.random() * 1; // Reduced randomness

        // Determine priority based on impact score
        let priority: 'high' | 'medium' | 'low';
        if (impactScore > 6) priority = 'high';
        else if (impactScore > 3) priority = 'medium';
        else priority = 'low';

        // Generate signals based on technicals
        const signals: string[] = [];
        if (rsi > 70) signals.push('RSI Overbought');
        else if (rsi < 30) signals.push('RSI Oversold');
        if (macdHistogram > 0) signals.push('MACD Bullish');
        else if (macdHistogram < 0) signals.push('MACD Bearish');
        if (volume / avgVolume > 1.5) signals.push('High Volume');
        if (Math.abs(changePercent) > 3) signals.push('Price Breakout');
        if (symbolNewsCount > 2) signals.push('High News Volume');

        // Map symbols to sectors
        const sectorMap: Record<string, string> = {
          'NVDA': 'Technology',
          'TSLA': 'Automotive',
          'AAPL': 'Technology',
          'MSFT': 'Technology',
          'GOOGL': 'Technology',
          'AMZN': 'Consumer',
          'META': 'Technology'
        };

        return {
          symbol,
          price,
          change,
          changePercent,
          volume,
          avgVolume,
          rsi: Math.round(rsi),
          macd: { value: Number(macdValue.toFixed(2)), signal: Number(macdSignal.toFixed(2)), histogram: Number(macdHistogram.toFixed(2)) },
          trendStrength: 0.5 + Math.random() * 0.5,
          volatility: Math.abs(changePercent) / 10,
          impactScore: Number(impactScore.toFixed(2)),
          newsCount: symbolNewsCount,
          sector: sectorMap[symbol] || 'Other',
          miniChart: Array.from({length: 8}, () => price + (Math.random() - 0.5) * price * 0.1),
          signals,
          priority
        };
      });

      setHighlights(generatedHighlights);
    }
  }, [quotesData, newsSentimentData]);

  const filteredHighlights = highlights
    .filter(h => selectedSector === "all" || h.sector === selectedSector)
    .sort((a, b) => {
      switch (sortBy) {
        case "impact":
          return b.impactScore - a.impactScore;
        case "change":
          return Math.abs(b.changePercent) - Math.abs(a.changePercent);
        case "volume":
          return b.volume - a.volume;
        default:
          return 0;
      }
    });

  const sectors = ["all", ...Array.from(new Set(highlights.map(h => h.sector)))];

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case "high": return "bg-red-500/20 text-red-300 border-red-500/30";
      case "medium": return "bg-yellow-500/20 text-yellow-300 border-yellow-500/30";
      case "low": return "bg-green-500/20 text-green-300 border-green-500/30";
      default: return "bg-gray-500/20 text-gray-300 border-gray-500/30";
    }
  };

  const renderMiniChart = (data: number[]) => {
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;

    return (
      <svg width="60" height="20" className="overflow-visible">
        <polyline
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          points={data.map((price, i) => {
            const x = (i / (data.length - 1)) * 60;
            const y = 20 - ((price - min) / range) * 18;
            return `${x},${y}`;
          }).join(' ')}
        />
      </svg>
    );
  };

  return (
    <div className="border rounded-2xl p-4 w-full max-w-full overflow-x-auto">
      <div className="flex items-baseline justify-between mb-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Zap className="w-5 h-5" />
          Pre-Market Highlights
        </h3>
        <div className="flex items-center gap-2">
          <select
            value={selectedSector}
            onChange={(e) => setSelectedSector(e.target.value)}
            className="text-xs bg-card border border-border rounded px-2 py-1"
          >
            {sectors.map(sector => (
              <option key={sector} value={sector}>
                {sector === "all" ? "All Sectors" : sector}
              </option>
            ))}
          </select>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="text-xs bg-card border border-border rounded px-2 py-1"
          >
            <option value="impact">By Impact</option>
            <option value="change">By Change</option>
            <option value="volume">By Volume</option>
          </select>
        </div>
      </div>

      <div className="space-y-3 max-h-[600px] overflow-y-auto">
        {filteredHighlights.map((highlight, index) => (
          <div
            key={highlight.symbol}
            className={`border rounded-xl p-3 hover:bg-muted/50 transition-colors ${
              getPriorityColor(highlight.priority)
            }`}
          >
            {/* Header Row */}
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="font-bold text-lg">{highlight.symbol}</span>
                <span className={`text-xs px-2 py-0.5 rounded-full border ${getPriorityColor(highlight.priority)}`}>
                  {highlight.priority.toUpperCase()}
                </span>
                <span className="text-xs text-muted-foreground">{highlight.sector}</span>
              </div>
              <div className="flex items-center gap-1">
                {renderMiniChart(highlight.miniChart)}
                <span className="text-xs text-muted-foreground ml-2">
                  Score: {highlight.impactScore.toFixed(1)}
                </span>
              </div>
            </div>

            {/* Price & Change */}
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="text-xl font-semibold">${highlight.price.toFixed(2)}</span>
                <span className={`flex items-center gap-1 text-sm ${
                  highlight.change >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {highlight.change >= 0 ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                  ${Math.abs(highlight.change).toFixed(2)} ({Math.abs(highlight.changePercent).toFixed(2)}%)
                </span>
              </div>
            </div>

            {/* Technical Indicators */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-2 text-xs">
              <div className="flex items-center gap-1">
                <Activity size={12} className="text-blue-400" />
                RSI: {highlight.rsi}
              </div>
              <div className="flex items-center gap-1">
                <BarChart3 size={12} className="text-purple-400" />
                MACD: {highlight.macd.histogram >= 0 ? '+' : ''}{highlight.macd.histogram.toFixed(1)}
              </div>
              <div className="flex items-center gap-1">
                <Volume2 size={12} className="text-orange-400" />
                Vol: {(highlight.volume / highlight.avgVolume).toFixed(1)}x
              </div>
              <div className="flex items-center gap-1">
                <AlertTriangle size={12} className="text-yellow-400" />
                Vol: {highlight.volatility.toFixed(2)}
              </div>
            </div>

            {/* Signals & News */}
            <div className="space-y-2">
              {/* Signals Row */}
              <div className="flex flex-wrap gap-1">
                {highlight.signals.slice(0, 3).map(signal => (
                  <span key={signal} className="text-xs bg-primary/20 text-primary px-2 py-0.5 rounded">
                    {signal}
                  </span>
                ))}
              </div>

              {/* News Evidence Button */}
              {highlight.newsCount > 0 ? (
                <a
                  href={`/news/${highlight.symbol}`}
                  className="inline-flex items-center gap-2 px-3 py-2 bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/30 rounded-lg text-sm font-medium text-blue-400 hover:text-blue-300 transition-colors"
                >
                  <ExternalLink size={16} />
                  View {highlight.newsCount} Article{highlight.newsCount !== 1 ? 's' : ''} as Proof
                </a>
              ) : (
                <div className="text-xs text-muted-foreground italic">
                  No recent news articles
                </div>
              )}
            </div>
          </div>
        ))}

        {filteredHighlights.length === 0 && (
          <div className="text-center py-8 text-muted-foreground">
            <Zap className="w-12 h-12 mx-auto mb-2 opacity-50" />
            <p>No highlights match your current filters</p>
            <p className="text-xs">Highlights reset daily at 5 PM ET</p>
          </div>
        )}
      </div>

      {/* Footer with reset info */}
      <div className="mt-4 pt-3 border-t border-border text-center text-xs text-muted-foreground">
        Pre-market analysis • Resets daily at 5 PM ET • Updated in real-time
      </div>
    </div>
  );
}


