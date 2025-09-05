import React, { useState } from 'react';
import { MarketQuote } from '@/components/market/MarketQuote';
import { MarketChart } from '@/components/market/MarketChart';
import { MarketQuoteList } from '@/components/market/MarketQuoteList';
import { PaperTradeButton } from '@/components/market/PaperTradeButton';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Timeframe } from '@/hooks/useBars';
import { useQuotesQuery } from '@/hooks/useQuotes';
import LiveMarketData from '@/components/LiveMarketData';

const DEFAULT_SYMBOLS = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN'];

const MarketDashboard: React.FC = () => {
  const [symbols, setSymbols] = useState<string[]>(DEFAULT_SYMBOLS);
  const [newSymbol, setNewSymbol] = useState('');
  const [selectedSymbol, setSelectedSymbol] = useState('SPY');
  const [timeframe, setTimeframe] = useState<Timeframe>('1Day');
  
  // Fetch all quotes in a single request
  const { data: quoteBatch } = useQuotesQuery(symbols);
  
  const handleAddSymbol = () => {
    if (!newSymbol) return;
    const symbol = newSymbol.trim().toUpperCase();
    if (symbol && !symbols.includes(symbol)) {
      setSymbols([...symbols, symbol]);
      setSelectedSymbol(symbol);
    }
    setNewSymbol('');
  };
  
  const handleRemoveSymbol = (symbol: string) => {
    setSymbols(symbols.filter(s => s !== symbol));
    if (selectedSymbol === symbol) {
      setSelectedSymbol(symbols[0] || '');
    }
  };
  
  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4 text-foreground">Market Dashboard</h1>
      
      {/* Live market data section */}
      <div className="mb-6">
        <LiveMarketData />
      </div>
      
      {/* Symbol selection and controls */}
      <div className="mb-4 flex flex-wrap gap-2">
        <div className="flex">
          <input
            type="text"
            value={newSymbol}
            onChange={(e) => setNewSymbol(e.target.value)}
            placeholder="Add symbol..."
            className="border border-border bg-card text-foreground placeholder:text-muted-foreground rounded-l px-3 py-2 w-32"
          />
          <button
            onClick={handleAddSymbol}
            className="bg-primary text-primary-foreground px-3 py-2 rounded-r"
          >
            Add
          </button>
        </div>
        
        <div className="flex gap-2">
          {symbols.map(symbol => (
            <button
              key={symbol}
              onClick={() => setSelectedSymbol(symbol)}
              className={`px-3 py-2 rounded flex items-center ${
                selectedSymbol === symbol
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted hover:bg-muted/80 text-foreground'
              }`}
            >
              {symbol}
              {symbols.length > 1 && (
                <span
                  className="ml-2 text-xs"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleRemoveSymbol(symbol);
                  }}
                >
                  ✕
                </span>
              )}
            </button>
          ))}
        </div>
        
        <div className="ml-auto">
          <select
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value as Timeframe)}
            className="border border-border bg-card text-foreground rounded px-3 py-2"
          >
            <option value="1Min">1 Minute</option>
            <option value="5Min">5 Minutes</option>
            <option value="15Min">15 Minutes</option>
            <option value="30Min">30 Minutes</option>
            <option value="1Hour">1 Hour</option>
            <option value="1Day">1 Day</option>
            <option value="1Week">1 Week</option>
            <option value="1Month">1 Month</option>
          </select>
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-6">
        {/* Main chart and quotes (3/4 width) */}
        <div className="lg:col-span-3 space-y-6">
          {/* Main chart for selected symbol */}
          <MarketChart
            symbol={selectedSymbol}
            timeframe={timeframe}
            height={400}
            title={`${selectedSymbol} Price Chart`}
          />
          
          {/* Quote cards grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {symbols.slice(0, 6).map(symbol => (
              <MarketQuote key={symbol} symbol={symbol} />
            ))}
          </div>
        </div>
        
        {/* Sidebar (1/4 width) */}
        <div className="space-y-6">
          {/* Watchlist */}
          <MarketQuoteList
            defaultSymbols={['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL']}
            title="Watchlist"
          />
          
          {/* Paper trade form */}
          <PaperTradeButton symbol={selectedSymbol} />
        </div>
      </div>
      
      {/* Batch data debug card */}
      <Card className="mb-4">
        <CardHeader>
          <CardTitle className="text-base">Batch Data Status</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-sm">
            <p>Symbols in batch: {symbols.join(', ')}</p>
            <p>Data received: {quoteBatch ? 'Yes' : 'No'}</p>
            <p>Quotes count: {quoteBatch?.quotes ? Object.keys(quoteBatch.quotes).length : 0}</p>
          </div>
          <div className="mt-2 p-2 bg-muted rounded text-xs overflow-auto max-h-40">
            <pre>{JSON.stringify(quoteBatch, null, 2)}</pre>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default MarketDashboard;
