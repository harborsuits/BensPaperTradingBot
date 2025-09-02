import React, { useState, useEffect, useMemo } from 'react';
import { useWebSocketContext } from '../contexts/WebSocketContext';
import { usePrices } from '../hooks/useQuotes';
import { fmtNum } from '@/utils/number';

// Default symbols to show if no websocket data
const DEFAULT_SYMBOLS = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL'];

const LiveMarketData: React.FC = () => {
  const { isConnected, lastMessage } = useWebSocketContext();
  
  type MarketEntry = { price: number; volume: number };
  type MarketWsPayload = { market_data: Record<string, MarketEntry>; timestamp: number | string } | null;
  
  const marketData: MarketWsPayload = useMemo(() => {
    if (lastMessage?.type === 'market_data') {
      return lastMessage.data as MarketWsPayload;
    }
    return null;
  }, [lastMessage]);
  const [symbols, setSymbols] = useState<string[]>(DEFAULT_SYMBOLS);
  
  // Use REST API as fallback when WebSocket is not connected
  const { prices, isLoading } = usePrices(
    !isConnected || !marketData ? symbols : [],
    2000 // Poll every 2 seconds
  );
  
  // Update symbols list if websocket provides different symbols
  useEffect(() => {
    if (isConnected && marketData?.market_data) {
      const wsSymbols = Object.keys(marketData.market_data);
      if (wsSymbols.length > 0) {
        setSymbols(wsSymbols);
      }
    }
  }, [isConnected, marketData]);
  
  // Show data from WebSocket if available, otherwise from REST API
  if (isConnected && marketData) {
    return (
      <div className="p-4 border rounded-lg bg-card border-border text-foreground">
        <div className="flex justify-between items-center mb-3">
          <h2 className="text-lg font-semibold">Live Market Data (WebSocket)</h2>
          <div className="text-xs text-muted-foreground">
            {(() => {
              const ts = typeof marketData.timestamp === 'number'
                ? new Date(marketData.timestamp * 1000)
                : new Date(marketData.timestamp);
              return `Last update: ${ts.toLocaleTimeString()}`;
            })()}
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Object.entries(marketData.market_data as Record<string, MarketEntry>).map(([symbol, data]) => (
            <div key={symbol} className="border rounded-md p-3 shadow-sm bg-card border-border">
              <div className="flex justify-between items-center">
                <h3 className="font-semibold text-foreground/90">{symbol}</h3>
                <div className="text-xs bg-muted px-2 py-1 rounded text-foreground">
                  Vol: {data.volume.toLocaleString()}
                </div>
              </div>
              <div className="text-2xl font-bold mt-2 text-foreground">
                ${fmtNum(data.price, 2)}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }
  
  // Show REST API data as fallback
  return (
    <div className="p-4 border rounded-lg bg-card border-border text-foreground">
      <div className="flex justify-between items-center mb-3">
        <h2 className="text-lg font-semibold">Live Market Data (REST API)</h2>
        <div className="text-xs text-muted-foreground">Last update: {new Date().toLocaleTimeString()}</div>
      </div>
      
      {isLoading && (
        <div className="text-muted-foreground mb-4">Loading latest prices...</div>
      )}
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {symbols.map(symbol => {
          const price = prices?.[symbol];
          
          return (
            <div key={symbol} className="border rounded-md p-3 shadow-sm bg-card border-border">
              <div className="flex justify-between items-center">
                <h3 className="font-semibold text-foreground/90">{symbol}</h3>
                <div className="text-xs bg-muted px-2 py-1 rounded text-foreground">
                  {price ? 'Live' : 'Loading'}
                </div>
              </div>
              <div className="text-2xl font-bold mt-2 text-foreground">
                {price ? `$${fmtNum(price, 2)}` : '—'}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default LiveMarketData;
