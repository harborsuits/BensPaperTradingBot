import React, { useState } from 'react';
import { usePrices } from '@/hooks/useQuotes';
import { fmtNum } from '@/utils/number';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Skeleton } from '@/components/ui/Skeleton';
import { ZeroState } from '@/components/ui/ZeroState';
import { RefreshCw, Plus, X } from 'lucide-react';

interface MarketQuoteListProps {
  defaultSymbols?: string[];
  title?: string;
  className?: string;
  refreshInterval?: number;
}

export const MarketQuoteList: React.FC<MarketQuoteListProps> = ({
  defaultSymbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL'],
  title = 'Market Quotes',
  className,
  refreshInterval = 3000
}) => {
  const [symbols, setSymbols] = useState<string[]>(defaultSymbols);
  const [newSymbol, setNewSymbol] = useState('');
  
  const { prices, isLoading, refetch } = usePrices(symbols, refreshInterval);
  
  const handleAddSymbol = () => {
    if (!newSymbol) return;
    
    const symbol = newSymbol.trim().toUpperCase();
    if (symbol && !symbols.includes(symbol)) {
      setSymbols([...symbols, symbol]);
    }
    setNewSymbol('');
  };
  
  const handleRemoveSymbol = (symbolToRemove: string) => {
    setSymbols(symbols.filter(s => s !== symbolToRemove));
  };
  
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleAddSymbol();
    }
  };
  
  return (
    <Card className={className}>
      <CardHeader className="pb-2">
        <CardTitle className="text-base flex items-center justify-between">
          <span>{title}</span>
          <button
            onClick={() => refetch()}
            className="p-1 rounded-full bg-primary/10 hover:bg-primary/20 text-primary transition-colors"
            title="Refresh quotes"
          >
            <RefreshCw size={14} />
          </button>
        </CardTitle>
      </CardHeader>
      
      <CardContent className="p-4">
        {/* Symbol input */}
        <div className="flex mb-4">
          <input
            type="text"
            value={newSymbol}
            onChange={(e) => setNewSymbol(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Add symbol..."
            className="border border-border bg-card text-foreground placeholder:text-muted-foreground rounded-l px-3 py-2 w-full"
          />
          <button
            onClick={handleAddSymbol}
            className="bg-primary text-primary-foreground px-3 py-2 rounded-r"
          >
            <Plus size={16} />
          </button>
        </div>
        
        {/* Loading state */}
        {isLoading && symbols.length > 0 && !prices && (
          <div className="space-y-2">
            {symbols.map(symbol => (
              <div key={symbol} className="flex items-center justify-between p-2 border border-border rounded bg-card">
                <div className="font-medium">{symbol}</div>
                <Skeleton className="h-6 w-20" />
              </div>
            ))}
          </div>
        )}
        
        {/* Empty state */}
        {symbols.length === 0 && (
          <ZeroState
            title="No symbols added"
            message="Add a symbol to see live quotes"
          />
        )}
        
        {/* Quotes list */}
        {!isLoading && prices && (
          <div className="space-y-2">
            {symbols.map(symbol => {
              const price = prices[symbol];
              
              return (
                <div 
                  key={symbol} 
                  className="flex items-center justify-between p-2 border border-border rounded hover:bg-muted bg-card"
                >
                  <div className="font-medium">{symbol}</div>
                  <div className="flex items-center gap-3">
                    <div className="text-lg font-bold">
                      {price !== undefined ? `$${fmtNum(price, 2)}` : '—'}
                    </div>
                    <button
                      onClick={() => handleRemoveSymbol(symbol)}
                      className="p-1 rounded-full hover:bg-muted text-muted-foreground"
                      title="Remove symbol"
                    >
                      <X size={14} />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default MarketQuoteList;
