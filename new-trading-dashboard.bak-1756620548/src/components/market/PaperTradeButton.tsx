import React, { useState } from 'react';
import { api } from '@/services/api';
import { useQueryClient } from '@tanstack/react-query';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { toast } from 'react-hot-toast';

interface PaperTradeButtonProps {
  symbol?: string;
  className?: string;
}

export const PaperTradeButton: React.FC<PaperTradeButtonProps> = ({
  symbol: defaultSymbol = 'SPY',
  className
}) => {
  const queryClient = useQueryClient();
  const [symbol, setSymbol] = useState(defaultSymbol);
  const [qty, setQty] = useState(1);
  const [side, setSide] = useState<'buy' | 'sell'>('buy');
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  const handleSubmit = async () => {
    if (!symbol || qty <= 0) return;
    
    setIsSubmitting(true);
    
    try {
      const response = await api.post('/live/order', {
        symbol: symbol.toUpperCase(),
        side,
        qty,
        type: 'market'
      });
      
      // Invalidate relevant queries to refresh data
      queryClient.invalidateQueries({ queryKey: ['orders'] });
      queryClient.invalidateQueries({ queryKey: ['positions'] });
      
      toast.success(`${side === 'buy' ? 'Bought' : 'Sold'} ${qty} ${symbol}`);
      console.log('Order submitted:', response.data);
    } catch (error) {
      console.error('Error submitting order:', error);
      toast.error(`Order failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsSubmitting(false);
    }
  };
  
  return (
    <Card className={className}>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Paper Trade</CardTitle>
      </CardHeader>
      
      <CardContent className="p-4">
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">Symbol</label>
            <input
              type="text"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              className="border rounded px-3 py-2 w-full"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-1">Quantity</label>
            <input
              type="number"
              value={qty}
              min={1}
              step={1}
              onChange={(e) => setQty(Math.max(1, parseInt(e.target.value) || 1))}
              className="border rounded px-3 py-2 w-full"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-1">Side</label>
            <div className="flex gap-2">
              <button
                onClick={() => setSide('buy')}
                className={`flex-1 py-2 rounded ${
                  side === 'buy'
                    ? 'bg-green-500 text-white'
                    : 'bg-gray-100 hover:bg-gray-200'
                }`}
              >
                Buy
              </button>
              <button
                onClick={() => setSide('sell')}
                className={`flex-1 py-2 rounded ${
                  side === 'sell'
                    ? 'bg-red-500 text-white'
                    : 'bg-gray-100 hover:bg-gray-200'
                }`}
              >
                Sell
              </button>
            </div>
          </div>
          
          <button
            onClick={handleSubmit}
            disabled={isSubmitting}
            className={`w-full py-2 rounded ${
              side === 'buy'
                ? 'bg-green-600 hover:bg-green-700 text-white'
                : 'bg-red-600 hover:bg-red-700 text-white'
            } ${isSubmitting ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            {isSubmitting
              ? 'Submitting...'
              : `${side === 'buy' ? 'Buy' : 'Sell'} ${qty} ${symbol}`}
          </button>
          
          <div className="text-xs text-gray-500 text-center">
            Paper trading - no real money will be used
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default PaperTradeButton;
