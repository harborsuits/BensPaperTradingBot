import React from 'react';
import { useTrades } from '../../hooks/useTrades';
import { Spinner } from '../ui/Spinner';
import { ErrorMessage } from '../ui/ErrorMessage';

export function RecentTradesTable() {
  const { data, isLoading, error, refetch } = useTrades();
  
  if (isLoading) return <Spinner />;
  if (error) return <ErrorMessage retry={() => refetch()} />;
  
  return (
    <div className="recent-trades bg-card border border-border rounded-lg p-4">
      <h3 className="text-lg font-bold mb-3 text-white">Recent Trades</h3>
      
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border">
              <th className="text-left py-2 px-2 font-medium text-white">Symbol</th>
              <th className="text-left py-2 px-2 font-medium text-white">Date</th>
              <th className="text-left py-2 px-2 font-medium text-white">Type</th>
              <th className="text-right py-2 px-2 font-medium text-white">Qty</th>
              <th className="text-right py-2 px-2 font-medium text-white">Price</th>
              <th className="text-right py-2 px-2 font-medium text-white">Total</th>
            </tr>
          </thead>
          <tbody>
            {data?.trades?.map((trade) => (
              <tr 
                key={trade.id} 
                className="border-b border-border hover:bg-muted/20"
              >
                <td className="py-2 px-2 font-medium text-white">{trade.symbol}</td>
                <td className="py-2 px-2 text-white">{trade.date}</td>
                <td className="py-2 px-2">
                  <span className={`px-2 py-0.5 rounded-full text-xs ${
                    trade.type === 'BUY' 
                      ? 'bg-green-500/20 text-green-600' 
                      : 'bg-red-500/20 text-red-600'
                  }`}>
                    {trade.type}
                  </span>
                </td>
                <td className="py-2 px-2 text-right text-white">{trade.quantity}</td>
                <td className="py-2 px-2 text-right text-white">${trade.price.toFixed(2)}</td>
                <td className="py-2 px-2 text-right text-white">${trade.total.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      <div className="mt-3 flex justify-center">
        <button className="text-xs text-primary hover:underline">
          View All Trades
        </button>
      </div>
    </div>
  );
} 