import React, { useState } from 'react';
import { format } from 'date-fns';
import { Trade } from '@/types/api.types';
import { 
  ArrowUpIcon, 
  ArrowDownIcon, 
  Clock, 
  Tags, 
  TrendingUp, 
  AlertCircle,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/Tooltip';

interface EnhancedTradeHistoryProps {
  trades?: Trade[];
  limit?: number;
  account: 'paper' | 'live';
  lastUpdate?: Date | null;
}

const EnhancedTradeHistory: React.FC<EnhancedTradeHistoryProps> = ({
  trades = [],
  limit = 20,
  account,
  lastUpdate
}) => {
  const [expandedTrades, setExpandedTrades] = useState<Record<string, boolean>>({});
  const [showAll, setShowAll] = useState(false);

  const safeTrades = Array.isArray(trades) ? trades : [];
  const displayTrades = showAll ? safeTrades : safeTrades.slice(0, limit);

  const toggleTradeExpand = (tradeId: string) => {
    setExpandedTrades(prev => ({
      ...prev,
      [tradeId]: !prev[tradeId]
    }));
  };

  const getActionColor = (side: string) => {
    return side.toLowerCase() === 'buy' ? 'text-green-500' : 'text-red-500';
  };

  const getActionBgColor = (side: string) => {
    return side.toLowerCase() === 'buy' ? 'bg-green-100 dark:bg-green-900/20' : 'bg-red-100 dark:bg-red-900/20';
  };

  return (
    <div className="space-y-4">
      {lastUpdate && (
        <div className="flex justify-end">
          <span className="text-xs text-muted-foreground">
            Last updated: {lastUpdate.toLocaleTimeString()}
          </span>
        </div>
      )}

      <div className="space-y-3 max-h-[600px] overflow-y-auto pr-2">
        {displayTrades.map(trade => (
          <Card 
            key={trade.id} 
            className={`p-4 border transition hover:shadow-sm cursor-pointer ${
              expandedTrades[trade.id] ? 'bg-muted/10' : ''
            }`}
            onClick={() => toggleTradeExpand(trade.id)}
          >
            <div className="flex items-start justify-between">
              <div className="flex items-start space-x-3">
                <div className={`p-2 rounded-full ${getActionBgColor(trade.side)}`}>
                  {trade.side.toLowerCase() === 'buy' ? (
                    <ArrowUpIcon className={getActionColor(trade.side)} size={16} />
                  ) : (
                    <ArrowDownIcon className={getActionColor(trade.side)} size={16} />
                  )}
                </div>
                
                <div>
                  <div className="flex items-center">
                    <span className={`font-medium ${getActionColor(trade.side)}`}>
                      {trade.side.toUpperCase()}
                    </span>
                    <span className="mx-1">•</span>
                    <span className="font-medium">{trade.symbol}</span>
                    
                    {trade.status === 'filled' && (
                      <Badge 
                        variant="outline" 
                        className="ml-2 bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300"
                      >
                        Filled
                      </Badge>
                    )}
                    
                    {trade.status === 'partial' && (
                      <Badge 
                        variant="outline" 
                        className="ml-2 bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300"
                      >
                        Partial
                      </Badge>
                    )}
                    
                    {trade.status === 'canceled' && (
                      <Badge 
                        variant="outline" 
                        className="ml-2 bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300"
                      >
                        Canceled
                      </Badge>
                    )}

                    {trade.is_market_order && (
                      <Badge 
                        variant="outline" 
                        className="ml-2"
                      >
                        Market
                      </Badge>
                    )}
                  </div>
                  
                  <div className="flex items-center text-sm text-muted-foreground">
                    <Clock size={14} className="mr-1" />
                    {(() => {
                      const iso = (trade as any).execution_time || (trade as any).timestamp || (trade as any).time;
                      const d = iso ? new Date(iso) : null;
                      return d && !isNaN(d.getTime())
                        ? format(d, 'MMM d, yyyy h:mm:ss a')
                        : '—';
                    })()}
                  </div>
                </div>
              </div>
              
              <div className="flex flex-col items-end">
                <div className="flex items-center">
                  <span className="font-medium mr-1">
                    {trade.quantity} @ ${trade.price.toFixed(2)}
                  </span>
                  <span className="text-muted-foreground">= ${(trade.quantity * trade.price).toFixed(2)}</span>
                </div>
                
                <div className="flex items-center mt-1">
                  {expandedTrades[trade.id] ? (
                    <ChevronUp size={16} className="text-muted-foreground" />
                  ) : (
                    <ChevronDown size={16} className="text-muted-foreground" />
                  )}
                </div>
              </div>
            </div>
            
            {expandedTrades[trade.id] && (
              <div className="mt-4 pt-3 border-t grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <h4 className="text-sm font-medium mb-1 flex items-center">
                    <Tags size={14} className="mr-1" />
                    Strategy
                  </h4>
                  {trade.strategy_name ? (
                    <div className="space-y-1">
                      <Badge variant="secondary" className="mr-1">
                        {trade.strategy_name}
                      </Badge>
                      {trade.strategy_type && (
                        <div className="text-xs text-muted-foreground">
                          Type: {trade.strategy_type}
                        </div>
                      )}
                    </div>
                  ) : (
                    <span className="text-sm text-muted-foreground">Manual trade</span>
                  )}
                </div>
                
                <div>
                  <h4 className="text-sm font-medium mb-1 flex items-center">
                    <TrendingUp size={14} className="mr-1" />
                    Details
                  </h4>
                  <div className="space-y-1 text-sm">
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Order Type:</span>
                      <span>{trade.order_type}</span>
                    </div>
                    {trade.commission !== undefined && (
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">Commission:</span>
                        <span>${trade.commission.toFixed(2)}</span>
                      </div>
                    )}
                    {trade.total_cost !== undefined && (
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">Total:</span>
                        <span>${trade.total_cost.toFixed(2)}</span>
                      </div>
                    )}
                  </div>
                </div>
                
                <div>
                  <h4 className="text-sm font-medium mb-1 flex items-center">
                    <AlertCircle size={14} className="mr-1" />
                    Reason
                  </h4>
                  <div className="text-sm">
                    {trade.entry_reason ? (
                      <div className="bg-muted/20 p-2 rounded-md text-xs">
                        {trade.entry_reason}
                      </div>
                    ) : (
                      <span className="text-muted-foreground">No reason provided</span>
                    )}
                  </div>
                </div>
              </div>
            )}
          </Card>
        ))}
      </div>

      {trades.length > limit && !showAll && (
        <Button 
          variant="outline" 
          className="w-full"
          onClick={() => setShowAll(true)}
        >
          Show All ({trades.length}) Trades
        </Button>
      )}
      
      {showAll && trades.length > limit && (
        <Button 
          variant="outline" 
          className="w-full"
          onClick={() => setShowAll(false)}
        >
          Show Less
        </Button>
      )}
      
      {trades.length === 0 && (
        <div className="py-12 flex flex-col items-center justify-center space-y-2">
          <AlertCircle size={32} className="text-muted-foreground" />
          <p className="text-muted-foreground">No trades in your {account} account</p>
        </div>
      )}
    </div>
  );
};

export default EnhancedTradeHistory;
