import React from 'react';
import { 
  ArrowUpRight, 
  ArrowDownRight,
  Clock,
  Zap,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import { Trade } from '@/types/api.types';
import { formatRelativeTime } from '@/utils/date';

interface TradeHistoryItemProps {
  trade: Trade;
  expanded?: boolean;
  onToggleExpand?: () => void;
}

const TradeHistoryItem: React.FC<TradeHistoryItemProps> = ({ 
  trade, 
  expanded = false,
  onToggleExpand
}) => {
  const isBuy = trade.action === 'buy';
  
  return (
    <div className="border border-border rounded-lg bg-card mb-2 overflow-hidden">
      <div 
        className="flex items-center p-3 cursor-pointer hover:bg-muted/20"
        onClick={onToggleExpand}
      >
        {/* Trade action and symbol */}
        <div className={`flex items-center p-1.5 rounded-md ${isBuy ? 'bg-bull/10 text-bull' : 'bg-bear/10 text-bear'} mr-4`}>
          {isBuy ? <ArrowUpRight size={16} /> : <ArrowDownRight size={16} />}
          <span className="ml-1 font-medium">{trade.action.toUpperCase()}</span>
        </div>
        
        <div className="font-medium text-lg">{trade.symbol}</div>
        
        {/* Quantity and price */}
        <div className="ml-4 text-sm">
          <span className="text-muted-foreground mr-1">Qty:</span>
          <span>{trade.quantity}</span>
        </div>
        
        <div className="ml-4 text-sm">
          <span className="text-muted-foreground mr-1">@</span>
          <span>${trade.price.toFixed(2)}</span>
        </div>
        
        {/* Strategy tag */}
        {trade.strategy && (
          <div className="ml-4 px-2 py-0.5 rounded-full bg-primary/10 text-primary text-xs">
            {trade.strategy}
          </div>
        )}
        
        {/* Time */}
        <div className="ml-auto flex items-center text-xs text-muted-foreground">
          <Clock size={12} className="mr-1" />
          <span>{formatRelativeTime(new Date(trade.timestamp))}</span>
        </div>
        
        <div className="ml-3">
          {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </div>
      </div>
      
      {/* Expanded details */}
      {expanded && (
        <div className="px-4 py-3 border-t border-border bg-muted/5">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-3">
            <div>
              <div className="text-xs text-muted-foreground mb-1">Order ID</div>
              <div className="text-sm">{trade.id}</div>
            </div>
            
            <div>
              <div className="text-xs text-muted-foreground mb-1">Total Value</div>
              <div className="text-sm">${(trade.quantity * trade.price).toLocaleString(undefined, {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
              })}</div>
            </div>
            
            <div>
              <div className="text-xs text-muted-foreground mb-1">Account</div>
              <div className="text-sm capitalize">{trade.account}</div>
            </div>
          </div>
          
          {/* Strategy details */}
          {trade.strategy && (
            <div className="mt-3">
              <div className="text-xs text-muted-foreground mb-1 flex items-center">
                <Zap size={12} className="mr-1" />
                Strategy Signal
              </div>
              <div className="text-sm bg-primary/5 p-2 rounded-md">
                {trade.strategySignal || `${trade.strategy} generated a ${trade.action.toUpperCase()} signal for ${trade.symbol}`}
              </div>
            </div>
          )}
          
          {/* Execution details */}
          <div className="mt-3">
            <div className="text-xs text-muted-foreground mb-1">Execution Details</div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              <div>
                <div className="text-xs text-muted-foreground">Order Type</div>
                <div>{trade.orderType || 'Market'}</div>
              </div>
              
              <div>
                <div className="text-xs text-muted-foreground">Status</div>
                <div className="capitalize">{trade.status || 'Filled'}</div>
              </div>
              
              <div>
                <div className="text-xs text-muted-foreground">Fees</div>
                <div>${trade.fees?.toFixed(2) || '0.00'}</div>
              </div>
              
              <div>
                <div className="text-xs text-muted-foreground">Filled At</div>
                <div>{new Date(trade.timestamp).toLocaleString()}</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TradeHistoryItem;
