import React from 'react';
import { 
  ArrowUp, 
  ArrowDown,
  DollarSign,
  Clock,
  BarChart2,
  Info,
  TrendingUp,
  Zap
} from 'lucide-react';
import { Position } from '@/types/api.types';
import { StatusBadge } from '@/components/ui/StatusBadge';

interface PositionDetailProps {
  position: Position;
  onClose: (symbol: string) => void;
}

const PositionDetail: React.FC<PositionDetailProps> = ({ position, onClose }) => {
  const isProfit = (position.lastPrice - position.avgCost) >= 0;
  const unrealizedPL = (position.lastPrice - position.avgCost) * position.quantity;
  const unrealizedPLPercent = ((position.lastPrice - position.avgCost) / position.avgCost) * 100;
  const marketValue = position.lastPrice * position.quantity;
  
  // Helper function to format currency
  const formatCurrency = (value: number) => {
    return value.toLocaleString(undefined, { 
      minimumFractionDigits: 2, 
      maximumFractionDigits: 2 
    });
  };
  
  return (
    <div className="bg-card border border-border rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border">
        <div className="flex items-center">
          <div className="mr-3">
            <StatusBadge 
              status={isProfit ? 'success' : 'error'} 
              label={position.symbol}
              size="lg"
            />
          </div>
          
          <div>
            <h3 className="font-medium text-lg">{position.name || position.symbol}</h3>
            <div className="text-sm text-muted-foreground">
              {position.sector ? `${position.sector} • ` : ''}
              {position.market || 'Stock'}
            </div>
          </div>
        </div>
        
        <button
          onClick={() => onClose(position.symbol)}
          className="py-1.5 px-3 bg-bear hover:bg-bear/90 text-white rounded-md text-sm flex items-center"
        >
          Close Position
        </button>
      </div>
      
      {/* Position Details */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-6 p-4">
        <div>
          <div className="text-xs text-muted-foreground flex items-center mb-1">
            <DollarSign size={12} className="mr-1" /> Market Value
          </div>
          <div className="text-xl font-bold">${formatCurrency(marketValue)}</div>
        </div>
        
        <div>
          <div className="text-xs text-muted-foreground flex items-center mb-1">
            <Info size={12} className="mr-1" /> Quantity
          </div>
          <div className="text-xl font-bold">{position.quantity}</div>
        </div>
        
        <div>
          <div className="text-xs text-muted-foreground flex items-center mb-1">
            <TrendingUp size={12} className="mr-1" /> Average Cost
          </div>
          <div className="text-xl font-bold">${formatCurrency(position.avgCost)}</div>
        </div>
        
        <div>
          <div className="text-xs text-muted-foreground flex items-center mb-1">
            <BarChart2 size={12} className="mr-1" /> Current Price
          </div>
          <div className="text-xl font-bold">${formatCurrency(position.lastPrice)}</div>
        </div>
      </div>
      
      {/* P&L Section */}
      <div className={`p-4 ${isProfit ? 'bg-bull/10' : 'bg-bear/10'} border-t border-border`}>
        <div className="flex flex-col md:flex-row md:items-center justify-between">
          <div>
            <div className="text-xs text-muted-foreground mb-1">Unrealized P&L</div>
            <div className={`text-xl font-bold flex items-center ${isProfit ? 'text-bull' : 'text-bear'}`}>
              {isProfit ? <ArrowUp size={20} className="mr-1" /> : <ArrowDown size={20} className="mr-1" />}
              ${formatCurrency(Math.abs(unrealizedPL))}
              <span className="ml-2 text-sm">
                ({unrealizedPLPercent >= 0 ? '+' : ''}{unrealizedPLPercent.toFixed(2)}%)
              </span>
            </div>
          </div>
          
          {position.strategyName && (
            <div className="mt-3 md:mt-0 bg-card/50 p-2 rounded-md flex items-center">
              <Zap size={16} className="mr-2 text-primary" />
              <div>
                <div className="text-xs text-muted-foreground">Strategy</div>
                <div className="font-medium">{position.strategyName}</div>
              </div>
            </div>
          )}
        </div>
      </div>
      
      {/* Additional details if available */}
      {position.lastUpdate && (
        <div className="p-3 text-xs text-muted-foreground border-t border-border flex items-center">
          <Clock size={12} className="mr-1" />
          Last updated: {new Date(position.lastUpdate).toLocaleString()}
        </div>
      )}
    </div>
  );
};

export default PositionDetail;
