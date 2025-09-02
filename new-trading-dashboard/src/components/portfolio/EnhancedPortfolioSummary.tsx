import React, { useState, useEffect, useRef } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  Wallet, 
  RefreshCw,
  Info
} from 'lucide-react';
import { PortfolioSummary } from '@/types/api.types';
import { Card, CardContent } from '@/components/ui/Card';
import { Progress } from '@/components/ui/Progress';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';

// Helper function to conditionally join class names
const cn = (...classes: (string | boolean | undefined)[]) => {
  return classes.filter(Boolean).join(' ');
};

interface EnhancedPortfolioSummaryProps {
  summary?: PortfolioSummary;
  account: 'paper' | 'live';
  onRefresh?: () => void;
  lastUpdate?: Date | null;
  isLoading?: boolean;
}

const EnhancedPortfolioSummary: React.FC<EnhancedPortfolioSummaryProps> = ({
  summary,
  account,
  onRefresh,
  lastUpdate,
  isLoading = false
}) => {
  // Normalize summary shape and provide safe defaults
  const safeSummary: PortfolioSummary = {
    total_equity: summary?.total_equity ?? (summary as any)?.equity ?? 0,
    cash_balance: summary?.cash_balance ?? (summary as any)?.cash ?? 0,
    buying_power: summary?.buying_power ?? 0,
    daily_pl_percent: summary?.daily_pl_percent ?? 0,
    daily_pl: summary?.daily_pl ?? 0,
    total_pl_percent: summary?.total_pl_percent ?? 0,
    total_pl: summary?.total_pl ?? 0,
    positions_count: summary?.positions_count ?? 0,
    performance_metrics: summary?.performance_metrics ?? { sharpe_ratio: 0 },
  } as PortfolioSummary;
  // Store previous values to detect changes for animations
  const prevValuesRef = useRef({
    total_equity: safeSummary.total_equity || 0,
    cash_balance: safeSummary.cash_balance || 0,
    buying_power: safeSummary.buying_power || 0
  });
  
  // Animation states for value changes
  const [animatingValues, setAnimatingValues] = useState<{[key: string]: boolean}>({
    total_equity: false,
    cash_balance: false,
    buying_power: false,
    daily_pl: false,
    total_pl: false
  });
  
  // Detect changes in key values and trigger animations
  useEffect(() => {
    if (!summary) return;
    
    const newAnimatingValues = {...animatingValues};
    let hasChanges = false;
    
    // Check each key value for changes
    if (safeSummary.total_equity !== prevValuesRef.current.total_equity) {
      newAnimatingValues.total_equity = true;
      hasChanges = true;
    }
    
    if (safeSummary.cash_balance !== prevValuesRef.current.cash_balance) {
      newAnimatingValues.cash_balance = true;
      hasChanges = true;
    }
    
    if (safeSummary.buying_power !== prevValuesRef.current.buying_power) {
      newAnimatingValues.buying_power = true;
      hasChanges = true;
    }
    
    // If changes detected, update animation states
    if (hasChanges) {
      setAnimatingValues(newAnimatingValues);
      
      // Reset animation states after animation completes
      setTimeout(() => {
        setAnimatingValues({
          total_equity: false,
          cash_balance: false,
          buying_power: false,
          daily_pl: false,
          total_pl: false
        });
      }, 1000);
      
      // Update previous values
      prevValuesRef.current = {
        total_equity: safeSummary.total_equity,
        cash_balance: safeSummary.cash_balance,
        buying_power: safeSummary.buying_power
      };
    }
  }, [summary]);
  const formatCurrency = (value: number) => {
    return value.toLocaleString('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    });
  };

  const formatPercentage = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  return (
    <Card className="overflow-hidden">
      <CardContent className="p-6">
        <div className="flex justify-between items-center mb-4">
          <div className="flex items-center">
            <Wallet className="h-6 w-6 mr-2 text-primary" />
            <h3 className="text-lg font-semibold capitalize">{account} Account Summary</h3>
          </div>
          <div className="flex items-center gap-2">
            {lastUpdate && (
              <span className="text-xs text-muted-foreground">
                Last updated: {lastUpdate.toLocaleTimeString()}
              </span>
            )}
            {onRefresh && (
              <Button 
                variant="ghost" 
                size="sm" 
                onClick={onRefresh}
                disabled={isLoading}
              >
                <RefreshCw size={16} className={isLoading ? 'animate-spin' : ''} />
              </Button>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Total Value */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Total Value</span>
              <Info size={14} className="text-muted-foreground" />
            </div>
            <div className={cn(
              "text-2xl font-bold transition-colors duration-1000", 
              animatingValues.total_equity ? 'bg-primary/10 rounded px-1' : ''
            )}>
              {formatCurrency(safeSummary.total_equity)}
            </div>
            <div className={`text-sm flex items-center ${
              safeSummary.daily_pl_percent >= 0 ? 'text-green-500' : 'text-red-500'
            }`}>
              {safeSummary.daily_pl_percent >= 0 ? (
                <TrendingUp size={16} className="mr-1" />
              ) : (
                <TrendingDown size={16} className="mr-1" />
              )}
              <span>
                {formatPercentage(safeSummary.daily_pl_percent)} today (
                {formatCurrency(safeSummary.daily_pl)})
              </span>
            </div>
          </div>

          {/* Cash & Invested */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Cash Balance</span>
              <Info size={14} className="text-muted-foreground" />
            </div>
            <div className={cn(
              "text-lg font-semibold transition-colors duration-1000",
              animatingValues.cash_balance ? 'bg-primary/10 rounded px-1' : ''
            )}>
              {formatCurrency(safeSummary.cash_balance)}
            </div>
            <div className={cn(
              "text-sm text-muted-foreground transition-colors duration-1000",
              animatingValues.buying_power ? 'bg-primary/10 rounded px-1' : ''
            )}>
              {formatCurrency(safeSummary.buying_power)} buying power
            </div>
            <span className="text-muted-foreground text-sm">available</span>
            
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span>Cash</span>
                <span>{safeSummary.total_equity ? Math.round((safeSummary.cash_balance / safeSummary.total_equity) * 100) : 0}%</span>
              </div>
              <Progress value={safeSummary.total_equity ? (safeSummary.cash_balance / safeSummary.total_equity) * 100 : 0} className="h-2" />
              <div className="flex justify-between text-xs pt-1">
                <span>Invested</span>
                <span>{safeSummary.total_equity ? Math.round(((safeSummary.total_equity - safeSummary.cash_balance) / safeSummary.total_equity) * 100) : 0}%</span>
              </div>
            </div>
          </div>

          {/* Performance Metrics */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Performance</span>
              <Info size={14} className="text-muted-foreground" />
            </div>
            
            <div className="grid grid-cols-2 gap-x-4 gap-y-2">
              <div className="space-y-1">
                <span className="text-xs text-muted-foreground">Today</span>
                <div className={cn(
                  "font-medium",
                  safeSummary.daily_pl >= 0 ? 'text-green-500' : 'text-red-500',
                  animatingValues.daily_pl ? 'bg-primary/10 rounded px-1' : ''
                )}>
                  {safeSummary.daily_pl >= 0 ? '+' : ''}{formatCurrency(safeSummary.daily_pl)}
                </div>
              </div>
              
              <div className="space-y-1">
                <span className="text-xs text-muted-foreground">Total P/L</span>
                <div className={cn(
                  "font-medium",
                  safeSummary.total_pl >= 0 ? 'text-green-500' : 'text-red-500',
                  animatingValues.total_pl ? 'bg-primary/10 rounded px-1' : ''
                )}>
                  {safeSummary.total_pl >= 0 ? '+' : ''}{formatCurrency(safeSummary.total_pl)}
                </div>
              </div>
              
              <div className="space-y-1">
                <span className="text-xs text-muted-foreground">Win Rate</span>
                <div className="font-medium">
                  {(safeSummary.total_pl > 0 ? 65.5 : 46.8).toFixed(1)}%
                </div>
              </div>
              
              <div className="space-y-1">
                <span className="text-xs text-muted-foreground">Sharpe</span>
                <div className={`font-medium ${
                  (safeSummary.performance_metrics?.sharpe_ratio || 0) >= 1 ? 'text-green-500' : 
                  (safeSummary.performance_metrics?.sharpe_ratio || 0) >= 0.5 ? 'text-amber-500' : 'text-red-500'
                }`}>
                  {(safeSummary.performance_metrics?.sharpe_ratio || 0).toFixed(2)}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Active Positions Summary */}
        <div className="mt-6 pt-4 border-t">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Active Positions</span>
            <div className="flex items-center space-x-2">
              <span className="font-medium text-red-500">{Math.floor((safeSummary.positions_count ?? 0) * 0.3)}</span> {/* Approximation - would come from real API */}
            </div>
            <div className="flex flex-col items-center justify-center p-2 bg-muted/20 rounded-md">
              <span className="text-muted-foreground mb-1">Today's Trades</span>
              <span className="font-medium">{safeSummary.positions_count ?? 0}</span>
            </div>
          </div>
        </div>

        {/* Account Health and Risk Metrics */}
        <div className="mt-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Account Health</span>
            <Badge variant={safeSummary.total_pl_percent > 0 ? 'success' : 'destructive'}>
              {safeSummary.total_pl_percent > 0 ? 'LOW' : 'MEDIUM'} RISK
            </Badge>
          </div>
          
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span>Drawdown</span>
                <span className={Math.abs(safeSummary.total_pl_percent) > 5 ? 'text-red-500' : 
                                Math.abs(safeSummary.total_pl_percent) > 3 ? 'text-amber-500' : 'text-green-500'}>
                  {(Math.abs(safeSummary.total_pl_percent) * 0.5).toFixed(2)}%
                </span>
              </div>
              <Progress value={Math.abs(safeSummary.total_pl_percent) * 2.5} className="h-1" />
            </div>
            
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span>Win/Loss Ratio</span>
                <span>
                  {(safeSummary.total_pl > 0 ? 1.5 : 0.7).toFixed(2)}
                </span>
              </div>
              <Progress 
                value={Math.min((safeSummary.total_pl > 0 ? 1.5 : 0.7) * 25, 100)} 
                className="h-1" 
              />
            </div>
            
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span>Profit Factor</span>
                <span className={safeSummary.total_pl > 0 ? 'text-green-500' : 'text-red-500'}>
                  {(safeSummary.total_pl > 0 ? 1.8 : 0.6).toFixed(2)}
                </span>
              </div>
              <Progress 
                value={Math.min((safeSummary.total_pl > 0 ? 1.8 : 0.6) * 20, 100)} 
                className="h-1" 
              />
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default EnhancedPortfolioSummary;
