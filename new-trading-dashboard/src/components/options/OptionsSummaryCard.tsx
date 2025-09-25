import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Link } from 'react-router-dom';
import { ArrowRight, TrendingUp, TrendingDown, Activity } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useOptionsMetrics, useOptionsPositions } from '@/hooks/useOptionsData';
import { useQuery } from '@tanstack/react-query';
import { get } from '@/lib/api';

export default function OptionsSummaryCard() {
  const { data: metrics, isLoading: metricsLoading } = useOptionsMetrics();
  const { data: positions, isLoading: positionsLoading } = useOptionsPositions();
  
  // Calculate today's P&L from recent trades
  const { data: trades } = useQuery({
    queryKey: ['trades', 'today'],
    queryFn: async () => {
      const data = await get<any>('/api/trades');
      return data?.items || [];
    }
  });
  
  const todayPnL = React.useMemo(() => {
    if (!trades) return 0;
    const today = new Date().toDateString();
    return trades
      .filter(t => new Date(t.ts).toDateString() === today)
      .filter(t => t.metadata?.isOption || /[A-Z]+\d{6}[CP]\d{8}/.test(t.symbol))
      .reduce((sum, t) => sum + (t.pnl || 0), 0);
  }, [trades]);
  
  const todayPnLPercent = metrics?.capitalAllocated ? (todayPnL / metrics.capitalAllocated) * 100 : 0;
  
  // Find next expiration
  const nextExpiration = React.useMemo(() => {
    if (!positions || positions.length === 0) return null;
    
    const sorted = [...positions].sort((a, b) => 
      new Date(a.expiration).getTime() - new Date(b.expiration).getTime()
    );
    
    const next = sorted[0];
    if (!next) return null;
    
    const daysToExpiry = Math.ceil(
      (new Date(next.expiration).getTime() - Date.now()) / (1000 * 60 * 60 * 24)
    );
    
    return {
      symbol: `${next.underlying} ${next.strike}${next.optionType[0].toUpperCase()}`,
      daysToExpiry,
      contracts: Math.abs(next.quantity)
    };
  }, [positions]);
  
  const isLoading = metricsLoading || positionsLoading;

  if (isLoading || !metrics) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            Options
            <Activity className="w-4 h-4 text-gray-400" />
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-2">
            <div className="h-4 bg-gray-200 rounded w-1/2"></div>
            <div className="h-8 bg-gray-200 rounded"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="hover:shadow-lg transition-shadow">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Options Trading</span>
          <Link to="/options" className="text-blue-600 hover:text-blue-700">
            <ArrowRight className="w-4 h-4" />
          </Link>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Main P&L Display */}
        <div>
          <p className="text-sm text-gray-500">Today's P&L</p>
          <div className="flex items-baseline gap-2">
            <span className={cn(
              "text-2xl font-bold",
              todayPnL >= 0 ? "text-green-600" : "text-red-600"
            )}>
              ${Math.abs(todayPnL).toFixed(2)}
            </span>
            <span className={cn(
              "text-sm flex items-center",
              todayPnLPercent >= 0 ? "text-green-600" : "text-red-600"
            )}>
              {todayPnLPercent >= 0 ? (
                <TrendingUp className="w-3 h-3 mr-1" />
              ) : (
                <TrendingDown className="w-3 h-3 mr-1" />
              )}
              {Math.abs(todayPnLPercent).toFixed(2)}%
            </span>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-2 gap-3">
          <div>
            <p className="text-xs text-gray-500">Active Positions</p>
            <p className="text-lg font-medium">{metrics.openPositions}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500">Total Contracts</p>
            <p className="text-lg font-medium">
              {metrics.totalContracts}
            </p>
          </div>
        </div>

        {/* Premium Collected */}
        <div className="pt-3 border-t">
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-500">Premium Collected</span>
            <span className="font-medium">${metrics.totalPremiumCollected.toFixed(0)}</span>
          </div>
        </div>

        {/* Next Expiration Alert */}
        {nextExpiration && (
          <div className="bg-amber-50 p-3 rounded">
            <p className="text-xs text-amber-800 font-medium">Next Expiration</p>
            <p className="text-sm mt-1">
              {nextExpiration.symbol} ({nextExpiration.contracts} contracts)
            </p>
            <p className="text-xs text-amber-600 mt-1">
              {nextExpiration.daysToExpiry} days
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
