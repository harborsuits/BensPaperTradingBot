import React from 'react';
import { usePortfolio } from '../../hooks/usePortfolio';
import { Spinner } from '../ui/Spinner';
import { ErrorMessage } from '../ui/ErrorMessage';

export function PortfolioOverview() {
  const { data, isLoading, error, refetch } = usePortfolio();
  
  if (isLoading) return <Spinner />;
  if (error) return <ErrorMessage retry={() => refetch()} />;
  
  // Calculate additional metrics from data
  const todayPnl = data?.currentValue ? (data.currentValue - data.initialValue) : 0;
  const todayPnlPercent = data?.initialValue ? (todayPnl / data.initialValue) * 100 : 0;
  
  return (
    <div className="portfolio-overview bg-card border border-border rounded-lg p-4">
      <h2 className="text-xl font-bold mb-4 text-white">Portfolio Overview</h2>
      
      <div className="stats-grid grid grid-cols-4 gap-4">
        <div className="stat-card bg-muted p-4 rounded-md">
          <h4 className="text-sm text-muted-foreground mb-1">Total Value</h4>
          <p className="text-2xl font-semibold text-white">${data?.currentValue?.toLocaleString() || '0.00'}</p>
          <span className={`text-xs ${data?.percentChange && data.percentChange >= 0 
            ? 'text-green-500' 
            : 'text-red-500'}`}>
            {data?.percentChange >= 0 ? '+' : ''}{data?.percentChange?.toFixed(2) || 0}%
          </span>
        </div>
        
        <div className="stat-card bg-muted p-4 rounded-md">
          <h4 className="text-sm text-muted-foreground mb-1">Daily P&L</h4>
          <p className="text-2xl font-semibold text-white">
            ${todayPnl.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}
          </p>
          <span className={`text-xs ${todayPnlPercent >= 0 
            ? 'text-green-500' 
            : 'text-red-500'}`}>
            {todayPnlPercent >= 0 ? '+' : ''}{todayPnlPercent.toFixed(2)}%
          </span>
        </div>
        
        <div className="stat-card bg-muted p-4 rounded-md">
          <h4 className="text-sm text-muted-foreground mb-1">Initial Investment</h4>
          <p className="text-2xl font-semibold text-white">${data?.initialValue?.toLocaleString() || '0.00'}</p>
          <span className="text-xs text-muted-foreground">
            {data?.startDate || 'Jan 1, 2024'}
          </span>
        </div>
        
        <div className="stat-card bg-muted p-4 rounded-md">
          <h4 className="text-sm text-muted-foreground mb-1">Performance</h4>
          <p className="text-2xl font-semibold text-white">{data?.percentChange || 0}%</p>
          <span className="text-xs text-muted-foreground">
            YTD
          </span>
        </div>
      </div>
    </div>
  );
} 