import React from 'react';
import { Card, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import ModeLabel, { TradingMode } from '@/components/ui/ModeLabel';
import {
  Flask,
  Target,
  CheckCircle,
  XCircle,
  AlertTriangle,
  DollarSign,
  TrendingUp,
  TrendingDown,
  Activity
} from 'lucide-react';

interface Trade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  pnl?: number;
  timestamp: Date;
  mode: TradingMode;
  strategy?: string;
  status: 'filled' | 'pending' | 'cancelled' | 'failed';
  experimentId?: string;
}

interface TradeModeIndicatorProps {
  trades: Trade[];
  showSummary?: boolean;
  className?: string;
}

const TradeModeIndicator: React.FC<TradeModeIndicatorProps> = ({
  trades,
  showSummary = true,
  className = ''
}) => {
  const getTradeStats = (mode: TradingMode) => {
    const modeTrades = trades.filter(t => t.mode === mode);
    const totalPnl = modeTrades.reduce((sum, t) => sum + (t.pnl || 0), 0);
    const winRate = modeTrades.length > 0
      ? modeTrades.filter(t => (t.pnl || 0) > 0).length / modeTrades.length
      : 0;

    return {
      count: modeTrades.length,
      totalPnl,
      winRate,
      avgPnl: modeTrades.length > 0 ? totalPnl / modeTrades.length : 0
    };
  };

  const researchStats = getTradeStats('research');
  const competitionStats = getTradeStats('competition');
  const validationStats = getTradeStats('validation');

  const formatPnl = (pnl: number) => {
    const sign = pnl >= 0 ? '+' : '';
    return `${sign}$${pnl.toFixed(2)}`;
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'filled':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'pending':
        return <Activity className="w-4 h-4 text-yellow-500" />;
      case 'cancelled':
        return <XCircle className="w-4 h-4 text-gray-500" />;
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-500" />;
      default:
        return <AlertTriangle className="w-4 h-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'filled':
        return 'text-green-600';
      case 'pending':
        return 'text-yellow-600';
      case 'cancelled':
        return 'text-gray-600';
      case 'failed':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  if (showSummary) {
    return (
      <Card className={className}>
        <CardContent className="p-4">
          <h3 className="text-lg font-semibold mb-4">Trading Mode Performance</h3>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Research Mode */}
            <div className="border rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <ModeLabel mode="research" size="sm" />
                <Badge variant="outline" className="text-xs">
                  {researchStats.count} trades
                </Badge>
              </div>

              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Total P&L:</span>
                  <span className={`font-medium ${researchStats.totalPnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {formatPnl(researchStats.totalPnl)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Win Rate:</span>
                  <span className="font-medium text-blue-600">
                    {(researchStats.winRate * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Avg Trade:</span>
                  <span className={`font-medium ${researchStats.avgPnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {formatPnl(researchStats.avgPnl)}
                  </span>
                </div>
              </div>

              <div className="mt-3 text-xs text-gray-500">
                🧪 Low-risk experiments with segregated capital
              </div>
            </div>

            {/* Competition Mode */}
            <div className="border rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <ModeLabel mode="competition" size="sm" />
                <Badge variant="outline" className="text-xs">
                  {competitionStats.count} trades
                </Badge>
              </div>

              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Total P&L:</span>
                  <span className={`font-medium ${competitionStats.totalPnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {formatPnl(competitionStats.totalPnl)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Win Rate:</span>
                  <span className="font-medium text-blue-600">
                    {(competitionStats.winRate * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Avg Trade:</span>
                  <span className={`font-medium ${competitionStats.avgPnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {formatPnl(competitionStats.avgPnl)}
                  </span>
                </div>
              </div>

              <div className="mt-3 text-xs text-gray-500">
                🎯 Main competition with full capital allocation
              </div>
            </div>

            {/* Validation Mode */}
            <div className="border rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <ModeLabel mode="validation" size="sm" />
                <Badge variant="outline" className="text-xs">
                  {validationStats.count} trades
                </Badge>
              </div>

              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Total P&L:</span>
                  <span className={`font-medium ${validationStats.totalPnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {formatPnl(validationStats.totalPnl)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Win Rate:</span>
                  <span className="font-medium text-blue-600">
                    {(validationStats.winRate * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Avg Trade:</span>
                  <span className={`font-medium ${validationStats.avgPnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {formatPnl(validationStats.avgPnl)}
                  </span>
                </div>
              </div>

              <div className="mt-3 text-xs text-gray-500">
                ✅ Testing promoted strategies before full deployment
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className={`space-y-2 ${className}`}>
      {trades.map((trade) => (
        <div
          key={trade.id}
          className="flex items-center justify-between p-3 border rounded-lg hover:bg-gray-50"
        >
          <div className="flex items-center space-x-3">
            {getStatusIcon(trade.status)}
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-medium">{trade.symbol}</span>
                <Badge variant={trade.side === 'buy' ? 'default' : 'secondary'} className="text-xs">
                  {trade.side.toUpperCase()}
                </Badge>
                <ModeLabel mode={trade.mode} size="sm" />
              </div>
              <div className="text-sm text-gray-600">
                {trade.quantity} @ ${trade.price.toFixed(2)}
                {trade.strategy && ` • ${trade.strategy}`}
                {trade.experimentId && ` • Exp: ${trade.experimentId.slice(-8)}`}
              </div>
            </div>
          </div>

          <div className="text-right">
            {trade.pnl !== undefined && (
              <div className={`font-medium ${trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {formatPnl(trade.pnl)}
              </div>
            )}
            <div className="text-xs text-gray-500">
              {trade.timestamp.toLocaleTimeString()}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default TradeModeIndicator;
