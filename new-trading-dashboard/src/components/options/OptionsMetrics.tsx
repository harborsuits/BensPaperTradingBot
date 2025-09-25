import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Progress } from '@/components/ui/Progress';
import { Badge } from '@/components/ui/Badge';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  DollarSign,
  Clock,
  Target,
  AlertTriangle,
  CheckCircle
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useOptionsMetrics, useOptionsPositions } from '@/hooks/useOptionsData';

export default function OptionsMetrics() {
  const { data: metrics, isLoading } = useOptionsMetrics();
  const { data: positions } = useOptionsPositions();
  
  // Calculate average days to expiry from positions
  const avgDaysToExpiry = positions && positions.length > 0
    ? positions.reduce((sum, p) => {
        const exp = new Date(p.expiration);
        const today = new Date();
        const days = Math.ceil((exp.getTime() - today.getTime()) / (1000 * 60 * 60 * 24));
        return sum + days;
      }, 0) / positions.length
    : 0;

  if (isLoading || !metrics) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[1, 2, 3, 4].map((i) => (
          <Card key={i}>
            <CardContent className="p-6">
              <div className="animate-pulse space-y-2">
                <div className="h-4 bg-gray-200 rounded w-1/2"></div>
                <div className="h-8 bg-gray-200 rounded"></div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  const capitalUsage = (metrics.capitalAllocated / metrics.maxCapital) * 100;

  return (
    <div className="space-y-4">
      {/* Main Metrics Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Total P&L Card */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center justify-between">
              Options P&L
              <DollarSign className="w-4 h-4 text-gray-400" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className={cn(
              "text-2xl font-bold",
              metrics.totalPnL >= 0 ? "text-green-600" : "text-red-600"
            )}>
              ${Math.abs(metrics.totalPnL).toFixed(2)}
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Premium collected: ${metrics.totalPremiumCollected}
            </p>
          </CardContent>
        </Card>

        {/* Win Rate Card */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center justify-between">
              Win Rate
              <Target className="w-4 h-4 text-gray-400" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {(metrics.winRate * 100).toFixed(0)}%
            </div>
            <Progress value={metrics.winRate * 100} className="mt-2" />
          </CardContent>
        </Card>

        {/* Capital Usage Card */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center justify-between">
              Capital Usage
              <Activity className="w-4 h-4 text-gray-400" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {capitalUsage.toFixed(0)}%
            </div>
            <div className="text-xs text-gray-500 mt-1">
              ${metrics.capitalAllocated.toLocaleString()} / ${metrics.maxCapital.toLocaleString()}
            </div>
          </CardContent>
        </Card>

        {/* Active Positions Card */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center justify-between">
              Active Positions
              <Clock className="w-4 h-4 text-gray-400" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {metrics.openPositions}
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Avg {Math.round(avgDaysToExpiry)} days to expiry
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
