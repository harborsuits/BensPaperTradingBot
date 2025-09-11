/**
 * Phase 3: EVO Pool Monitor Card
 * Displays real-time EVO pool status on main dashboard
 */

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import {
  Shield,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  DollarSign,
  Activity,
  Settings,
  RefreshCw,
  Loader2
} from 'lucide-react';
import { useEvoPoolStatus } from '@/hooks/useEvoQueries';

interface EvoPoolMonitorCardProps {
  className?: string;
}

const EvoPoolMonitorCard: React.FC<EvoPoolMonitorCardProps> = ({ className = '' }) => {
  const { data: poolStatus, isLoading, error } = useEvoPoolStatus();

  if (error) {
    return (
      <Card className={`${className} border-red-200`}>
        <CardContent className="p-4">
          <div className="flex items-center space-x-2 text-red-600">
            <AlertTriangle className="w-4 h-4" />
            <span className="text-sm">Failed to load EVO pool status</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'high': return 'text-red-600';
      case 'medium': return 'text-yellow-600';
      case 'low': return 'text-green-600';
      default: return 'text-gray-600';
    }
  };

  const getUtilizationColor = (utilization: number) => {
    if (utilization >= 0.9) return 'text-red-600';
    if (utilization >= 0.7) return 'text-yellow-600';
    return 'text-green-600';
  };

  const getCapWarning = () => {
    if (!poolStatus) return null;
    const utilization = poolStatus.utilizationPct;
    const cap = poolStatus.capPct;

    if (utilization >= cap * 0.9) {
      return {
        message: 'Pool approaching capacity limit',
        type: 'warning' as const
      };
    }
    if (utilization >= cap * 0.8) {
      return {
        message: 'Pool utilization high',
        type: 'info' as const
      };
    }
    return null;
  };

  const warning = getCapWarning();

  return (
    <Card className={`${className} border-green-200`}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Shield className="w-5 h-5 text-green-600" />
            <CardTitle className="text-lg">EVO Pool</CardTitle>
            {isLoading && <Loader2 className="w-4 h-4 animate-spin text-green-600" />}
          </div>
          <div className="flex items-center space-x-2">
            <Badge variant="outline" className="text-xs">
              PAPER ONLY
            </Badge>
            <Button size="sm" variant="ghost" className="h-6 w-6 p-0">
              <RefreshCw className="w-3 h-3" />
            </Button>
          </div>
        </div>
        {warning && (
          <div className={`flex items-center space-x-2 text-xs ${
            warning.type === 'warning' ? 'text-yellow-600' : 'text-blue-600'
          }`}>
            <AlertTriangle className="w-3 h-3" />
            <span>{warning.message}</span>
          </div>
        )}
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Pool Capacity Gauge */}
        {poolStatus && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-600">Pool Utilization</span>
              <span className={getUtilizationColor(poolStatus.utilizationPct)}>
                {(poolStatus.utilizationPct * 100).toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all duration-300 ${
                  poolStatus.utilizationPct >= 0.9 ? 'bg-red-500' :
                  poolStatus.utilizationPct >= 0.7 ? 'bg-yellow-500' : 'bg-green-500'
                }`}
                style={{ width: `${Math.min(poolStatus.utilizationPct * 100, 100)}%` }}
              />
            </div>
            <div className="flex justify-between text-xs text-gray-500">
              <span>0%</span>
              <span>Cap: {(poolStatus.capPct * 100).toFixed(1)}%</span>
              <span>100%</span>
            </div>
          </div>
        )}

        {/* Key Metrics */}
        <div className="grid grid-cols-2 gap-4">
          <div className="text-center">
            <div className="flex items-center justify-center space-x-1 mb-1">
              <DollarSign className="w-4 h-4 text-green-600" />
              <span className="text-xs text-gray-600">Available</span>
            </div>
            <div className="text-lg font-bold text-green-600">
              {poolStatus ? `$${poolStatus.availableCapacity.toLocaleString()}` : '--'}
            </div>
          </div>

          <div className="text-center">
            <div className="flex items-center justify-center space-x-1 mb-1">
              <Activity className="w-4 h-4 text-blue-600" />
              <span className="text-xs text-gray-600">Active</span>
            </div>
            <div className="text-lg font-bold text-blue-600">
              {poolStatus?.activeCount || 0}
            </div>
          </div>
        </div>

        {/* P&L and Risk */}
        <div className="grid grid-cols-2 gap-4">
          <div className="text-center">
            <div className="flex items-center justify-center space-x-1 mb-1">
              {poolStatus && poolStatus.poolPnl >= 0 ? (
                <TrendingUp className="w-4 h-4 text-green-600" />
              ) : (
                <TrendingDown className="w-4 h-4 text-red-600" />
              )}
              <span className="text-xs text-gray-600">P&L</span>
            </div>
            <div className={`text-lg font-bold ${
              poolStatus && poolStatus.poolPnl >= 0 ? 'text-green-600' : 'text-red-600'
            }`}>
              {poolStatus ? `$${poolStatus.poolPnl.toFixed(2)}` : '--'}
            </div>
          </div>

          <div className="text-center">
            <div className="flex items-center justify-center space-x-1 mb-1">
              <Settings className="w-4 h-4 text-gray-600" />
              <span className="text-xs text-gray-600">Risk</span>
            </div>
            <div className={`text-sm font-medium uppercase ${getRiskColor(poolStatus?.riskLevel || 'low')}`}>
              {poolStatus?.riskLevel || '--'}
            </div>
          </div>
        </div>

        {/* Status Indicator */}
        <div className="flex items-center justify-between pt-2 border-t">
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${
              poolStatus ? 'bg-green-500' : 'bg-gray-400'
            }`} />
            <span className="text-xs text-gray-600">
              {poolStatus ? 'Active' : 'Offline'}
            </span>
          </div>
          <Button size="sm" variant="outline" className="text-xs h-6">
            Manage EVO
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default EvoPoolMonitorCard;
