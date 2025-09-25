import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card';
import { Progress } from '@/components/ui/Progress';
import { Badge } from '@/components/ui/Badge';
import { 
  Activity, 
  DollarSign, 
  LineChart, 
  ShoppingCart, 
  Target,
  RefreshCw,
  CheckCircle,
  AlertCircle
} from 'lucide-react';
import { useDataSync } from '@/contexts/DataSyncContext';
import { SyncStatusIndicator } from '@/components/SyncStatusIndicator';
import { formatDistanceToNow } from 'date-fns';

export function DataSyncDashboard() {
  const {
    portfolio,
    account,
    positions,
    quotes,
    openOrders,
    recentTrades,
    decisions,
    marketContext,
    strategies,
    isConnected,
    lastSync,
  } = useDataSync();
  
  const dataSources = [
    {
      name: 'Portfolio',
      icon: DollarSign,
      status: portfolio ? 'synced' : 'loading',
      lastUpdate: portfolio?.asOf,
      details: portfolio ? `$${portfolio.equity?.toLocaleString()} equity` : 'Loading...',
    },
    {
      name: 'Positions',
      icon: LineChart,
      status: positions.length > 0 ? 'synced' : 'empty',
      lastUpdate: lastSync?.toISOString(),
      details: `${positions.length} positions`,
    },
    {
      name: 'Orders',
      icon: ShoppingCart,
      status: openOrders ? 'synced' : 'loading',
      lastUpdate: lastSync?.toISOString(),
      details: `${openOrders.length} open orders`,
    },
    {
      name: 'Quotes',
      icon: Activity,
      status: quotes.size > 0 ? 'synced' : 'loading',
      lastUpdate: lastSync?.toISOString(),
      details: `${quotes.size} symbols tracked`,
    },
    {
      name: 'Strategies',
      icon: Target,
      status: strategies.length > 0 ? 'synced' : 'loading',
      lastUpdate: lastSync?.toISOString(),
      details: `${strategies.filter((s: any) => s.active).length} active`,
    },
  ];
  
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'synced': return 'success';
      case 'loading': return 'secondary';
      case 'empty': return 'default';
      case 'error': return 'destructive';
      default: return 'default';
    }
  };
  
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'synced': return <CheckCircle className="h-4 w-4" />;
      case 'loading': return <RefreshCw className="h-4 w-4 animate-spin" />;
      case 'error': return <AlertCircle className="h-4 w-4" />;
      default: return null;
    }
  };
  
  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Data Synchronization</CardTitle>
            <CardDescription>
              Real-time sync status across all data sources
            </CardDescription>
          </div>
          <SyncStatusIndicator />
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Connection Status */}
          <div className="flex items-center justify-between p-3 bg-muted rounded-lg">
            <div className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              <span className="font-medium">Connection Status</span>
            </div>
            <Badge variant={isConnected ? 'success' : 'destructive'}>
              {isConnected ? 'Connected' : 'Disconnected'}
            </Badge>
          </div>
          
          {/* Data Sources */}
          <div className="space-y-3">
            {dataSources.map((source) => {
              const Icon = source.icon;
              return (
                <div
                  key={source.name}
                  className="flex items-center justify-between p-3 border rounded-lg"
                >
                  <div className="flex items-center gap-3">
                    <Icon className="h-5 w-5 text-muted-foreground" />
                    <div>
                      <div className="font-medium">{source.name}</div>
                      <div className="text-sm text-muted-foreground">
                        {source.details}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {source.lastUpdate && (
                      <span className="text-xs text-muted-foreground">
                        {formatDistanceToNow(new Date(source.lastUpdate), { addSuffix: true })}
                      </span>
                    )}
                    <Badge variant={getStatusColor(source.status) as any}>
                      <div className="flex items-center gap-1">
                        {getStatusIcon(source.status)}
                        {source.status}
                      </div>
                    </Badge>
                  </div>
                </div>
              );
            })}
          </div>
          
          {/* Sync Progress */}
          {lastSync && (
            <div className="mt-4 pt-4 border-t">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Sync Health</span>
                <span className="text-sm text-muted-foreground">
                  Last full sync: {formatDistanceToNow(lastSync, { addSuffix: true })}
                </span>
              </div>
              <Progress 
                value={isConnected ? 100 : 0} 
                className="h-2"
              />
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
