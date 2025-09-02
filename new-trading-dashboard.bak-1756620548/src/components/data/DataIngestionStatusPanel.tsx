import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { 
  AlertCircle, 
  CheckCircle, 
  Clock, 
  Database, 
  RefreshCw, 
  Server,
  Activity
} from 'lucide-react';
import { format, formatDistanceToNow } from 'date-fns';

import { ingestionApi } from '@/services/api';
import { useWebSocketMessage } from '@/services/websocket';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import Tooltip from '@/components/ui/Tooltip';
import { Progress } from '@/components/ui/Progress';

/**
 * DataIngestionStatusPanel displays the health and status of data ingestion sources
 * as well as metrics about the data being processed.
 */
const DataIngestionStatusPanel: React.FC = () => {
  // Fetch data status summary using React Query
  const { 
    data: statusData,
    isLoading: isStatusLoading,
    isError: isStatusError,
    refetch: refetchStatus
  } = useQuery({
    queryKey: ['dataIngestion', 'status'],
    queryFn: async () => {
      const response = await ingestionApi.getDataStatus();
      return response.success ? response.data : null;
    },
    staleTime: 30000 // 30 seconds
  });

  // Listen for WebSocket updates on the data channel
  useWebSocketMessage<any>(
    'data_status_update',
    (message) => {
      if (message.data) {
        // Invalidate the query cache to trigger a refresh
        refetchStatus();
      }
    }
  );

  // Format a timestamp as a relative time (e.g., "2 minutes ago") or absolute if older
  const formatTimestamp = (timestamp: string) => {
    if (!timestamp) return 'N/A';
    
    const date = new Date(timestamp);
    const minutesAgo = Math.floor((new Date().getTime() - date.getTime()) / (1000 * 60));
    
    if (minutesAgo < 60) {
      return formatDistanceToNow(date, { addSuffix: true });
    } else {
      return format(date, 'MMM d, h:mm a');
    }
  };

  // Generate a health status indicator
  const getStatusIndicator = (status: string, healthScore?: number) => {
    if (status === 'healthy' || healthScore && healthScore > 80) {
      return (
        <Badge variant="outline" className="flex items-center gap-1 bg-green-100 text-green-800 border-green-200">
          <CheckCircle size={12} />
          <span>Healthy</span>
        </Badge>
      );
    } else if (status === 'degraded' || healthScore && healthScore > 50) {
      return (
        <Badge variant="outline" className="flex items-center gap-1 bg-yellow-100 text-yellow-800 border-yellow-200">
          <AlertCircle size={12} />
          <span>Degraded</span>
        </Badge>
      );
    } else {
      return (
        <Badge variant="destructive" className="flex items-center gap-1">
          <AlertCircle size={12} />
          <span>Unhealthy</span>
        </Badge>
      );
    }
  };

  if (isStatusLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Database size={16} />
            Data Ingestion Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-6">
            <RefreshCw className="animate-spin text-muted-foreground" size={24} />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (isStatusError || !statusData) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Database size={16} />
            Data Ingestion Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center py-6 text-destructive">
            <AlertCircle size={24} className="mb-2" />
            <p className="text-sm">Failed to load data ingestion status</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const { sources = [], metrics } = statusData;

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base flex items-center gap-2">
          <Database size={16} />
          Data Ingestion Status
          <span className="ml-auto text-xs text-muted-foreground">
            Last updated: {formatTimestamp(statusData.timestamp)}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {/* Overall status summary */}
        <div className="grid grid-cols-3 gap-4 mb-4">
          <div className="bg-muted/30 p-3 rounded-lg flex flex-col items-center">
            <span className="text-xs text-muted-foreground mb-1">Data Sources</span>
            <div className="flex items-center gap-2">
              <Server size={14} />
              <span className="text-lg font-semibold">{sources.length}</span>
            </div>
          </div>
          
          <div className="bg-muted/30 p-3 rounded-lg flex flex-col items-center">
            <span className="text-xs text-muted-foreground mb-1">Symbols Tracked</span>
            <div className="flex items-center gap-2">
              <Activity size={14} />
              <span className="text-lg font-semibold">{metrics?.totalSymbolsTracked || 0}</span>
            </div>
          </div>
          
          <div className="bg-muted/30 p-3 rounded-lg flex flex-col items-center">
            <span className="text-xs text-muted-foreground mb-1">Last Sync</span>
            <div className="flex items-center gap-2">
              <Clock size={14} />
              <span className="text-sm font-semibold">{formatTimestamp(metrics?.lastFullSyncCompleted || '')}</span>
            </div>
          </div>
        </div>
        
        {/* Data sources list */}
        <div className="space-y-3">
          <div className="text-xs font-medium text-muted-foreground mb-1">Data Sources</div>
          {sources.map((source) => (
            <div key={source.id} className="flex items-center justify-between py-2 border-b border-border">
              <div className="flex items-center gap-2">
                <Server size={14} className="text-muted-foreground" />
                <div>
                  <div className="font-medium text-sm">{source.name}</div>
                  <div className="text-xs text-muted-foreground">{source.type}</div>
                </div>
              </div>
              
              <div className="flex items-center gap-3">
                <div className="text-xs">
                  <Tooltip content={`Last update: ${source.lastUpdate}`}>
                    <div className="flex items-center gap-1 text-muted-foreground cursor-default">
                      <Clock size={12} />
                      <span>{formatTimestamp(source.lastUpdate)}</span>
                    </div>
                  </Tooltip>
                </div>
                
                {getStatusIndicator(source.status, source.healthScore)}
              </div>
            </div>
          ))}
        </div>
        
        {/* Metrics section */}
        {metrics && (
          <div className="mt-4 space-y-3">
            <div className="text-xs font-medium text-muted-foreground">Performance Metrics</div>
            
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs">Error Rate</span>
                <span className="text-xs font-medium">{metrics.errorRate.toFixed(2)}%</span>
              </div>
              <Progress value={Math.min(metrics.errorRate * 2, 100)} className="h-1.5" variant={metrics.errorRate > 5 ? "destructive" : "default"} />
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs">Latency</span>
                <span className="text-xs font-medium">{metrics.averageLatency.toFixed(0)}ms</span>
              </div>
              <Progress value={Math.min(metrics.averageLatency / 10, 100)} className="h-1.5" variant={metrics.averageLatency > 500 ? "destructive" : "default"} />
            </div>
            
            <div className="flex items-center justify-between text-xs mt-2">
              <span>Data Points Ingested:</span>
              <span className="font-medium">{metrics.dataPointsIngested.toLocaleString()}</span>
            </div>
            
            <div className="flex items-center justify-between text-xs">
              <span>Requests Last Hour:</span>
              <span className="font-medium">{metrics.requestsLastHour.toLocaleString()}</span>
            </div>
          </div>
        )}
        
        {/* Issues summary if there are problems */}
        {metrics?.symbolsWithErrors?.length > 0 && (
          <div className="mt-4 p-2 border border-destructive/30 bg-destructive/5 rounded-md">
            <div className="flex items-center gap-2 text-destructive text-xs font-medium mb-1">
              <AlertCircle size={12} />
              <span>Issues Detected</span>
            </div>
            <p className="text-xs text-muted-foreground">
              {metrics.symbolsWithErrors.length} symbols with data issues: {metrics.symbolsWithErrors.slice(0, 5).join(', ')}
              {metrics.symbolsWithErrors.length > 5 && '...'}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default DataIngestionStatusPanel;
