import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { toast } from 'react-hot-toast';
import {
  AlertCircle,
  Info,
  AlertTriangle,
  Clock,
  Filter,
  Download,
  RefreshCcw,
  CheckCircle
} from 'lucide-react';

import { loggingApi } from '@/services/api';
import { qk } from '@/services/qk';
import { useWebSocketChannel, useWebSocketMessage } from '@/services/websocket';
import { LogEvent } from '@/types/api.types';

import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { StatusBadge } from '@/components/ui/StatusBadge';
import WebSocketIndicator from '@/components/ui/WebSocketIndicator';
import { ErrorBoundary } from '@/components/util/ErrorBoundary';

const LogsPage: React.FC = () => {
  const [logs, setLogs] = useState<LogEvent[]>([]);
  const [alerts, setAlerts] = useState<LogEvent[]>([]);
  const [levelFilter, setLevelFilter] = useState<'ALL' | 'INFO' | 'WARNING' | 'ERROR'>('ALL');
  const [sourceFilter, setSourceFilter] = useState<string>('ALL');
  const [activeTab, setActiveTab] = useState<'logs' | 'alerts'>('logs');
  
  // Connect to WebSocket for real-time updates
  const { isConnected } = useWebSocketChannel('logging', true);
  
  // Fetch logs
  const { 
    data: logsData, 
    isLoading: isLoadingLogs,
    refetch: refetchLogs
  } = useQuery({
    queryKey: qk.logs(levelFilter),
    queryFn: () => loggingApi.getLogs(levelFilter === 'ALL' ? undefined : levelFilter.toLowerCase(), 100),
    onSuccess: (response) => {
      if (response.success && response.data) {
        setLogs(response.data);
      }
    },
  });
  
  // Fetch alerts
  const { 
    data: alertsData, 
    isLoading: isLoadingAlerts,
    refetch: refetchAlerts
  } = useQuery({
    queryKey: ['alerts'],
    queryFn: () => loggingApi.getAlerts(20),
    onSuccess: (response) => {
      if (response.success && response.data) {
        setAlerts(response.data);
      }
    },
  });
  
  // WebSocket message handlers
  useWebSocketMessage<LogEvent>(
    'log',
    (message) => {
      const logEvent = message.data;
      
      // Add to logs list if it matches the current filter
      if (levelFilter === 'ALL' || logEvent.level === levelFilter) {
        setLogs(prev => [logEvent, ...prev].slice(0, 100));
      }
      
      // Add to alerts list if it's a warning or error
      if (logEvent.level === 'WARNING' || logEvent.level === 'ERROR') {
        setAlerts(prev => [logEvent, ...prev].slice(0, 20));
        
        // Show toast notification for warnings and errors
        toast(
          <div className="flex items-start gap-2">
            {logEvent.level === 'ERROR' ? (
              <AlertCircle className="text-bear mt-0.5" size={16} />
            ) : (
              <AlertTriangle className="text-warning mt-0.5" size={16} />
            )}
            <div>
              <p className="font-medium">{logEvent.level}</p>
              <p className="text-sm">{logEvent.message}</p>
            </div>
          </div>
        );
      }
    },
    [levelFilter]
  );
  
  // Get all available log sources for filtering
  const logSources = Array.from(
    new Set(logs.map(log => log.source))
  ).filter(Boolean);
  
  // Get filtered logs based on current filters
  const filteredLogs = logs.filter(log => {
    if (sourceFilter !== 'ALL' && log.source !== sourceFilter) {
      return false;
    }
    return true;
  });
  
  // Format timestamp
  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
  };
  
  // Get icon for log level
  const getLogLevelIcon = (level: string) => {
    switch (level) {
      case 'ERROR':
        return <AlertCircle size={16} className="text-bear" />;
      case 'WARNING':
        return <AlertTriangle size={16} className="text-warning" />;
      case 'INFO':
        return <Info size={16} className="text-info" />;
      default:
        return <CheckCircle size={16} className="text-muted-foreground" />;
    }
  };
  
  // Export logs to JSON file
  const exportLogs = () => {
    const dataToExport = activeTab === 'logs' ? filteredLogs : alerts;
    const dataStr = JSON.stringify(dataToExport, null, 2);
    const dataUri = `data:application/json;charset=utf-8,${encodeURIComponent(dataStr)}`;
    
    const exportFileDefaultName = `${activeTab}-${new Date().toISOString().slice(0, 10)}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between space-y-2 md:space-y-0">
        <div>
          <h1 className="text-2xl font-bold">Logs & Alerts</h1>
          <p className="text-muted-foreground">Monitor system logs and important alerts</p>
        </div>
        <div className="flex items-center gap-4">
          <WebSocketIndicator isConnected={isConnected} />
          <button
            onClick={exportLogs}
            className="flex items-center gap-2 px-3 py-1.5 bg-muted rounded-md text-sm"
          >
            <Download size={16} />
            Export {activeTab}
          </button>
        </div>
      </div>
      
      {/* Tabs */}
      <div className="flex bg-muted/30 rounded-md overflow-hidden">
        <button
          onClick={() => setActiveTab('logs')}
          className={`px-4 py-2 text-sm flex items-center gap-2 ${
            activeTab === 'logs' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground'
          }`}
        >
          <Info size={16} />
          System Logs
        </button>
        <button
          onClick={() => setActiveTab('alerts')}
          className={`px-4 py-2 text-sm flex items-center gap-2 ${
            activeTab === 'alerts' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground'
          }`}
        >
          <AlertTriangle size={16} />
          Alerts
          {alerts.length > 0 && (
            <span className="bg-bear rounded-full w-5 h-5 flex items-center justify-center text-xs">
              {alerts.length}
            </span>
          )}
        </button>
      </div>
      
      {/* Filters (only show for logs tab) */}
      {activeTab === 'logs' && (
        <Card>
          <CardContent className="p-4">
            <div className="flex flex-col sm:flex-row sm:items-center space-y-2 sm:space-y-0 sm:space-x-4">
              <div className="flex items-center space-x-2">
                <span className="text-sm text-muted-foreground">Level:</span>
                <div className="flex bg-muted/30 rounded-md overflow-hidden">
                  <button
                    onClick={() => setLevelFilter('ALL')}
                    className={`px-3 py-1.5 text-xs ${
                      levelFilter === 'ALL' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground'
                    }`}
                  >
                    All
                  </button>
                  <button
                    onClick={() => setLevelFilter('INFO')}
                    className={`px-3 py-1.5 text-xs ${
                      levelFilter === 'INFO' ? 'bg-info/30 text-info' : 'text-muted-foreground'
                    }`}
                  >
                    Info
                  </button>
                  <button
                    onClick={() => setLevelFilter('WARNING')}
                    className={`px-3 py-1.5 text-xs ${
                      levelFilter === 'WARNING' ? 'bg-warning/30 text-warning' : 'text-muted-foreground'
                    }`}
                  >
                    Warning
                  </button>
                  <button
                    onClick={() => setLevelFilter('ERROR')}
                    className={`px-3 py-1.5 text-xs ${
                      levelFilter === 'ERROR' ? 'bg-bear/30 text-bear' : 'text-muted-foreground'
                    }`}
                  >
                    Error
                  </button>
                </div>
              </div>
              
              {logSources.length > 0 && (
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-muted-foreground">Source:</span>
                  <select 
                    value={sourceFilter}
                    onChange={(e) => setSourceFilter(e.target.value)}
                    className="bg-muted/30 border border-border rounded-md px-3 py-1.5 text-xs"
                  >
                    <option value="ALL">All Sources</option>
                    {logSources.map(source => (
                      <option key={source} value={source}>
                        {source}
                      </option>
                    ))}
                  </select>
                </div>
              )}
              
              <div className="flex-1"></div>
              
              <button
                onClick={() => activeTab === 'logs' ? refetchLogs() : refetchAlerts()}
                className="flex items-center gap-2 px-3 py-1.5 bg-muted rounded-md text-sm ml-auto"
              >
                <RefreshCcw size={16} />
                Refresh
              </button>
            </div>
          </CardContent>
        </Card>
      )}
      
      {/* Logs or Alerts Content */}
      <ErrorBoundary>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle>
            {activeTab === 'logs' ? 'System Logs' : 'Important Alerts'}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {((activeTab === 'logs' && isLoadingLogs) || (activeTab === 'alerts' && isLoadingAlerts)) ? (
            <div className="py-12 flex items-center justify-center">
              <p className="text-muted-foreground">Loading...</p>
            </div>
          ) : activeTab === 'logs' ? (
            filteredLogs.length > 0 ? (
              <div className="space-y-2 max-h-[700px] overflow-y-auto p-1">
                {filteredLogs.map((log, index) => (
                  <div 
                    key={log.id || index} 
                    className="border border-border rounded-md p-3 hover:bg-muted/10"
                  >
                    <div className="flex items-start">
                      <div className="mr-3 mt-0.5">
                        {getLogLevelIcon(log.level)}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-1">
                          <div className="flex items-center gap-2">
                            <StatusBadge 
                              variant={
                                log.level === 'ERROR' ? 'bear' : 
                                log.level === 'WARNING' ? 'warning' : 
                                'info'
                              }
                              size="sm"
                            >
                              {log.level}
                            </StatusBadge>
                            <span className="text-sm font-medium">{log.source}</span>
                          </div>
                          <div className="flex items-center text-xs text-muted-foreground">
                            <Clock size={12} className="mr-1" />
                            {formatTimestamp(log.timestamp)}
                          </div>
                        </div>
                        <p className="text-sm">{log.message}</p>
                        
                        {log.data && Object.keys(log.data).length > 0 && (
                          <details className="mt-2">
                            <summary className="text-xs text-muted-foreground cursor-pointer hover:text-foreground">
                              View details
                            </summary>
                            <pre className="text-xs mt-2 p-2 bg-muted/20 rounded-md overflow-x-auto">
                              {JSON.stringify(log.data, null, 2)}
                            </pre>
                          </details>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="py-12 flex flex-col items-center justify-center space-y-2">
                <Info size={32} className="text-muted-foreground" />
                <p className="text-muted-foreground">No logs found matching your filters</p>
              </div>
            )
          ) : alerts.length > 0 ? (
            <div className="space-y-2 max-h-[700px] overflow-y-auto p-1">
              {alerts.map((alert, index) => (
                <div 
                  key={alert.id || index} 
                  className={`border border-border rounded-md p-3 hover:bg-muted/10
                    ${alert.level === 'ERROR' ? 'bg-bear/5' : 'bg-warning/5'}`
                  }
                >
                  <div className="flex items-start">
                    <div className="mr-3 mt-0.5">
                      {getLogLevelIcon(alert.level)}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-1">
                        <div className="flex items-center gap-2">
                          <StatusBadge 
                            variant={alert.level === 'ERROR' ? 'bear' : 'warning'}
                            size="sm"
                          >
                            {alert.level}
                          </StatusBadge>
                          <span className="text-sm font-medium">{alert.source}</span>
                        </div>
                        <div className="flex items-center text-xs text-muted-foreground">
                          <Clock size={12} className="mr-1" />
                          {formatTimestamp(alert.timestamp)}
                        </div>
                      </div>
                      <p className="text-sm">{alert.message}</p>
                      
                      {alert.data && Object.keys(alert.data).length > 0 && (
                        <details className="mt-2">
                          <summary className="text-xs text-muted-foreground cursor-pointer hover:text-foreground">
                            View details
                          </summary>
                          <pre className="text-xs mt-2 p-2 bg-muted/20 rounded-md overflow-x-auto">
                            {JSON.stringify(alert.data, null, 2)}
                          </pre>
                        </details>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="py-12 flex flex-col items-center justify-center space-y-2">
              <CheckCircle size={32} className="text-bull" />
              <p className="text-muted-foreground">No alerts at the moment</p>
            </div>
          )}
        </CardContent>
      </Card>
      </ErrorBoundary>
    </div>
  );
};

export default LogsPage;
