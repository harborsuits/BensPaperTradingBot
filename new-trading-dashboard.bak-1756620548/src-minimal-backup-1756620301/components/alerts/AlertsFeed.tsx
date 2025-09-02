import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  AlertCircle, 
  AlertTriangle, 
  CheckCircle, 
  Info, 
  Clock, 
  Bell, 
  Filter,
  X,
  Eye,
  Check,
  RefreshCw,
} from 'lucide-react';
import { format, formatDistanceToNow } from 'date-fns';

import { alertsApi } from '@/services/alertsApi';
import { Alert, AlertSeverity, AlertSource, AlertStatus, AlertFilterOptions } from '@/types/alerts.types';
import useAlertsWebSocket from '@/hooks/useAlertsWebSocket';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Checkbox } from '@/components/ui/Checkbox';
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuCheckboxItem } from '@/components/ui/DropdownMenu';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/Tabs';
import { Skeleton } from '@/components/ui/Skeleton';

interface AlertsFeedProps {
  className?: string;
  compact?: boolean;
  limit?: number;
}

const AlertsFeed: React.FC<AlertsFeedProps> = ({ 
  className = '', 
  compact = false,
  limit = 50
}) => {
  // Initialize the alerts websocket hook
  useAlertsWebSocket();
  
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [filters, setFilters] = useState<AlertFilterOptions>({
    severity: undefined,
    source: undefined,
    status: compact ? ['new'] : undefined,
  });

  // Fetch alerts with the current filters
  const { 
    data: alerts,
    isLoading,
    isError,
    refetch
  } = useQuery({
    queryKey: ['alerts', filters],
    queryFn: async () => {
      const response = await alertsApi.getAlerts(filters);
      return response.success ? response.data : [];
    },
    staleTime: 30000 // 30 seconds
  });

  // Fetch alerts state for unread counts and status info
  const { data: alertsState } = useQuery({
    queryKey: ['alertsState'],
    queryFn: async () => {
      const response = await alertsApi.getAlertsState();
      return response.success ? response.data : { alerts: [], unreadCount: 0, hasErrors: false, lastChecked: new Date().toISOString() };
    },
    staleTime: 30000
  });

  // Mutation for acknowledging alerts
  const acknowledgeMutation = useMutation({
    mutationFn: (ids: string[]) => alertsApi.acknowledgeAlerts(ids),
    onSuccess: () => {
      queryClient.invalidateQueries(['alerts']);
      queryClient.invalidateQueries(['alertsState']);
    }
  });

  // Helper function to get the appropriate icon based on severity
  const getSeverityIcon = (severity: AlertSeverity) => {
    switch (severity) {
      case 'error':
        return <AlertCircle className="text-destructive" size={compact ? 16 : 20} />;
      case 'warning':
        return <AlertTriangle className="text-amber-500" size={compact ? 16 : 20} />;
      case 'success':
        return <CheckCircle className="text-green-500" size={compact ? 16 : 20} />;
      default:
        return <Info className="text-blue-500" size={compact ? 16 : 20} />;
    }
  };

  // Helper function to format the timestamp
  const formatTime = (timestamp: string) => {
    if (compact) {
      return formatDistanceToNow(new Date(timestamp), { addSuffix: true });
    }
    
    const date = new Date(timestamp);
    const now = new Date();
    const diffHours = Math.abs(now.getTime() - date.getTime()) / 36e5;
    
    if (diffHours < 24) {
      return formatDistanceToNow(date, { addSuffix: true });
    } else {
      return format(date, 'MMM d, h:mm a');
    }
  };

  // Function to acknowledge a single alert
  const acknowledgeAlert = async (alertId: string) => {
    acknowledgeMutation.mutate([alertId]);
  };

  // Function to acknowledge all visible alerts
  const acknowledgeAllAlerts = async () => {
    if (alerts && alerts.length > 0) {
      const ids = alerts.filter(alert => alert.status === 'new').map(alert => alert.id);
      if (ids.length > 0) {
        acknowledgeMutation.mutate(ids);
      }
    }
  };

  // Filter alerts based on tab and search query
  const getFilteredAlerts = () => {
    if (!alerts) return [];
    
    let filtered = [...alerts];
    
    // Filter by tab
    if (activeTab === 'errors') {
      filtered = filtered.filter(alert => alert.severity === 'error');
    } else if (activeTab === 'warnings') {
      filtered = filtered.filter(alert => alert.severity === 'warning');
    } else if (activeTab === 'info') {
      filtered = filtered.filter(alert => alert.severity === 'info');
    } else if (activeTab === 'new') {
      filtered = filtered.filter(alert => alert.status === 'new');
    }
    
    // Filter by search query if present
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(alert => 
        alert.title.toLowerCase().includes(query) || 
        alert.message.toLowerCase().includes(query) ||
        (alert.relatedSymbols && alert.relatedSymbols.some(symbol => symbol.toLowerCase().includes(query)))
      );
    }
    
    // Apply limit if specified
    if (limit) {
      filtered = filtered.slice(0, limit);
    }
    
    return filtered;
  };

  // Display appropriate UI for different states
  if (isLoading) {
    return (
      <Card className={className}>
        <CardHeader className={`${compact ? 'p-3' : ''}`}>
          <CardTitle className={`${compact ? 'text-sm' : 'text-base'} flex items-center gap-2`}>
            <Bell size={compact ? 16 : 18} />
            Alerts & Notifications
          </CardTitle>
        </CardHeader>
        <CardContent className={`${compact ? 'p-3' : ''}`}>
          <div className="space-y-2">
            {[1, 2, 3].map(i => (
              <div key={i} className="flex items-start gap-3 pb-2 border-b border-border last:border-0">
                <Skeleton className="w-5 h-5 rounded-full flex-shrink-0 mt-1" />
                <div className="space-y-1 flex-1">
                  <Skeleton className="h-4 w-3/4" />
                  <Skeleton className="h-3 w-full" />
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (isError) {
    return (
      <Card className={className}>
        <CardHeader className={`${compact ? 'p-3' : ''}`}>
          <CardTitle className={`${compact ? 'text-sm' : 'text-base'} flex items-center gap-2`}>
            <Bell size={compact ? 16 : 18} />
            Alerts & Notifications
          </CardTitle>
        </CardHeader>
        <CardContent className={`${compact ? 'p-3' : ''}`}>
          <div className="flex flex-col items-center justify-center py-6 text-destructive">
            <AlertCircle size={24} className="mb-2" />
            <p className="text-sm">Failed to load alerts</p>
            <Button variant="outline" size="sm" className="mt-2" onClick={() => refetch()}>
              <RefreshCw size={14} className="mr-1" />
              Retry
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  const filteredAlerts = getFilteredAlerts();
  const unreadCount = alertsState?.unreadCount || 0;

  return (
    <Card className={className}>
      <CardHeader className={`${compact ? 'p-3 pb-0' : 'pb-2'}`}>
        <CardTitle className={`${compact ? 'text-sm' : 'text-base'} flex items-center justify-between`}>
          <div className="flex items-center gap-2">
            <Bell size={compact ? 16 : 18} />
            Alerts & Notifications
            {unreadCount > 0 && (
              <Badge variant="destructive" className="ml-1">
                {unreadCount}
              </Badge>
            )}
          </div>
          
          {!compact && (
            <div className="flex items-center gap-2">
              <Button 
                variant="outline" 
                size="sm" 
                className="flex items-center gap-1"
                onClick={acknowledgeAllAlerts}
                disabled={!alerts || alerts.filter(a => a.status === 'new').length === 0}
              >
                <Check size={14} />
                Mark All Read
              </Button>
              
              <Button 
                variant="outline" 
                size="sm" 
                className="flex items-center gap-1"
                onClick={() => refetch()}
              >
                <RefreshCw size={14} />
                Refresh
              </Button>
            </div>
          )}
        </CardTitle>
      </CardHeader>

      <CardContent className={`${compact ? 'p-3 pt-2' : 'pt-2'}`}>
        {!compact && (
          <div>
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <div className="flex items-center justify-between mb-2">
                <TabsList>
                  <TabsTrigger value="all">All</TabsTrigger>
                  <TabsTrigger value="new">
                    New
                    {unreadCount > 0 && (
                      <Badge variant="secondary" className="ml-1 h-4 min-w-4 px-1 text-xs">
                        {unreadCount}
                      </Badge>
                    )}
                  </TabsTrigger>
                  <TabsTrigger value="errors">Errors</TabsTrigger>
                  <TabsTrigger value="warnings">Warnings</TabsTrigger>
                  <TabsTrigger value="info">Info</TabsTrigger>
                </TabsList>
                
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline" size="sm" className="gap-1">
                      <Filter size={14} />
                      Filter
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-56">
                    <div className="p-2">
                      <h4 className="mb-2 text-sm font-medium">Filter by Source</h4>
                      {(['system', 'trade', 'data', 'security'] as AlertSource[]).map(source => (
                        <div key={source} className="flex items-center space-x-2 mb-1">
                          <Checkbox 
                            id={`source-${source}`} 
                            checked={filters.source?.includes(source)} 
                            onCheckedChange={(checked) => {
                              if (checked) {
                                setFilters({
                                  ...filters,
                                  source: [...(filters.source || []), source]
                                });
                              } else {
                                setFilters({
                                  ...filters,
                                  source: filters.source?.filter(s => s !== source)
                                });
                              }
                            }}
                          />
                          <label 
                            htmlFor={`source-${source}`}
                            className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 capitalize"
                          >
                            {source}
                          </label>
                        </div>
                      ))}
                    </div>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
              
              <div className="mb-3">
                <Input
                  placeholder="Search alerts..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full"
                />
              </div>
            </Tabs>
          </div>
        )}
        
        {filteredAlerts.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-6 text-muted-foreground">
            <Bell size={24} className="mb-2 text-muted-foreground" />
            <p className="text-sm">No alerts to display</p>
          </div>
        ) : (
          <div className={`space-y-2 ${compact ? 'max-h-64' : 'max-h-[500px]'} overflow-y-auto pr-1`}>
            {filteredAlerts.map((alert) => (
              <div 
                key={alert.id} 
                className={`flex gap-3 pb-2 border-b border-border last:border-0 ${
                  alert.status === 'new' ? 'bg-muted/20' : ''
                }`}
              >
                <div className="flex-shrink-0 mt-1">
                  {getSeverityIcon(alert.severity)}
                </div>
                
                <div className="flex-1">
                  <div className="flex justify-between">
                    <h4 className={`${compact ? 'text-xs' : 'text-sm'} font-medium`}>{alert.title}</h4>
                    {!compact && alert.status === 'new' && (
                      <Button 
                        variant="ghost" 
                        size="sm" 
                        className="h-6 px-2"
                        onClick={() => acknowledgeAlert(alert.id)}
                      >
                        <Eye size={14} className="mr-1" />
                        <span className="text-xs">Mark Read</span>
                      </Button>
                    )}
                  </div>
                  
                  <p className={`${compact ? 'text-xs' : 'text-sm'} text-muted-foreground mt-0.5`}>
                    {alert.message}
                  </p>
                  
                  <div className="flex flex-wrap gap-1 mt-1">
                    <Badge variant="outline" className="text-[10px] py-0 px-1 capitalize">
                      {alert.source}
                    </Badge>
                    
                    {alert.relatedSymbols && alert.relatedSymbols.length > 0 && (
                      <Badge variant="secondary" className="text-[10px] py-0 px-1">
                        {alert.relatedSymbols.join(', ')}
                      </Badge>
                    )}
                    
                    <span className="text-[10px] text-muted-foreground flex items-center">
                      <Clock size={10} className="mr-0.5" />
                      {formatTime(alert.timestamp)}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
        
        {compact && filteredAlerts.length > 0 && (
          <div className="flex justify-center mt-2">
            <Button variant="ghost" size="sm" className="text-xs">
              View All Alerts
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default AlertsFeed;
