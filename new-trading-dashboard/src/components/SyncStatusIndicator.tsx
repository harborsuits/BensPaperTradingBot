import React from 'react';
import { RefreshCw, Wifi, WifiOff, Clock } from 'lucide-react';
import { useSyncStatus, useRefreshData } from '@/hooks/useSyncedData';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/Tooltip';

export function SyncStatusIndicator() {
  const { isConnected, lastSync, timeSinceSync } = useSyncStatus();
  const { refreshAll } = useRefreshData();
  const [isRefreshing, setIsRefreshing] = React.useState(false);
  
  const handleRefresh = async () => {
    setIsRefreshing(true);
    await refreshAll();
    setTimeout(() => setIsRefreshing(false), 1000);
  };
  
  const formatTimeSince = (ms: number | null) => {
    if (!ms) return 'Never';
    const seconds = Math.floor(ms / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    return `${hours}h ago`;
  };
  
  const getSyncStatus = () => {
    if (!isConnected) return { color: 'destructive', text: 'Disconnected' };
    if (!timeSinceSync) return { color: 'secondary', text: 'Syncing...' };
    if (timeSinceSync < 5000) return { color: 'success', text: 'Live' };
    if (timeSinceSync < 30000) return { color: 'default', text: 'Connected' };
    return { color: 'warning', text: 'Delayed' };
  };
  
  const status = getSyncStatus();
  
  return (
    <div className="flex items-center gap-2">
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Badge variant={status.color as any} className="flex items-center gap-1">
              {isConnected ? (
                <Wifi className="h-3 w-3" />
              ) : (
                <WifiOff className="h-3 w-3" />
              )}
              {status.text}
            </Badge>
          </TooltipTrigger>
          <TooltipContent>
            <div className="text-sm">
              <p>Connection: {isConnected ? 'Active' : 'Disconnected'}</p>
              <p>Last sync: {formatTimeSince(timeSinceSync)}</p>
            </div>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
      
      {lastSync && (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                <Clock className="h-3 w-3" />
                {formatTimeSince(timeSinceSync)}
              </div>
            </TooltipTrigger>
            <TooltipContent>
              <p className="text-sm">
                Last updated: {lastSync.toLocaleTimeString()}
              </p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      )}
      
      <Button
        variant="ghost"
        size="icon"
        onClick={handleRefresh}
        disabled={isRefreshing}
        className="h-8 w-8"
      >
        <RefreshCw 
          className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} 
        />
      </Button>
    </div>
  );
}
