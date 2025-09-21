import { useQuery } from '@tanstack/react-query';
import { Power, Timer, CheckCircle, AlertCircle, XCircle } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

const AutoRunnerStrip = () => {
  const { data: status } = useQuery({
    queryKey: ['autoloop', 'status'],
    queryFn: async () => {
      const res = await fetch('/api/autoloop/status');
      if (res.status === 404) {
        // Treat as disabled; return a stable object so UI stays quiet
        return {
          enabled: false,
          isRunning: false,
          status: 'Disabled (endpoint not found)',
          lastRun: null,
          interval: 30000,
        };
      }
      if (!res.ok) {
        throw new Error(`API error: ${res.status} ${res.statusText}`);
      }
      return res.json();
    },
    refetchInterval: 5000,
    retry: (failureCount, error) => {
      // Don't retry on 4xx errors (client errors)
      if (error.message.includes(' 4') || error.message.includes('404')) {
        return false;
      }
      return failureCount < 3;
    },
    // Provide fallback data if the request fails
    placeholderData: {
      enabled: false,
      isRunning: false,
      status: 'Disabled',
      lastRun: null,
      interval: 30000
    }
  });

  // Normalize backend variants
  const isRunning = (status as any)?.isRunning ?? (status as any)?.is_running ?? (status as any)?.running ?? false;
  const interval = (status as any)?.interval ?? (status as any)?.interval_ms ?? (status as any)?.tick_ms ?? 30000;
  const lastRunRaw = (status as any)?.lastRun ?? (status as any)?.last_run ?? (status as any)?.timestamp ?? null;

  const nextCheckIn = lastRunRaw
    ? interval - (new Date().getTime() - new Date(lastRunRaw).getTime())
    : interval;

  const nextCheckSeconds = Math.max(0, Math.floor((nextCheckIn || 0) / 1000));
  
  const StatusIcon = isRunning ? CheckCircle : (status?.status?.startsWith('BLOCKED') ? XCircle : AlertCircle);

  return (
    <div className="bg-card border border-border rounded-md p-2 text-xs flex items-center justify-between">
      <div className="flex items-center space-x-4">
        <span className={`flex items-center font-bold ${isRunning ? 'text-green-500' : 'text-muted-foreground'}`}>
          <Power size={14} className="mr-1" />
          Auto: {isRunning ? 'ON' : 'OFF'}
        </span>
        <span>
          Next check: 00:{nextCheckSeconds.toString().padStart(2, '0')}
        </span>
        <span>
          Last: {lastRunRaw ? formatDistanceToNow(new Date(lastRunRaw), { addSuffix: true }) : 'never'}
        </span>
        <span className="flex items-center">
            Status: {status?.status || '...'}
        </span>
      </div>
    </div>
  );
};

export default AutoRunnerStrip;
