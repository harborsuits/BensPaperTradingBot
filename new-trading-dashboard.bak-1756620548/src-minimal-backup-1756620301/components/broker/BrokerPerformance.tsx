import React from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { 
  RefreshCw, 
  AlertTriangle, 
  ArrowRight
} from 'lucide-react';
import { brokerApi } from '@/services/brokerApi';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { toast } from 'react-hot-toast';

// Define types for broker performance data
interface BrokerPerformanceMetrics {
  pnl: number;
  fillRate: number;
  slippage: number;
  latency: number;
  errorRate: number;
  uptime: number;
  status: 'online' | 'degraded' | 'offline' | 'circuit_broken';
  isActive: boolean;
}

interface BrokerPerformanceData {
  [broker: string]: BrokerPerformanceMetrics;
}

interface BrokerPerformanceProps {
  className?: string;
}

const BrokerPerformance: React.FC<BrokerPerformanceProps> = ({ className }) => {
  const queryClient = useQueryClient();
  
  // Fetch broker performance data
  const { 
    data: brokerData, 
    isLoading, 
    error,
    refetch 
  } = useQuery({
    queryKey: ['broker-performance'],
    queryFn: async () => {
      try {
        // This will be replaced with actual API call once backend endpoint is implemented
        const response = await fetch('/api/broker/performance');
        if (!response.ok) {
          throw new Error(`Error ${response.status}: ${response.statusText}`);
        }
        return await response.json();
      } catch (err: any) {
        console.error('Error fetching broker performance data:', err);
        toast.error(`Failed to load broker data: ${err.message || 'Unknown error'}`);
        throw err;
      }
    },
    staleTime: 60000, // 1 minute
    retry: 2,
    // Fallback to mock data for development
    placeholderData: {
      'Primary Broker (Tradier)': {
        pnl: 1250.75,
        fillRate: 0.98,
        slippage: 0.12,
        latency: 42,
        errorRate: 0.02,
        uptime: 0.999,
        status: 'online',
        isActive: true
      },
      'Secondary Broker (Alpaca)': {
        pnl: 950.25,
        fillRate: 0.96,
        slippage: 0.18,
        latency: 48,
        errorRate: 0.04,
        uptime: 0.995,
        status: 'online',
        isActive: false
      },
      'Backup Broker': {
        pnl: 0,
        fillRate: 0.95,
        slippage: 0.20,
        latency: 56,
        errorRate: 0.05,
        uptime: 0.990,
        status: 'circuit_broken',
        isActive: false
      }
    }
  });

  // Function to trigger broker failover
  const handleBrokerFailover = async (from: string, to: string) => {
    const toastId = `failover-${from}-${to}`;
    try {
      toast.loading(`Initiating failover from ${from} to ${to}...`, { id: toastId });
      
      // Use the broker API service
      const response = await brokerApi.triggerFailover(from, to);
      
      if (!response.success) {
        throw new Error(response.error || 'Failover request failed');
      }
      
      toast.success(`Successfully failed over to ${response.data.activeBroker}`, { id: toastId });
      
      // Refresh broker data
      refetch();
      
    } catch (err: any) {
      console.error('Error during broker failover:', err);
      toast.error(`Failover failed: ${err.message || 'Unknown error'}`, { id: toastId });
      
      // Provide user with retry option
      setTimeout(() => {
        toast((
          <div className="flex flex-col">
            <p>Failover attempt failed. Would you like to retry?</p>
            <Button 
              variant="outline"
              size="sm"
              className="mt-2"
              onClick={() => handleBrokerFailover(from, to)}
            >
              Retry Failover
            </Button>
          </div>
        ), { duration: 10000 });
      }, 1000);
    }
  };
  
  // Render status badge with appropriate color
  const renderStatusBadge = (status: string) => {
    switch(status) {
      case 'online':
        return <Badge className="bg-green-500">Online</Badge>;
      case 'degraded':
        return <Badge className="bg-amber-500">Degraded</Badge>;
      case 'offline':
        return <Badge className="bg-red-500">Offline</Badge>;
      case 'circuit_broken':
        return <Badge className="bg-purple-500">Circuit Broken</Badge>;
      default:
        return <Badge className="bg-slate-500">Unknown</Badge>;
    }
  };
  
  // Get the name of the active broker
  const getActiveBroker = () => {
    if (!brokerData) return null;
    
    return Object.entries(brokerData).find(
      ([_, metrics]) => (metrics as BrokerPerformanceMetrics).isActive
    )?.[0] || null;
  };
  
  const activeBroker = getActiveBroker();
  // Failover buttons are shown individually for each broker based on its status
  
  if (isLoading) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="text-xl">Broker Intelligence</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="h-8 w-48 bg-muted/20 animate-pulse rounded-md"></div>
            <div className="space-y-2">
              {[1, 2, 3].map(i => (
                <div key={i} className="h-20 bg-muted/20 animate-pulse rounded-md"></div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }
  
  if (error || !brokerData) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="text-xl">Broker Intelligence</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="py-4 flex flex-col items-center justify-center">
            <AlertTriangle size={40} className="text-amber-500 mb-2" />
            <p className="text-muted-foreground text-center mb-2">Could not load broker performance data</p>
            <Button onClick={() => refetch()} size="sm">Retry</Button>
          </div>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <Card className={className}>
      <CardHeader className="relative">
        <CardTitle className="text-xl">Broker Intelligence</CardTitle>
        <Button
          variant="ghost" 
          size="icon"
          className="absolute top-2 right-2 p-1 h-8 w-8"
          onClick={() => {
            refetch();
            toast.success('Refreshing broker data...');
          }}
          aria-label="Refresh"
          title="Refresh broker data"
        >
          <RefreshCw size={16} />
          <span className="sr-only">Refresh broker data</span>
        </Button>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {activeBroker && (
            <div className="flex items-center mb-4">
              <span className="font-medium" id="active-broker-label">Active Broker:</span>
              <Badge className="bg-primary ml-2" aria-labelledby="active-broker-label">{activeBroker}</Badge>
            </div>
          )}
          
          <div className="space-y-4">
            {Object.entries(brokerData).map(([broker, metrics]) => (
              <Card key={broker} className={`border ${metrics.isActive ? 'border-primary' : 'border-border'}`}>
                <CardContent className="p-4">
                  <div className="flex justify-between items-center mb-2">
                    <div>
                      <h3 className="font-medium">{broker}</h3>
                      <div className="flex items-center space-x-2 mt-1">
                        {renderStatusBadge(metrics.status)}
                        {metrics.isActive && <Badge variant="outline" className="bg-blue-500/10 text-blue-500">Active</Badge>}
                      </div>
                    </div>
                    
                    {/* Show failover button only for inactive brokers that are online */}
                    {!(metrics as BrokerPerformanceMetrics).isActive && (metrics as BrokerPerformanceMetrics).status === 'online' && activeBroker && (
                      <Button 
                        variant="outline" 
                        size="sm"
                        onClick={() => handleBrokerFailover(activeBroker, broker)}
                        className="text-xs"
                        aria-label={`Failover from ${activeBroker} to ${broker}`}
                      >
                        Failover <ArrowRight size={12} className="ml-1" />
                        <span className="sr-only">from {activeBroker} to {broker}</span>
                      </Button>
                    )}
                  </div>
                  
                  <div className="grid grid-cols-3 gap-2 mt-4">
                    <div className="text-center p-2 bg-background rounded-md">
                      <p className="text-xs text-muted-foreground mb-1">PnL</p>
                      <p className={`text-sm font-bold ${(metrics as BrokerPerformanceMetrics).pnl >= 0 ? 'text-bull' : 'text-bear'}`}>
                        ${(metrics as BrokerPerformanceMetrics).pnl.toFixed(2)}
                      </p>
                    </div>
                    <div className="text-center p-2 bg-background rounded-md">
                      <p className="text-xs text-muted-foreground mb-1">Fill Rate</p>
                      <p className="text-sm font-bold">{((metrics as BrokerPerformanceMetrics).fillRate * 100).toFixed(1)}%</p>
                    </div>
                    <div className="text-center p-2 bg-background rounded-md">
                      <p className="text-xs text-muted-foreground mb-1">Slippage</p>
                      <p className="text-sm font-bold">{((metrics as BrokerPerformanceMetrics).slippage * 100).toFixed(2)}%</p>
                    </div>
                    <div className="text-center p-2 bg-background rounded-md">
                      <p className="text-xs text-muted-foreground mb-1">Latency</p>
                      <p className="text-sm font-bold">{(metrics as BrokerPerformanceMetrics).latency} ms</p>
                    </div>
                    <div className="text-center p-2 bg-background rounded-md">
                      <p className="text-xs text-muted-foreground mb-1">Error Rate</p>
                      <p className="text-sm font-bold">{((metrics as BrokerPerformanceMetrics).errorRate * 100).toFixed(2)}%</p>
                    </div>
                    <div className="text-center p-2 bg-background rounded-md">
                      <p className="text-xs text-muted-foreground mb-1">Uptime</p>
                      <p className="text-sm font-bold">{((metrics as BrokerPerformanceMetrics).uptime * 100).toFixed(2)}%</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default BrokerPerformance;
