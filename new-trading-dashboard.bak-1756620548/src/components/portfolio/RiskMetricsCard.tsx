import React from 'react';
import { Activity } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { useWebSocketChannel, useWebSocketMessage } from '@/services/websocket';
import { Progress } from '@/components/ui/Progress';

interface RiskMetric {
  name: string;
  value: number;
  change?: number;
  threshold?: number;
  maxThreshold?: number;
}

interface RiskMetricsCardProps {
  className?: string;
  account: 'paper' | 'live';
}

const RiskMetricsCard: React.FC<RiskMetricsCardProps> = ({ className, account }) => {
  const [metrics, setMetrics] = React.useState<RiskMetric[]>([
    { name: 'Portfolio VaR', value: 2.35, threshold: 5, maxThreshold: 8 },
    { name: 'Concentration', value: 28.4, threshold: 40, maxThreshold: 60 },
    { name: 'Correlation', value: 0.42, threshold: 0.6, maxThreshold: 0.8 },
    { name: 'Market Exposure', value: 65.8, threshold: 80, maxThreshold: 90 },
  ]);
  
  const [lastUpdated, setLastUpdated] = React.useState<Date | null>(null);
  
  // Subscribe to risk metrics channel
  useWebSocketChannel('risk_metrics' as any, true);
  
  // Handle real-time risk metrics updates
  useWebSocketMessage<any>(
    'risk_metrics_update',
    (message) => {
      if (message.data && message.data.account === account) {
        setMetrics(message.data.metrics);
        setLastUpdated(new Date());
      }
    },
    [account]
  );
  
  const getThresholdColor = (value: number, threshold?: number, maxThreshold?: number) => {
    if (!threshold || !maxThreshold) return 'bg-primary';
    
    if (value > maxThreshold) return 'bg-destructive';
    if (value > threshold) return 'bg-amber-500';
    return 'bg-green-500';
  };
  
  return (
    <Card className={className}>
      <CardHeader className="pb-2">
        <CardTitle className="text-base flex items-center">
          <Activity className="mr-2 h-4 w-4" />
          Risk Metrics
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {metrics.map((metric) => (
            <div key={metric.name} className="space-y-1">
              <div className="flex justify-between text-sm">
                <span>{metric.name}</span>
                <span className="font-medium">{metric.value}%</span>
              </div>
              <div className="relative pt-1">
                <div className="overflow-hidden h-1.5 mb-1 text-xs flex rounded bg-muted/30">
                  <Progress 
                    value={metric.value} 
                    className={`w-full h-full flex flex-col justify-center text-center text-white ${
                      getThresholdColor(metric.value, metric.threshold, metric.maxThreshold)
                    }`}
                  />
                </div>
                {metric.threshold && metric.maxThreshold && (
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Low</span>
                    <span>Medium</span>
                    <span>High</span>
                  </div>
                )}
              </div>
            </div>
          ))}
          
          {lastUpdated && (
            <div className="text-xs text-muted-foreground pt-2 border-t border-border mt-2">
              Updated: {lastUpdated.toLocaleTimeString()}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default RiskMetricsCard;
