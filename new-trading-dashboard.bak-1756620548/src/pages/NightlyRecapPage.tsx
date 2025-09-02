import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { toast } from 'react-hot-toast';
import { 
  TrendingUp, 
  AlertTriangle, 
  BarChart3, 
  Calendar, 
  ArrowDown, 
  ArrowUp, 
  AlertCircle,
  FileBarChart,
  RefreshCcw,
  ChevronRight
} from 'lucide-react';

import { Button } from '@/components/ui/Button';
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from '@/components/ui/Card';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/Tabs';
import { Badge } from '@/components/ui/Badge';
import TimeSeriesChart, { ChartDataPoint } from '@/components/ui/TimeSeriesChart';
import { DataTable } from '@/components/ui/DataTable';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/Table";
import { Progress } from '@/components/ui/Progress';

import { api } from '@/services/api';

// Types for nightly recap data
interface PerformanceMetric {
  name: string;
  value: number;
  change: number;
  benchmark?: number;
  threshold?: number;
  status: 'good' | 'warning' | 'critical';
}

interface StrategyAlert {
  metric: string;
  value: number;
  threshold: number;
  severity: 'warning' | 'medium' | 'high';
  message: string;
  compared_to?: number;
  deterioration?: number;
}

interface StrategySuggestion {
  action: string;
  current_weight: number;
  suggested_weight: number;
  reduction_percentage: number;
  reason: string;
  details: string[];
}

interface StrategyPerformance {
  strategy: string;
  current_weight: number;
  sharpe_ratio: number;
  win_rate: number;
  max_drawdown: number;
  total_pnl: number;
  trades_total: number;
  alerts: StrategyAlert[];
  suggestion?: StrategySuggestion;
  status: 'healthy' | 'warning' | 'critical';
}

interface BenchmarkPerformance {
  symbol: string;
  date: string;
  close: number;
  daily_return: number;
}

interface NightlyRecapData {
  date: string;
  daily_pnl: number;
  daily_return: number;
  equity_value: number;
  total_trades: number;
  win_rate: number;
  benchmarks: Record<string, BenchmarkPerformance>;
  performance_metrics: PerformanceMetric[];
  strategy_performance: StrategyPerformance[];
  historical_equity: ChartDataPoint[];
}

const NightlyRecapPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'strategies' | 'suggestions'>('overview');
  const [recapData, setRecapData] = useState<NightlyRecapData | null>(null);
  const [selectedDate, setSelectedDate] = useState<string>(new Date().toISOString().split('T')[0]);
  const [availableDates, setAvailableDates] = useState<string[]>([]);
  
  // Fetch available recap dates
  const { isLoading: isLoadingDates } = useQuery({
    queryKey: ['recap-dates'],
    queryFn: async () => {
      try {
        const response = await api.get('/api/recap/dates');
        if (response.data?.success) {
          const dates = response.data.dates || [];
          setAvailableDates(dates);
          
          // Set most recent date as default if available
          if (dates.length > 0) {
            setSelectedDate(dates[0]);
          }
        }
        return response.data;
      } catch (error) {
        toast.error('Failed to fetch available recap dates');
        return { success: false };
      }
    },
    staleTime: 60 * 60 * 1000, // 1 hour
  });
  
  // Fetch recap data for selected date
  const { 
    isLoading, 
    error, 
    refetch: refetchRecap
  } = useQuery({
    queryKey: ['recap-data', selectedDate],
    queryFn: async () => {
      try {
        const response = await api.get(`/api/recap/data?date=${selectedDate}`);
        if (response.data?.success) {
          setRecapData(response.data.data);
        }
        return response.data;
      } catch (error) {
        toast.error('Failed to fetch recap data');
        return { success: false };
      }
    },
    enabled: !!selectedDate,
    staleTime: 60 * 60 * 1000, // 1 hour
  });
  
  // Trigger a manual recap
  const triggerManualRecap = async () => {
    try {
      toast.loading('Running performance recap...', { id: 'recap-trigger' });
      const response = await api.post('/api/recap/trigger');
      
      if (response.data?.success) {
        toast.success('Recap completed successfully!', { id: 'recap-trigger' });
        // Refetch data after successful trigger
        refetchRecap();
      } else {
        toast.error(`Recap failed: ${response.data?.error || 'Unknown error'}`, { id: 'recap-trigger' });
      }
    } catch (error) {
      toast.error('Failed to trigger recap', { id: 'recap-trigger' });
    }
  };
  
  // Apply suggestion to update strategy weights
  const applySuggestion = async (strategy: string, newWeight: number) => {
    try {
      toast.loading(`Updating ${strategy} weight...`, { id: `update-${strategy}` });
      const response = await api.post('/api/strategies/update-weight', {
        strategy,
        weight: newWeight
      });
      
      if (response.data?.success) {
        toast.success(`${strategy} weight updated successfully!`, { id: `update-${strategy}` });
        refetchRecap();
      } else {
        toast.error(`Failed to update weight: ${response.data?.error || 'Unknown error'}`, { id: `update-${strategy}` });
      }
    } catch (error) {
      toast.error('Failed to update strategy weight', { id: `update-${strategy}` });
    }
  };
  
  // Render alert severity badge
  const renderSeverityBadge = (severity: string) => {
    switch (severity) {
      case 'high':
        return <Badge variant="destructive">Critical</Badge>;
      case 'medium':
        return <Badge variant="warning">Warning</Badge>;
      case 'warning':
        return <Badge variant="outline">Caution</Badge>;
      default:
        return <Badge variant="outline">{severity}</Badge>;
    }
  };
  
  // Render metric status indicator
  const renderMetricStatus = (status: string) => {
    switch (status) {
      case 'good':
        return <span className="text-green-500">●</span>;
      case 'warning':
        return <span className="text-amber-500">●</span>;
      case 'critical':
        return <span className="text-red-500">●</span>;
      default:
        return <span className="text-gray-500">●</span>;
    }
  };
  
  // Get class for numeric value display
  const getValueClass = (value: number, isNegativeBad = true) => {
    if (value === 0) return 'text-muted-foreground';
    if (isNegativeBad) {
      return value > 0 ? 'text-green-500' : 'text-red-500';
    } else {
      return value < 0 ? 'text-green-500' : 'text-red-500';
    }
  };
  
  return (
    <div className="container py-6 space-y-6">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight mb-1">Nightly Performance Recap</h1>
          <p className="text-muted-foreground">
            Automated analysis of trading performance with actionable insights
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground whitespace-nowrap">Recap Date:</span>
            <select
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
              className="h-8 rounded-md border border-input bg-background px-3 py-1 text-sm ring-offset-background"
              disabled={isLoadingDates || availableDates.length === 0}
            >
              {availableDates.length === 0 ? (
                <option value="">No reports available</option>
              ) : (
                availableDates.map(date => (
                  <option key={date} value={date}>
                    {new Date(date).toLocaleDateString(undefined, { weekday: 'short', year: 'numeric', month: 'short', day: 'numeric' })}
                  </option>
                ))
              )}
            </select>
          </div>
          
          <Button 
            variant="outline" 
            size="sm"
            onClick={refetchRecap}
            disabled={isLoading}
          >
            <RefreshCcw size={14} className="mr-1" />
            Refresh
          </Button>
          
          <Button 
            onClick={triggerManualRecap}
            size="sm"
          >
            <FileBarChart size={14} className="mr-1" />
            Run New Recap
          </Button>
        </div>
      </div>
      
      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as any)}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="overview">Performance Overview</TabsTrigger>
          <TabsTrigger value="strategies">Strategy Health</TabsTrigger>
          <TabsTrigger value="suggestions">Recommendations</TabsTrigger>
        </TabsList>
        
        <div className="mt-6">
          {isLoading ? (
            <div className="w-full h-96 flex items-center justify-center">
              <div className="flex flex-col items-center">
                <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
                <p className="mt-2 text-sm text-muted-foreground">Loading recap data...</p>
              </div>
            </div>
          ) : !recapData ? (
            <div className="w-full py-12 flex flex-col items-center justify-center border rounded-lg border-dashed">
              <AlertCircle size={36} className="text-muted-foreground mb-4" />
              <h3 className="text-lg font-medium">No recap data available</h3>
              <p className="text-sm text-muted-foreground mt-1 mb-4">
                {availableDates.length === 0 
                  ? "No performance recaps have been generated yet."
                  : "Failed to load performance recap data for the selected date."}
              </p>
              <Button onClick={triggerManualRecap} variant="outline">
                Generate Recap
              </Button>
            </div>
          ) : (
            <>
              {/* Daily Summary */}
              <TabsContent value="overview" className="mt-0">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-medium">Daily P&L</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="flex items-baseline justify-between">
                        <div className={`text-2xl font-bold ${getValueClass(recapData.daily_pnl)}`}>
                          ${recapData.daily_pnl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                        </div>
                        <div className={`flex items-center text-sm ${getValueClass(recapData.daily_return)}`}>
                          {recapData.daily_return > 0 ? (
                            <ArrowUp size={14} className="mr-1" />
                          ) : (
                            <ArrowDown size={14} className="mr-1" />
                          )}
                          {Math.abs(recapData.daily_return).toFixed(2)}%
                        </div>
                      </div>
                      
                      <div className="text-xs text-muted-foreground mt-1">
                        vs benchmarks:
                        {Object.entries(recapData.benchmarks).map(([symbol, data]) => (
                          <span key={symbol} className="ml-2">
                            {symbol}: <span className={getValueClass(data.daily_return)}>{data.daily_return > 0 ? '+' : ''}{data.daily_return.toFixed(2)}%</span>
                          </span>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-medium">Portfolio Value</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">
                        ${recapData.equity_value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </div>
                      <div className="text-xs text-muted-foreground mt-1">
                        {recapData.total_trades} trades | {recapData.win_rate.toFixed(1)}% win rate
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-medium">Alert Status</CardTitle>
                    </CardHeader>
                    <CardContent>
                      {recapData.strategy_performance.some(s => s.status === 'critical') ? (
                        <div className="flex items-center text-destructive">
                          <AlertTriangle size={18} className="mr-2" />
                          <span className="text-lg font-medium">Critical Alerts</span>
                        </div>
                      ) : recapData.strategy_performance.some(s => s.status === 'warning') ? (
                        <div className="flex items-center text-amber-500">
                          <AlertCircle size={18} className="mr-2" />
                          <span className="text-lg font-medium">Warnings Active</span>
                        </div>
                      ) : (
                        <div className="flex items-center text-green-500">
                          <TrendingUp size={18} className="mr-2" />
                          <span className="text-lg font-medium">All Systems Healthy</span>
                        </div>
                      )}
                      
                      <div className="text-xs text-muted-foreground mt-2">
                        {recapData.strategy_performance.filter(s => s.alerts.length > 0).length} strategies with alerts
                      </div>
                    </CardContent>
                  </Card>
                </div>
                
                {/* Performance Chart */}
                <Card className="mb-6">
                  <CardHeader>
                    <CardTitle className="flex items-center">
                      <BarChart3 size={18} className="mr-2" />
                      Portfolio Performance
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <TimeSeriesChart
                      data={recapData.historical_equity}
                      height={300}
                      xAxisType="date"
                      xAxisLabel="Date"
                      yAxisLabel="Value"
                      formatTooltipValue={(value) => `$${value.toLocaleString(undefined, { minimumFractionDigits: 2 })}`}
                    />
                  </CardContent>
                </Card>
                
                {/* Key Metrics Table */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center">
                      <TrendingUp size={18} className="mr-2" />
                      Key Performance Metrics
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Metric</TableHead>
                          <TableHead className="text-right">Current Value</TableHead>
                          <TableHead className="text-right">Change</TableHead>
                          <TableHead className="text-right">Benchmark</TableHead>
                          <TableHead className="text-right">Threshold</TableHead>
                          <TableHead className="w-[80px] text-right">Status</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {recapData.performance_metrics.map((metric, index) => (
                          <TableRow key={index}>
                            <TableCell className="font-medium">{metric.name}</TableCell>
                            <TableCell className="text-right">
                              {typeof metric.value === 'number' && metric.name.toLowerCase().includes('rate') 
                                ? `${metric.value.toFixed(2)}%` 
                                : typeof metric.value === 'number' && metric.name.toLowerCase().includes('ratio')
                                  ? metric.value.toFixed(2)
                                  : typeof metric.value === 'number'
                                    ? `$${metric.value.toLocaleString(undefined, { minimumFractionDigits: 2 })}`
                                    : metric.value
                              }
                            </TableCell>
                            <TableCell className={`text-right ${getValueClass(metric.change, !metric.name.toLowerCase().includes('drawdown'))}`}>
                              {metric.change > 0 ? '+' : ''}{metric.change.toFixed(2)}
                              {metric.name.toLowerCase().includes('rate') ? '%' : ''}
                            </TableCell>
                            <TableCell className="text-right">
                              {metric.benchmark !== undefined 
                                ? metric.name.toLowerCase().includes('rate')
                                  ? `${metric.benchmark.toFixed(2)}%`
                                  : metric.benchmark.toFixed(2)
                                : '-'
                              }
                            </TableCell>
                            <TableCell className="text-right">
                              {metric.threshold !== undefined 
                                ? metric.name.toLowerCase().includes('rate')
                                  ? `${metric.threshold.toFixed(2)}%`
                                  : metric.threshold.toFixed(2)
                                : '-'
                              }
                            </TableCell>
                            <TableCell className="text-right">{renderMetricStatus(metric.status)}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </CardContent>
                </Card>
              </TabsContent>
            
              {/* Strategy Health */}
              <TabsContent value="strategies" className="mt-0">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {recapData.strategy_performance.map((strategy, index) => (
                    <Card key={index} className={`
                      ${strategy.status === 'critical' ? 'border-red-400 border-2' : 
                        strategy.status === 'warning' ? 'border-amber-400 border-2' : ''}
                    `}>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-base">
                          <div className="flex justify-between items-center">
                            <span>{strategy.strategy}</span>
                            {strategy.status === 'critical' && (
                              <Badge variant="destructive">Critical</Badge>
                            )}
                            {strategy.status === 'warning' && (
                              <Badge variant="warning">Warning</Badge>
                            )}
                            {strategy.status === 'healthy' && (
                              <Badge variant="outline" className="bg-green-500/10 text-green-700 border-green-300">Healthy</Badge>
                            )}
                          </div>
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-2 mb-3">
                          <div className="flex justify-between items-center">
                            <span className="text-sm text-muted-foreground">Current Weight:</span>
                            <span className="font-medium">{(strategy.current_weight * 100).toFixed(1)}%</span>
                          </div>
                          
                          <div className="flex justify-between items-center">
                            <span className="text-sm text-muted-foreground">Sharpe Ratio:</span>
                            <span className={`font-medium ${strategy.sharpe_ratio < 0.5 ? 'text-amber-500' : 'text-green-500'}`}>
                              {strategy.sharpe_ratio.toFixed(2)}
                            </span>
                          </div>
                          
                          <div className="flex justify-between items-center">
                            <span className="text-sm text-muted-foreground">Win Rate:</span>
                            <span className={`font-medium ${strategy.win_rate < 45 ? 'text-amber-500' : 'text-green-500'}`}>
                              {strategy.win_rate.toFixed(1)}%
                            </span>
                          </div>
                          
                          <div className="flex justify-between items-center">
                            <span className="text-sm text-muted-foreground">Max Drawdown:</span>
                            <span className={`font-medium ${strategy.max_drawdown < -10 ? 'text-red-500' : 'text-muted-foreground'}`}>
                              {strategy.max_drawdown.toFixed(1)}%
                            </span>
                          </div>
                          
                          <div className="flex justify-between items-center">
                            <span className="text-sm text-muted-foreground">Total P&L:</span>
                            <span className={`font-medium ${getValueClass(strategy.total_pnl)}`}>
                              ${strategy.total_pnl.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                            </span>
                          </div>
                        </div>
                        
                        {strategy.alerts.length > 0 && (
                          <div className="mt-4">
                            <h4 className="text-sm font-semibold mb-2">Alerts ({strategy.alerts.length})</h4>
                            <div className="space-y-2 max-h-[120px] overflow-y-auto pr-1">
                              {strategy.alerts.map((alert, alertIndex) => (
                                <div key={alertIndex} className="text-xs p-2 bg-muted rounded-md">
                                  <div className="flex justify-between mb-1">
                                    <span className="font-medium">{alert.metric}</span>
                                    {renderSeverityBadge(alert.severity)}
                                  </div>
                                  <p className="text-muted-foreground">{alert.message}</p>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </CardContent>
                      
                      {strategy.suggestion && (
                        <CardFooter className="pt-0">
                          <div className="w-full text-xs bg-muted p-2 rounded-md border border-border">
                            <div className="font-medium mb-1 flex items-center text-amber-600">
                              <AlertTriangle size={12} className="mr-1" />
                              Suggested Action
                            </div>
                            <p className="text-muted-foreground mb-2">{strategy.suggestion.reason}</p>
                            <Button 
                              size="sm" 
                              variant="outline" 
                              className="w-full"
                              onClick={() => applySuggestion(strategy.strategy, strategy.suggestion?.suggested_weight || 0)}
                            >
                              Reduce Weight to {(strategy.suggestion.suggested_weight * 100).toFixed(1)}% 
                              <span className="ml-1 text-muted-foreground">
                                (-{strategy.suggestion.reduction_percentage.toFixed(1)}%)
                              </span>
                              <ChevronRight size={14} className="ml-1" />
                            </Button>
                          </div>
                        </CardFooter>
                      )}
                    </Card>
                  ))}
                </div>
              </TabsContent>
              
              {/* Suggestions */}
              <TabsContent value="suggestions" className="mt-0">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center">
                      <AlertTriangle size={18} className="mr-2" />
                      Actionable Recommendations
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {recapData.strategy_performance.filter(s => s.suggestion).length === 0 ? (
                      <div className="py-8 text-center">
                        <h3 className="text-lg font-medium mb-2">No recommendations available</h3>
                        <p className="text-sm text-muted-foreground">
                          All strategies are performing within acceptable parameters.
                        </p>
                      </div>
                    ) : (
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Strategy</TableHead>
                            <TableHead>Issue</TableHead>
                            <TableHead>Current Weight</TableHead>
                            <TableHead>Recommended Weight</TableHead>
                            <TableHead>Change</TableHead>
                            <TableHead className="text-right">Action</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {recapData.strategy_performance
                            .filter(s => s.suggestion)
                            .map((strategy, index) => (
                              <TableRow key={index}>
                                <TableCell className="font-medium">{strategy.strategy}</TableCell>
                                <TableCell>
                                  <div className="flex flex-col">
                                    <span className="text-sm">{strategy.suggestion?.reason}</span>
                                    <span className="text-xs text-muted-foreground mt-1">
                                      {strategy.alerts.length} issues detected
                                    </span>
                                  </div>
                                </TableCell>
                                <TableCell>{(strategy.current_weight * 100).toFixed(1)}%</TableCell>
                                <TableCell>{(strategy.suggestion?.suggested_weight || 0 * 100).toFixed(1)}%</TableCell>
                                <TableCell className="text-red-500">
                                  -{strategy.suggestion?.reduction_percentage.toFixed(1)}%
                                </TableCell>
                                <TableCell className="text-right">
                                  <Button 
                                    size="sm" 
                                    onClick={() => applySuggestion(
                                      strategy.strategy, 
                                      strategy.suggestion?.suggested_weight || 0
                                    )}
                                  >
                                    Apply
                                  </Button>
                                </TableCell>
                              </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    )}
                  </CardContent>
                </Card>
                
                <Card className="mt-6">
                  <CardHeader>
                    <CardTitle className="flex items-center">
                      <RefreshCcw size={18} className="mr-2" />
                      Automated Optimization
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex flex-col md:flex-row gap-6">
                      <div className="flex-1">
                        <h3 className="text-lg font-medium mb-2">Strategy Retraining</h3>
                        <p className="text-sm text-muted-foreground mb-4">
                          Automatically retrain strategies that have deteriorated beyond their thresholds.
                          This will use updated market data to fine-tune parameters.
                        </p>
                        <Button>
                          Retrain Flagged Strategies
                        </Button>
                      </div>
                      
                      <div className="flex-1">
                        <h3 className="text-lg font-medium mb-2">Weight Optimization</h3>
                        <p className="text-sm text-muted-foreground mb-4">
                          Run portfolio optimization to redistribute weights based on
                          recent performance and current market conditions.
                        </p>
                        <Button>
                          Optimize Portfolio Weights
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </>
          )}
        </div>
      </Tabs>
    </div>
  );
};

export default NightlyRecapPage;
