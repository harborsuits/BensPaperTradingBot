import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  TooltipProps
} from 'recharts';
import { cn } from '@/utils/cn';

export interface ChartDataPoint {
  timestamp: string | number;
  [key: string]: any;
}

export interface ChartSeries {
  dataKey: string;
  name: string;
  color: string;
  type?: 'line' | 'area';
  strokeWidth?: number;
  fillOpacity?: number;
}

interface TimeSeriesChartProps {
  data: ChartDataPoint[];
  series: ChartSeries[];
  height?: number | string;
  className?: string;
  showGrid?: boolean;
  showXAxis?: boolean;
  showYAxis?: boolean;
  showLegend?: boolean;
  showTooltip?: boolean;
  xAxisDataKey?: string;
  formatXAxisTick?: (value: any) => string;
  formatYAxisTick?: (value: any) => string;
  formatTooltipValue?: (value: any, name: string, entry: any) => React.ReactNode;
  formatTooltipLabel?: (label: any) => React.ReactNode;
}

const TimeSeriesChart: React.FC<TimeSeriesChartProps> = ({
  data,
  series,
  height = 300,
  className,
  showGrid = true,
  showXAxis = true,
  showYAxis = true,
  showLegend = true,
  showTooltip = true,
  xAxisDataKey = 'timestamp',
  formatXAxisTick,
  formatYAxisTick,
  formatTooltipValue,
  formatTooltipLabel,
}) => {
  // Determine if we need area chart or line chart
  const chartType = useMemo(() => {
    return series.some(s => s.type === 'area') ? 'area' : 'line';
  }, [series]);
  
  // Custom tooltip component
  const CustomTooltip = ({ active, payload, label }: TooltipProps<any, any>) => {
    if (!active || !payload || !payload.length) {
      return null;
    }

    return (
      <div className="bg-card/95 backdrop-blur-sm border border-border p-2 rounded shadow-md text-sm">
        <div className="font-medium pb-1 border-b border-border mb-1">
          {formatTooltipLabel ? formatTooltipLabel(label) : label}
        </div>
        <div className="space-y-1">
          {payload.map((entry: any, index: number) => (
            <div key={`item-${index}`} className="flex items-center gap-2">
              <div 
                className="w-3 h-3 rounded-full" 
                style={{ backgroundColor: entry.color }}
              />
              <span className="font-medium">{entry.name}:</span>
              <span>
                {formatTooltipValue 
                  ? formatTooltipValue(entry.value, entry.name, entry) 
                  : entry.value
                }
              </span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className={cn("w-full h-full", className)}>
      <ResponsiveContainer width="100%" height={height}>
        {chartType === 'area' ? (
          <AreaChart data={data}>
            {showGrid && <CartesianGrid stroke="#2d3748" strokeDasharray="3 3" />}
            
            {showXAxis && (
              <XAxis 
                dataKey={xAxisDataKey} 
                tick={{ fill: '#94a3b8' }} 
                tickLine={{ stroke: '#475569' }}
                axisLine={{ stroke: '#475569' }}
                tickFormatter={formatXAxisTick}
              />
            )}
            
            {showYAxis && (
              <YAxis 
                tick={{ fill: '#94a3b8' }} 
                tickLine={{ stroke: '#475569' }}
                axisLine={{ stroke: '#475569' }}
                tickFormatter={formatYAxisTick}
              />
            )}
            
            {showTooltip && <Tooltip content={<CustomTooltip />} />}
            
            {showLegend && (
              <Legend 
                wrapperStyle={{ paddingTop: 10 }}
                formatter={(value) => <span className="text-foreground">{value}</span>}
              />
            )}
            
            {series.map((s, index) => (
              s.type === 'area' || chartType === 'area' ? (
                <Area 
                  key={`${s.dataKey}-${index}`}
                  type="monotone"
                  dataKey={s.dataKey}
                  name={s.name}
                  stroke={s.color}
                  fill={s.color}
                  fillOpacity={s.fillOpacity || 0.2}
                  strokeWidth={s.strokeWidth || 2}
                  activeDot={{ r: 6 }}
                />
              ) : (
                <Line 
                  key={`${s.dataKey}-${index}`}
                  type="monotone"
                  dataKey={s.dataKey}
                  name={s.name}
                  stroke={s.color}
                  strokeWidth={s.strokeWidth || 2}
                  dot={false}
                  activeDot={{ r: 6 }}
                />
              )
            ))}
          </AreaChart>
        ) : (
          <LineChart data={data}>
            {showGrid && <CartesianGrid stroke="#2d3748" strokeDasharray="3 3" />}
            
            {showXAxis && (
              <XAxis 
                dataKey={xAxisDataKey} 
                tick={{ fill: '#94a3b8' }} 
                tickLine={{ stroke: '#475569' }}
                axisLine={{ stroke: '#475569' }}
                tickFormatter={formatXAxisTick}
              />
            )}
            
            {showYAxis && (
              <YAxis 
                tick={{ fill: '#94a3b8' }} 
                tickLine={{ stroke: '#475569' }}
                axisLine={{ stroke: '#475569' }}
                tickFormatter={formatYAxisTick}
              />
            )}
            
            {showTooltip && <Tooltip content={<CustomTooltip />} />}
            
            {showLegend && (
              <Legend 
                wrapperStyle={{ paddingTop: 10 }}
                formatter={(value) => <span className="text-foreground">{value}</span>}
              />
            )}
            
            {series.map((s, index) => (
              <Line 
                key={`${s.dataKey}-${index}`}
                type="monotone"
                dataKey={s.dataKey}
                name={s.name}
                stroke={s.color}
                strokeWidth={s.strokeWidth || 2}
                dot={false}
                activeDot={{ r: 6 }}
              />
            ))}
          </LineChart>
        )}
      </ResponsiveContainer>
    </div>
  );
};

export default TimeSeriesChart;
