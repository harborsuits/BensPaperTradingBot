import React from 'react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  TooltipProps
} from 'recharts';
import { Card } from '@/components/ui/Card';
import styles from './FitnessTrendChart.module.css';

// Helper function to get CSS color class based on entry name or color
const getColorClass = (nameOrColor: string): string => {
  const colorMap: Record<string, string> = {
    'Best Fitness': styles.colorBlue,
    'Average Fitness': styles.colorPurple,
    'Population Diversity': styles.colorTeal,
    'Return': styles.colorGreen,
    'Drawdown': styles.colorRed,
    'Volatility': styles.colorOrange,
    'Sharpe': styles.colorPink,
    // Add fallback colors for hex values
    '#3182CE': styles.colorBlue,
    '#805AD5': styles.colorPurple,
    '#38B2AC': styles.colorTeal,
    '#48BB78': styles.colorGreen,
    '#E53E3E': styles.colorRed,
    '#ED8936': styles.colorOrange,
    '#D53F8C': styles.colorPink,
  };
  
  return colorMap[nameOrColor] || styles.colorGray; // Default to gray if no match
};

interface FitnessDataPoint {
  generation: number;
  bestFitness: number;
  avgFitness: number;
  diversityScore?: number;
  [key: string]: any;
}

interface FitnessTrendChartProps {
  data: FitnessDataPoint[];
  height?: number | string;
  className?: string;
}

/**
 * Custom tooltip component for the fitness trend chart
 * Uses a cleaner approach to styling
 */
const CustomTooltip = ({ active, payload, label }: TooltipProps<number, string>) => {
  if (active && payload && payload.length) {
    return (
      <Card className="bg-navy-800 border-navy-700 shadow-lg p-2">
        <p className="text-gray-300 text-xs font-medium">Generation {label}</p>
        {payload.map((entry, index) => {
          return (
            <div 
              key={`item-${index}`} 
              className={styles.tooltipItem}
            >
              <span className={`text-xs ${getColorClass(entry.name || entry.color || '#A0AEC0')}`}>
                {entry.name}:
              </span>
              <span className={`text-xs font-medium ml-2 ${styles.entryColoredText}`}>
                {typeof entry.value === 'number' ? entry.value.toFixed(4) : entry.value}
              </span>
            </div>
          );
        })}
      </Card>
    );
  }

  return null;
};

const FitnessTrendChart: React.FC<FitnessTrendChartProps> = ({ 
  data, 
  height = '100%',
  className = '' 
}) => {
  // If no data, show placeholder
  if (!data || data.length === 0) {
    return (
      <div className={`flex items-center justify-center h-full text-gray-500 ${className}`}>
        No fitness data available yet
      </div>
    );
  }

  // Calculate domain values to make the chart more readable
  const bestFitnessValues = data.map(d => d.bestFitness);
  const avgFitnessValues = data.map(d => d.avgFitness);
  
  const minBestFitness = Math.min(...bestFitnessValues);
  const maxBestFitness = Math.max(...bestFitnessValues);
  
  const minAvgFitness = Math.min(...avgFitnessValues);
  const maxAvgFitness = Math.max(...avgFitnessValues);
  
  // Define a buffer to add to the min/max
  const buffer = Math.max(0.1, Math.abs(maxBestFitness - minBestFitness) * 0.1);
  
  // Calculate domain
  const domainMin = Math.floor(Math.min(minBestFitness, minAvgFitness) - buffer);
  const domainMax = Math.ceil(Math.max(maxBestFitness, maxAvgFitness) + buffer);
  
  return (
    <div className={`h-${typeof height === 'number' ? height : 'full'} ${className}`}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={data}
          margin={{
            top: 5,
            right: 20,
            left: 5,
            bottom: 5,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#2D3748" opacity={0.3} />
          <XAxis 
            dataKey="generation"
            stroke="#A0AEC0" 
            fontSize={12}
            tickLine={{ stroke: '#4A5568' }}
            axisLine={{ stroke: '#4A5568' }}
          />
          <YAxis 
            stroke="#A0AEC0" 
            fontSize={12}
            tickLine={{ stroke: '#4A5568' }}
            axisLine={{ stroke: '#4A5568' }}
            domain={[domainMin, domainMax]}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="bestFitness" 
            name="Best Fitness"
            stroke="#3182CE" 
            strokeWidth={2}
            activeDot={{ r:
              6, fill: '#3182CE', stroke: '#E2E8F0' }} 
            dot={{ r: 2, fill: '#3182CE' }}
            isAnimationActive={true}
            animationDuration={500}
          />
          <Line 
            type="monotone" 
            dataKey="avgFitness" 
            name="Average Fitness"
            stroke="#805AD5" 
            strokeWidth={1.5}
            activeDot={{ r: 5, fill: '#805AD5', stroke: '#E2E8F0' }} 
            dot={{ r: 0 }}
            isAnimationActive={true}
            animationDuration={500}
          />
          {data.some(d => typeof d.diversityScore !== 'undefined') && (
            <Line 
              type="monotone" 
              dataKey="diversityScore" 
              name="Population Diversity"
              stroke="#38B2AC" 
              strokeWidth={1}
              strokeDasharray="5 5"
              activeDot={{ r: 4, fill: '#38B2AC', stroke: '#E2E8F0' }} 
              dot={{ r: 0 }}
              isAnimationActive={true}
              animationDuration={500}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default FitnessTrendChart;
