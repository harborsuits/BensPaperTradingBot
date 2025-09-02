import React, { useMemo } from 'react';
import './PortfolioAllocation.css';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { Position } from '@/types/api.types';

interface PortfolioAllocationProps {
  positions: Position[];
  account: 'paper' | 'live';
}

// Generate colors for pie chart segments
const COLORS = [
  '#3B82F6', // blue-500
  '#10B981', // emerald-500
  '#F59E0B', // amber-500
  '#8B5CF6', // violet-500
  '#EC4899', // pink-500
  '#06B6D4', // cyan-500
  '#F97316', // orange-500
  '#14B8A6', // teal-500
  '#6366F1', // indigo-500
  '#EF4444', // red-500
];

const PortfolioAllocation: React.FC<PortfolioAllocationProps> = ({ positions }) => {
  // Calculate allocation data from positions
  const allocationData = useMemo(() => {
    if (!positions.length) return [];

    // Group by symbol - we could add sector grouping once available in the Position type
    const groupedBySymbol = positions.reduce((acc, position) => {
      const key = position.symbol;
      if (!acc[key]) {
        acc[key] = {
          name: position.symbol,
          value: 0,
          positions: [],
        };
      }
      
      const marketValue = position.quantity * position.last_price;
      acc[key].value += marketValue;
      acc[key].positions.push(position);
      
      return acc;
    }, {} as Record<string, { name: string; value: number; positions: Position[] }>);

    // Convert to array and sort by value
    return Object.values(groupedBySymbol)
      .sort((a, b) => b.value - a.value)
      .map((item, index) => ({
        ...item,
        color: COLORS[index % COLORS.length],
      }));
  }, [positions]);

  // Calculate total portfolio value
  const totalValue = useMemo(() => {
    return allocationData.reduce((sum, item) => sum + item.value, 0);
  }, [allocationData]);

  if (!positions.length) {
    return (
      <div className="flex items-center justify-center h-64 bg-card border border-border rounded-lg">
        <p className="text-muted-foreground">No positions to display</p>
      </div>
    );
  }

  return (
    <div className="bg-card border border-border rounded-lg p-4">
      <h3 className="text-md font-medium mb-4">Portfolio Allocation</h3>
      
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={allocationData}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={80}
              paddingAngle={2}
              dataKey="value"
              label={({ name, percent }) => `${name} (${(percent * 100).toFixed(1)}%)`}
              labelLine={false}
            >
              {allocationData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip
              formatter={(value: number) => [
                `$${value.toLocaleString(undefined, { 
                  minimumFractionDigits: 2, 
                  maximumFractionDigits: 2 
                })} (${((value / totalValue) * 100).toFixed(1)}%)`,
                'Allocation'
              ]}
            />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>
      
      <div className="mt-4 space-y-2">
        {allocationData.map((item, i) => (
          <div key={item.name} className="allocation-item">
            <div className="allocation-item-label">
              <div 
                className={`allocation-color-dot color-dot-${i % 10}`}
              />
              <span>{item.name}</span>
            </div>
            <div className="allocation-item-value">
              <div>${item.value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
              <div className="allocation-item-percentage">{((item.value / totalValue) * 100).toFixed(1)}%</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default PortfolioAllocation;
