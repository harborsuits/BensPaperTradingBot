import React from 'react';
import { useAllocation } from '../../hooks/useAllocation'; 
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { Spinner } from '../ui/Spinner';
import { ErrorMessage } from '../ui/ErrorMessage';

// Default colors if no colors provided in data
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#a480ff', '#ff80e5'];

export function AllocationBreakdown() {
  const { data: allocation, isLoading, error, refetch } = useAllocation();
  
  if (isLoading) return <Spinner />;
  if (error) return <ErrorMessage retry={() => refetch()} />;
  
  // Format for pie chart
  const pieData = allocation?.allocations.map(item => ({
    name: item.name,
    value: item.percent,
    amount: item.value,
  })) || [];
  
  return (
    <div className="allocation-breakdown-container h-full w-full">
      {pieData.length > 0 ? (
        <div className="h-full w-full">
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={80}
                paddingAngle={2}
                dataKey="value"
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                labelLine={false}
              >
                {pieData.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={entry.color || COLORS[index % COLORS.length]} 
                  />
                ))}
              </Pie>
              <Tooltip 
                formatter={(value) => `${value.toFixed(2)}%`}
                contentStyle={{ backgroundColor: '#1e1e1e', borderColor: '#333' }}
                itemStyle={{ color: '#fff' }}
                labelStyle={{ color: '#aaa' }}
              />
              <Legend 
                formatter={(value) => <span style={{ color: '#ccc' }}>{value}</span>}
              />
            </PieChart>
          </ResponsiveContainer>
          
          {/* Text details below chart */}
          <div className="allocation-items mt-4 space-y-2">
            {pieData.map((item, index) => (
              <div 
                key={`item-${index}`} 
                className="allocation-item flex justify-between items-center p-2 bg-muted/30 rounded"
              >
                <span className="text-sm font-medium">{item.name}</span>
                <div className="flex items-center">
                  <span className="text-sm mr-2">${item.amount.toLocaleString()}</span>
                  <span className="text-xs bg-muted px-2 py-1 rounded">{item.value.toFixed(1)}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="flex items-center justify-center h-full bg-muted/30 rounded text-muted-foreground">
          No allocation data available
        </div>
      )}
    </div>
  );
} 