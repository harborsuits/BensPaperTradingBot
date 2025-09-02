import React from 'react';
import { useAllocation } from '../../hooks/useAllocation';
import { Spinner } from '../ui/Spinner';
import { ErrorMessage } from '../ui/ErrorMessage';

// Simple pie chart component
export function AllocationChart() {
  const { data, isLoading, error, refetch } = useAllocation();
  
  if (isLoading) return <Spinner />;
  if (error) return <ErrorMessage retry={() => refetch()} />;
  
  return (
    <div className="allocation-chart bg-card border border-border rounded-lg p-4">
      <h3 className="text-lg font-bold mb-3">Asset Allocation</h3>
      
      {/* Placeholder for allocation chart - in a real app we would use Recharts or similar */}
      <div className="pie-chart-placeholder h-48 flex items-center justify-center bg-muted/20 rounded-md mb-3">
        <div className="text-muted-foreground">Allocation chart will appear here</div>
      </div>
      
      {/* Legend */}
      <div className="allocation-legend grid grid-cols-2 gap-2">
        {data?.categories?.map((category, index) => (
          <div key={index} className="flex items-center">
            <div 
              className="w-3 h-3 rounded-sm mr-2" 
              style={{ 
                backgroundColor: getColorForIndex(index) 
              }}
            ></div>
            <span className="text-sm truncate">{category.name}</span>
            <span className="text-sm ml-auto font-medium">{category.value}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// Helper function to get colors
function getColorForIndex(index: number): string {
  const colors = [
    '#3b82f6', // blue
    '#06b6d4', // cyan
    '#8b5cf6', // violet
    '#ec4899', // pink
    '#f97316', // orange
    '#84cc16', // lime
    '#14b8a6', // teal
    '#f43f5e', // rose
  ];
  
  return colors[index % colors.length];
} 