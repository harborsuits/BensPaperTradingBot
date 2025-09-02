import React from 'react';
import { usePortfolio } from '../../hooks/usePortfolio';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Spinner } from '../ui/Spinner';
import { ErrorMessage } from '../ui/ErrorMessage';

export function PortfolioGraph() {
  const { data: portfolio, isLoading, error, refetch } = usePortfolio();
  
  if (isLoading) return <Spinner />;
  if (error) return <ErrorMessage retry={() => refetch()} />;
  
  return (
    <div className="portfolio-graph-container h-64 w-full">
      {portfolio?.history && portfolio.history.length > 0 ? (
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={portfolio.history}
            margin={{
              top: 10,
              right: 30,
              left: 0,
              bottom: 0,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis 
              dataKey="date" 
              tick={{ fill: '#aaa' }}
              tickFormatter={(value) => {
                const date = new Date(value);
                return `${date.getMonth() + 1}/${date.getDate()}`;
              }}
            />
            <YAxis 
              tick={{ fill: '#aaa' }}
              tickFormatter={(value) => `$${value.toLocaleString()}`}
              width={80}
            />
            <Tooltip 
              formatter={(value) => [`$${Number(value).toLocaleString()}`, 'Portfolio Value']}
              labelFormatter={(label) => new Date(label).toLocaleDateString()}
              contentStyle={{ backgroundColor: '#1e1e1e', borderColor: '#333' }}
              itemStyle={{ color: '#fff' }}
              labelStyle={{ color: '#aaa' }}
            />
            <Area 
              type="monotone" 
              dataKey="value" 
              stroke="#8884d8" 
              fill="#8884d8" 
              fillOpacity={0.3} 
            />
          </AreaChart>
        </ResponsiveContainer>
      ) : (
        <div className="flex items-center justify-center h-full bg-muted/30 rounded text-muted-foreground">
          No portfolio data available
        </div>
      )}
    </div>
  );
} 