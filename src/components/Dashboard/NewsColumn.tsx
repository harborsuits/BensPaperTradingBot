import React from 'react';
import { useNews } from '../../hooks/useNews';
import { Spinner } from '../ui/Spinner';
import { ErrorMessage } from '../ui/ErrorMessage';

interface NewsColumnProps {
  title: string;
  category: string;
}

export function NewsColumn({ title, category }: NewsColumnProps) {
  const { data, isLoading, error, refetch } = useNews();
  
  if (isLoading) return <Spinner />;
  if (error) return <ErrorMessage retry={() => refetch()} />;
  
  // Filter news for this category (in a real app, we would likely fetch by category)
  const newsItems = data?.items || [];
  
  return (
    <div className="news-column bg-card border border-border rounded-lg p-4">
      <h3 className="text-lg font-bold mb-3 text-white">{title} News</h3>
      
      <div className="news-list space-y-3">
        {newsItems.length > 0 ? (
          newsItems.map((item, index) => (
            <div 
              key={item.id || index} 
              className="news-item p-2 border-b border-border last:border-b-0"
            >
              <h4 className="font-medium text-sm text-white">{item.title}</h4>
              <div className="flex justify-between items-center mt-1">
                <span className="text-xs text-muted-foreground">{item.source}</span>
                <span className="text-xs text-muted-foreground">{item.date}</span>
              </div>
              <div className="mt-1">
                <span className={`inline-block text-xs px-2 py-0.5 rounded-full ${
                  item.sentiment === 'positive' 
                    ? 'bg-green-500/10 text-green-600' 
                    : item.sentiment === 'negative'
                    ? 'bg-red-500/10 text-red-600'
                    : 'bg-blue-500/10 text-blue-600'
                }`}>
                  {item.sentiment || 'neutral'}
                </span>
              </div>
            </div>
          ))
        ) : (
          <div className="text-center text-muted-foreground py-4">
            No news available
          </div>
        )}
      </div>
    </div>
  );
} 