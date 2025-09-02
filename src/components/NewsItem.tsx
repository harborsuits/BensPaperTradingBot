import React from 'react';
import { NewsItem as NewsItemType } from '@/hooks/useNews';

interface NewsItemProps {
  item: NewsItemType;
}

export function NewsItem({ item }: NewsItemProps) {
  function formatTimeAgo(dateString: string) {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    
    if (diffMins < 60) {
      return `${diffMins}m ago`;
    } else if (diffHours < 24) {
      return `${diffHours}h ago`;
    } else {
      return date.toLocaleDateString();
    }
  }
  
  return (
    <div className={`bg-muted rounded-md p-3 mb-3 ${
      item.sentiment === 'positive' ? 'border-l-2 border-green-500' : 
      item.sentiment === 'negative' ? 'border-l-2 border-red-500' : ''
    }`}>
      <p className="text-sm font-medium">{item.title}</p>
      <div className="flex justify-between mt-1">
        <p className="text-xs text-muted-foreground">{item.source}</p>
        <p className="text-xs text-muted-foreground">{formatTimeAgo(item.timestamp)}</p>
      </div>
    </div>
  );
} 