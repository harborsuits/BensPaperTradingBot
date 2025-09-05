import React from 'react';

interface NewsItemProps {
  item: {
    id: string;
    title: string;
    summary?: string;
    source?: string;
    timestamp?: string;
    sentiment?: 'positive' | 'neutral' | 'negative';
    impact?: number;
    url?: string;
  };
}

export const NewsItem: React.FC<NewsItemProps> = ({ item }) => {
  // Format the timestamp if it exists
  const formattedTime = item.timestamp ? new Date(item.timestamp).toLocaleTimeString() : '';
  
  // Determine sentiment color
  const sentimentColor = {
    positive: 'text-green-400',
    neutral: 'text-blue-400',
    negative: 'text-red-400',
  }[item.sentiment || 'neutral'];
  
  // Format impact if it exists (0-100 scale)
  const impactFormatted = typeof item.impact === 'number' ? `${Math.round(item.impact)}%` : '';
  
  return (
    <div className="news-item border border-border rounded-md p-3 mb-3 hover:bg-card/80 transition-colors">
      <div className="flex justify-between items-start mb-1">
        <h3 className="font-medium text-sm">{item.title}</h3>
        {item.sentiment && (
          <span className={`text-xs font-medium ${sentimentColor}`}>
            {item.sentiment.charAt(0).toUpperCase() + item.sentiment.slice(1)}
            {impactFormatted && ` (${impactFormatted})`}
          </span>
        )}
      </div>
      
      {item.summary && (
        <p className="text-xs text-muted-foreground mb-2 line-clamp-2">{item.summary}</p>
      )}
      
      <div className="flex justify-between items-center text-xs text-muted-foreground">
        <span>{item.source || 'Unknown source'}</span>
        {formattedTime && <span>{formattedTime}</span>}
      </div>
      
      {item.url && (
        <a 
          href={item.url} 
          target="_blank" 
          rel="noopener noreferrer"
          className="mt-2 text-xs text-primary hover:underline block"
        >
          Read more
        </a>
      )}
    </div>
  );
};
