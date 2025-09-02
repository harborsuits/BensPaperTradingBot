import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { useWebSocketChannel, useWebSocketChannelMessage } from '@/services/websocketManager';
import { Newspaper, TrendingUp, TrendingDown, AlertTriangle } from 'lucide-react';

// Types for news items
interface NewsItem {
  id: string;
  headline: string;
  source: string;
  url?: string;
  published_at: string;
  sentiment_score: number;
  summary?: string;
  keywords?: string[];
  impact?: 'high' | 'medium' | 'low';
}

const NewsSentimentFeed: React.FC = () => {
  const [newsItems, setNewsItems] = useState<NewsItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdateTime, setLastUpdateTime] = useState<Date | null>(null);

  // Subscribe to the market_context and alerts channels
  useWebSocketChannel('market_context');
  useWebSocketChannel('alerts');

  // Fetch initial news data
  useEffect(() => {
    const fetchNewsData = async () => {
      try {
        const response = await fetch('/api/context/news');
        if (response.ok) {
          const data = await response.json();
          setNewsItems(data);
          setLastUpdateTime(new Date());
        }
      } catch (error) {
        console.error('Error fetching news data:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchNewsData();
  }, []);

  // Listen for news updates via WebSocket
  useWebSocketChannelMessage('market_context', (message) => {
    if (message.event === 'sentiment_update') {
      // Update sentiment indicators but not necessarily the news items
      setLastUpdateTime(new Date());
    }
  });

  // Listen for news alerts
  useWebSocketChannelMessage('alerts', (message) => {
    if (message.event === 'news_alert') {
      const newsItem = message.payload;
      // Add the new item to the top of the list
      setNewsItems((prevItems) => [newsItem, ...prevItems.slice(0, 19)]);
      setLastUpdateTime(new Date());
    }
  });

  // Get sentiment icon and color based on score
  const getSentimentDisplay = (score: number) => {
    if (score > 0.3) {
      return { icon: <TrendingUp size={16} />, color: 'text-green-500', label: 'Positive' };
    } else if (score < -0.3) {
      return { icon: <TrendingDown size={16} />, color: 'text-red-500', label: 'Negative' };
    } else {
      return { icon: <AlertTriangle size={16} />, color: 'text-amber-500', label: 'Neutral' };
    }
  };

  // Format relative time
  const getRelativeTime = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMins / 60);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${Math.floor(diffHours / 24)}d ago`;
  };

  return (
    <Card className="h-full">
      <CardHeader className="pb-2">
        <div className="flex justify-between items-center">
          <CardTitle className="flex items-center text-lg">
            <Newspaper size={18} className="mr-2" />
            News & Sentiment
          </CardTitle>
          {lastUpdateTime && (
            <span className="text-xs text-muted-foreground">
              Updated: {lastUpdateTime.toLocaleTimeString()}
            </span>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-3">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="h-16 bg-muted/20 animate-pulse rounded-md"></div>
            ))}
          </div>
        ) : newsItems.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8">
            <AlertTriangle size={24} className="text-muted-foreground mb-2" />
            <p className="text-muted-foreground">No news items available</p>
          </div>
        ) : (
          <div className="space-y-3 max-h-[500px] overflow-y-auto pr-1">
            {newsItems.map((item) => {
              const sentiment = getSentimentDisplay(item.sentiment_score);
              return (
                <div 
                  key={item.id}
                  className="p-3 border rounded-md hover:bg-muted/20 transition-colors"
                >
                  <div className="flex justify-between items-start mb-1">
                    <h3 className="font-medium text-sm line-clamp-2">{item.headline}</h3>
                    <div className={`flex items-center ${sentiment.color} text-xs ml-2`}>
                      {sentiment.icon}
                      <span className="ml-1">{sentiment.label}</span>
                    </div>
                  </div>
                  <div className="flex justify-between items-center text-xs text-muted-foreground">
                    <span>{item.source}</span>
                    <span>{getRelativeTime(item.published_at)}</span>
                  </div>
                  {item.impact && (
                    <div className="mt-1">
                      <span className={`text-xs px-1.5 py-0.5 rounded ${
                        item.impact === 'high' ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300' :
                        item.impact === 'medium' ? 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300' :
                        'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300'
                      }`}>
                        {item.impact.toUpperCase()} IMPACT
                      </span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default NewsSentimentFeed;
