import React from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Smile, Frown, Meh, Clock, RefreshCw } from 'lucide-react';
import { SentimentData, contextApi } from '@/services/contextApi';
import { Card, CardContent } from '@/components/ui/Card';
import { formatRelativeTime } from '@/utils/date';
import { useContextUpdates } from '@/hooks/useWebSocketSubscriptions';
import { toast } from 'react-hot-toast';
import styles from './SentimentAnalysis.module.css';

interface SentimentAnalysisProps {
  className?: string;
}

const SentimentAnalysis: React.FC<SentimentAnalysisProps> = ({ className }) => {
  const queryClient = useQueryClient();
  
  const { 
    data, 
    isLoading, 
    error,
    refetch
  } = useQuery({
    queryKey: ['market-sentiment'],
    queryFn: async () => {
      try {
        const response = await contextApi.getSentiment();
        if (!response.success) {
          throw new Error(response.error || 'Failed to fetch sentiment data');
        }
        return response.data;
      } catch (err: any) {
        console.error('Error fetching sentiment data:', err);
        toast.error(`Failed to load sentiment data: ${err.message || 'Unknown error'}`);
        throw err;
      }
    },
    staleTime: 300000, // 5 minutes
    retry: 2,
  });
  
  // Subscribe to sentiment updates via WebSocket
  useContextUpdates(
    undefined, // No regime change handler
    undefined, // No feature update handler
    (newsData) => {
      // When we get news data that affects sentiment, refresh the sentiment data
      if (newsData && newsData.sentiment_impact) {
        // Show notification about sentiment-impacting news
        toast(`New market news affecting sentiment: ${newsData.headline}`, {
          icon: newsData.sentiment_impact > 0 ? '📈' : 
                newsData.sentiment_impact < 0 ? '📉' : 'ℹ️'
        });
        
        // Update the sentiment data if included
        if (newsData.updated_sentiment) {
          queryClient.setQueryData(['market-sentiment'], newsData.updated_sentiment);
        } else {
          // Otherwise just refetch
          refetch();
        }
      }
    }
  );
  
  if (isLoading) {
    return (
      <Card className={className}>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-medium text-muted-foreground">Market Sentiment</h3>
              <div className="h-8 w-32 bg-muted/20 animate-pulse rounded-md mt-2"></div>
            </div>
            <div className="h-10 w-10 bg-muted/20 animate-pulse rounded-full"></div>
          </div>
          <div className="mt-4 space-y-2">
            <div className="h-4 w-full bg-muted/20 animate-pulse rounded-md"></div>
            <div className="h-4 w-3/4 bg-muted/20 animate-pulse rounded-md"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error || !data) {
    return (
      <Card className={className}>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-medium text-muted-foreground">Market Sentiment</h3>
              <p className="text-lg font-bold">Unknown</p>
            </div>
            <Meh className="text-muted" size={24} />
          </div>
          <p className="text-xs text-muted-foreground mt-2">Unable to fetch sentiment data</p>
        </CardContent>
      </Card>
    );
  }

  const sentiment = data as SentimentData;
  const scorePercent = Math.round((sentiment.overall_score + 1) * 50); // Convert -1 to 1 scale to 0-100%
  
  const getSentimentIcon = () => {
    switch (sentiment.market_sentiment) {
      case 'positive':
        return <Smile className="text-bull" size={24} />;
      case 'negative':
        return <Frown className="text-bear" size={24} />;
      default:
        return <Meh className="text-muted-foreground" size={24} />;
    }
  };
  
  const getSentimentColor = () => {
    switch (sentiment.market_sentiment) {
      case 'positive':
        return 'bg-bull/10 text-bull border-bull/20';
      case 'negative':
        return 'bg-bear/10 text-bear border-bear/20';
      default:
        return 'bg-muted/10 text-muted-foreground border-muted/20';
    }
  };

  return (
    <Card className={className}>
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-medium text-muted-foreground">Market Sentiment</h3>
            <p className="text-lg font-bold capitalize">{sentiment.market_sentiment}</p>
          </div>
          {getSentimentIcon()}
        </div>
        
        <div className="mt-2">
          <div className="flex justify-between text-xs mb-1">
            <span>Sentiment Score: {scorePercent}%</span>
            <span className="flex items-center text-muted-foreground">
              <Clock size={12} className="mr-1" /> 
              {formatRelativeTime(new Date(sentiment.timestamp))}
            </span>
          </div>
          <div className={styles.sentimentBar}>
            <div className={styles.sentimentBarWrapper}>
              <div 
                className={`${getSentimentColor().split(' ')[0]} ${styles.sentimentFill} ${styles[`sentiment${Math.floor(scorePercent / 10) * 10}`]}`}
                data-testid="sentiment-score-bar"
              ></div>
            </div>
          </div>
        </div>
        
        <div className="mt-3 space-y-2">
          {sentiment.positive_factors.length > 0 && (
            <div>
              <h4 className="text-xs font-medium mb-1 text-bull">Positive Factors:</h4>
              <ul className="text-xs list-disc list-inside space-y-1">
                {sentiment.positive_factors.slice(0, 3).map((factor, i) => (
                  <li key={i}>{factor}</li>
                ))}
              </ul>
            </div>
          )}
          
          {sentiment.negative_factors.length > 0 && (
            <div>
              <h4 className="text-xs font-medium mb-1 text-bear">Negative Factors:</h4>
              <ul className="text-xs list-disc list-inside space-y-1">
                {sentiment.negative_factors.slice(0, 3).map((factor, i) => (
                  <li key={i}>{factor}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
        
        <p className="text-xs text-muted-foreground mt-2">Source: {sentiment.source}</p>
        
        {/* Refresh button */}
        <button 
          onClick={() => { 
            refetch();
            toast.success('Refreshing sentiment data...');
          }}
          className="absolute top-3 right-3 p-1 rounded-full bg-primary/10 hover:bg-primary/20 text-primary transition-colors"
          title="Refresh sentiment data"
        >
          <RefreshCw size={14} />
        </button>
      </CardContent>
    </Card>
  );
};

export default SentimentAnalysis;
