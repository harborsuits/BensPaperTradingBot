import React, { useState } from 'react';
import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { NewsItem } from '@/components/NewsItem';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { ArrowLeft, RefreshCw, Filter } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const NewsPage: React.FC = () => {
  const { symbol } = useParams<{ symbol?: string }>();
  const navigate = useNavigate();
  const [selectedSentiment, setSelectedSentiment] = useState<'all' | 'positive' | 'neutral' | 'negative'>('all');

  // Fetch news data
  const { data: newsData, isLoading, error, refetch } = useQuery({
    queryKey: ['news', symbol],
    queryFn: async () => {
      const response = await fetch('/api/news');
      return response.json();
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
  });

  // Filter news by symbol if specified
  const filteredNews = newsData?.filter((item: any) => {
    if (!symbol) return true;
    return item.symbols?.includes(symbol.toUpperCase());
  }).filter((item: any) => {
    if (selectedSentiment === 'all') return true;
    return item.sentiment === selectedSentiment;
  }) || [];

  const sentimentCounts = {
    all: filteredNews.length,
    positive: filteredNews.filter((item: any) => item.sentiment === 'positive').length,
    neutral: filteredNews.filter((item: any) => item.sentiment === 'neutral').length,
    negative: filteredNews.filter((item: any) => item.sentiment === 'negative').length,
  };

  return (
    <div className="w-full py-6">
      <div className="w-full max-w-7xl mx-auto px-4 xl:px-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => navigate(-1)}
              className="flex items-center gap-2"
            >
              <ArrowLeft size={16} />
              Back
            </Button>
            <div>
              <h1 className="text-2xl font-bold">
                {symbol ? `${symbol} News` : 'Market News'}
              </h1>
              <p className="text-sm text-muted-foreground">
                Real-time news analysis and sentiment tracking
              </p>
            </div>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetch()}
            className="flex items-center gap-2"
          >
            <RefreshCw size={16} />
            Refresh
          </Button>
        </div>

        {/* Filters */}
        <Card className="mb-6">
          <div className="p-4">
            <div className="flex items-center gap-4 mb-4">
              <Filter size={16} />
              <span className="font-medium">Filter by Sentiment:</span>
            </div>
            <div className="flex gap-2">
              {(['all', 'positive', 'neutral', 'negative'] as const).map((sentiment) => (
                <Button
                  key={sentiment}
                  variant={selectedSentiment === sentiment ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setSelectedSentiment(sentiment)}
                  className="flex items-center gap-2"
                >
                  <span className={`w-2 h-2 rounded-full ${
                    sentiment === 'positive' ? 'bg-green-500' :
                    sentiment === 'negative' ? 'bg-red-500' :
                    sentiment === 'neutral' ? 'bg-blue-500' : 'bg-gray-500'
                  }`} />
                  {sentiment.charAt(0).toUpperCase() + sentiment.slice(1)}
                  <span className="text-xs opacity-70">({sentimentCounts[sentiment]})</span>
                </Button>
              ))}
            </div>
          </div>
        </Card>

        {/* News Feed */}
        <div className="space-y-4">
          {isLoading ? (
            <Card className="p-8 text-center">
              <RefreshCw className="w-8 h-8 mx-auto mb-4 animate-spin text-muted-foreground" />
              <p className="text-muted-foreground">Loading news...</p>
            </Card>
          ) : error ? (
            <Card className="p-8 text-center border-red-200">
              <p className="text-red-600 mb-2">Failed to load news</p>
              <Button onClick={() => refetch()} variant="outline" size="sm">
                Try Again
              </Button>
            </Card>
          ) : filteredNews.length === 0 ? (
            <Card className="p-8 text-center">
              <p className="text-muted-foreground">
                {symbol
                  ? `No news found for ${symbol} with current filters`
                  : 'No news available'
                }
              </p>
            </Card>
          ) : (
            <>
              <div className="text-sm text-muted-foreground mb-4">
                Showing {filteredNews.length} article{filteredNews.length !== 1 ? 's' : ''}
                {symbol && ` for ${symbol}`}
              </div>
              {filteredNews.map((item: any) => (
                <NewsItem
                  key={item.id}
                  item={{
                    id: item.id,
                    title: item.title || item.headline,
                    summary: item.summary || item.content,
                    source: item.source,
                    timestamp: item.published_at || item.timestamp,
                    sentiment: item.sentiment_score > 0.2 ? 'positive' :
                              item.sentiment_score < -0.2 ? 'negative' : 'neutral',
                    impact: Math.abs(item.sentiment_score || 0) * 100,
                    url: item.url
                  }}
                />
              ))}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default NewsPage;
