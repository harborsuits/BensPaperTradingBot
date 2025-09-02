import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { ExternalLink, Clock, ArrowUpRight, ArrowDownRight, Minus, Tag } from 'lucide-react';
import { NewsItem, contextApi } from '@/services/contextApi';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { ScrollArea } from '@/components/ui/ScrollArea';
import { formatRelativeTime } from '@/utils/date';

interface NewsFeedProps {
  className?: string;
  limit?: number;
}

const NewsFeed: React.FC<NewsFeedProps> = ({ className, limit = 8 }) => {
  const [expanded, setExpanded] = useState<string | null>(null);
  
  const { data, isLoading, error } = useQuery({
    queryKey: ['news-feed', limit],
    queryFn: async () => {
      const response = await contextApi.getLatestNews(limit);
      return response.success ? response.data : null;
    },
    staleTime: 300000, // 5 minutes
  });

  const getSentimentIcon = (score: number) => {
    if (score > 0.2) return <ArrowUpRight className="text-bull" size={16} />;
    if (score < -0.2) return <ArrowDownRight className="text-bear" size={16} />;
    return <Minus className="text-muted-foreground" size={16} />;
  };

  const getSentimentColor = (score: number) => {
    if (score > 0.2) return 'text-bull';
    if (score < -0.2) return 'text-bear';
    return 'text-muted-foreground';
  };

  const getImpactBadge = (impact: 'high' | 'medium' | 'low') => {
    switch (impact) {
      case 'high':
        return <Badge variant="destructive">High Impact</Badge>;
      case 'medium':
        return <Badge variant="default">Medium Impact</Badge>;
      case 'low':
        return <Badge variant="outline">Low Impact</Badge>;
    }
  };

  const renderNewsItem = (item: NewsItem) => {
    const isExpandedItem = expanded === item.id;
    
    return (
      <div key={item.id} className="py-3 border-b last:border-0 border-border/50">
        <div className="flex justify-between items-start gap-2">
          <h3 
            className="text-sm font-medium hover:text-primary cursor-pointer"
            onClick={() => setExpanded(isExpandedItem ? null : item.id)}
          >
            {item.headline}
          </h3>
          <div className="flex items-center gap-1 whitespace-nowrap">
            {getSentimentIcon(item.sentiment_score)}
            <span className={`text-xs ${getSentimentColor(item.sentiment_score)}`}>
              {(item.sentiment_score * 100).toFixed(0)}%
            </span>
          </div>
        </div>
        
        <div className="flex items-center gap-2 text-xs text-muted-foreground mt-1">
          <span>{item.source}</span>
          <span>•</span>
          <span className="flex items-center">
            <Clock size={12} className="mr-1" />
            {formatRelativeTime(new Date(item.published_at))}
          </span>
        </div>
        
        {isExpandedItem && (
          <div className="mt-2 space-y-2">
            <p className="text-sm">{item.summary}</p>
            
            <div className="flex flex-wrap items-center gap-1.5 mt-2">
              {getImpactBadge(item.impact)}
              
              {item.categories.map(category => (
                <div key={category} className="flex items-center text-xs text-muted-foreground bg-muted/30 px-2 py-0.5 rounded-full">
                  <Tag size={10} className="mr-1" />
                  {category}
                </div>
              ))}
              
              {item.symbols && item.symbols.map(symbol => (
                <div key={symbol} className="text-xs font-medium px-2 py-0.5 bg-primary/10 text-primary rounded-full">
                  ${symbol}
                </div>
              ))}
            </div>
            
            <div className="pt-1">
              <a 
                href={item.url} 
                target="_blank" 
                rel="noopener noreferrer"
                className="inline-flex items-center text-xs text-primary hover:underline"
              >
                Read full article <ExternalLink size={12} className="ml-1" />
              </a>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderSkeleton = () => {
    return Array(limit).fill(0).map((_, index) => (
      <div key={index} className="py-3 border-b last:border-0 border-border/50">
        <div className="flex justify-between">
          <div className="h-5 bg-muted/20 animate-pulse rounded-md w-3/4"></div>
          <div className="h-5 bg-muted/20 animate-pulse rounded-md w-10"></div>
        </div>
        <div className="flex gap-2 mt-2">
          <div className="h-4 bg-muted/20 animate-pulse rounded-md w-16"></div>
          <div className="h-4 bg-muted/20 animate-pulse rounded-md w-24"></div>
        </div>
      </div>
    ));
  };

  if (error) {
    return (
      <Card className={className}>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Market News</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">Failed to load news feed</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardHeader className="pb-2">
        <div className="flex justify-between items-center">
          <CardTitle className="text-base">Market News</CardTitle>
          <Button variant="ghost" size="sm" className="h-8 px-2">
            <ExternalLink size={16} className="mr-1" /> View All
          </Button>
        </div>
      </CardHeader>
      <CardContent className="px-4 py-0">
        <ScrollArea className="h-[400px] pr-4">
          {isLoading ? (
            renderSkeleton()
          ) : Array.isArray(data) ? (
            data.map(renderNewsItem)
          ) : Array.isArray((data as any)?.items) ? (
            (data as any).items.map(renderNewsItem)
          ) : (
            <div className="py-6 text-sm text-muted-foreground">No news available.</div>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
};

export default NewsFeed;
