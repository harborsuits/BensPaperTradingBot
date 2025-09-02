import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { 
  Smile, 
  Frown, 
  Meh, 
  Clock, 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle,
  Info 
} from 'lucide-react';
import { SentimentData, SentimentHistoryItem, SentimentAnomaly, contextApi } from '@/services/contextApi';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { formatRelativeTime } from '@/utils/date';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/Tabs';
import { Badge } from '@/components/ui/Badge';
import Tooltip from '@/components/ui/Tooltip';
import styles from './EnhancedSentimentAnalysis.module.css';
import { toArray, len, safeSplit } from '@/utils/safe';
import ZeroState from '@/components/ui/ZeroState';

interface EnhancedSentimentAnalysisProps {
  className?: string;
}

const EnhancedSentimentAnalysis: React.FC<EnhancedSentimentAnalysisProps> = ({ className }) => {
  const [activeView, setActiveView] = useState('current');
  
  // Fetch current sentiment
  const { 
    data: sentimentData, 
    isLoading: isLoadingSentiment, 
    error: sentimentError 
  } = useQuery({
    queryKey: ['market-sentiment'],
    queryFn: async () => {
      const response = await contextApi.getSentiment();
      return response.success ? response.data : null;
    },
    staleTime: 300000, // 5 minutes
  });
  
  // Fetch historical sentiment data
  const { 
    data: historyData, 
    isLoading: isLoadingHistory
  } = useQuery({
    queryKey: ['sentiment-history'],
    queryFn: async () => {
      const response = await contextApi.getSentimentHistory();
      return response.success ? response.data : [];
    },
    staleTime: 300000, // 5 minutes
  });
  
  // Fetch sentiment anomalies
  const { 
    data: anomaliesData, 
    isLoading: isLoadingAnomalies
  } = useQuery({
    queryKey: ['sentiment-anomalies'],
    queryFn: async () => {
      const response = await contextApi.getSentimentAnomalies();
      return response.success ? response.data : [];
    },
    staleTime: 300000, // 5 minutes
  });
  
  if (isLoadingSentiment || isLoadingHistory || isLoadingAnomalies) {
    return (
      <Card className={className}>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Market Sentiment</CardTitle>
        </CardHeader>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
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

  if (sentimentError || !sentimentData) {
    return (
      <Card className={className}>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Market Sentiment</CardTitle>
        </CardHeader>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-lg font-bold">Unknown</p>
            </div>
            <Meh className="text-muted" size={24} />
          </div>
          <p className="text-xs text-muted-foreground mt-2">Unable to fetch sentiment data</p>
        </CardContent>
      </Card>
    );
  }

  const sentiment = sentimentData as SentimentData;
  const scorePercent = Math.round(((sentiment?.overall_score || 0) + 1) * 50); // Convert -1 to 1 scale to 0-100%
  const history = toArray<SentimentHistoryItem>(historyData);
  const anomalies = toArray<SentimentAnomaly>(anomaliesData);
  
  // Check if we have valid data to display
  if (!sentiment || !sentiment.market_sentiment) {
    return (
      <Card className={className}>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Market Sentiment</CardTitle>
        </CardHeader>
        <CardContent className="p-4">
          <ZeroState
            title="No sentiment data"
            message="Unable to load sentiment analysis."
            action={{ label: "Retry", onClick: () => window.location.reload() }}
          />
        </CardContent>
      </Card>
    );
  }
  
  const getSentimentIcon = (sentimentType: string) => {
    switch (sentimentType) {
      case 'positive':
        return <Smile className="text-bull" size={24} />;
      case 'negative':
        return <Frown className="text-bear" size={24} />;
      default:
        return <Meh className="text-muted-foreground" size={24} />;
    }
  };
  
  const getSentimentColor = (sentimentType: string) => {
    switch (sentimentType) {
      case 'positive':
        return 'bg-bull/10 text-bull border-bull/20';
      case 'negative':
        return 'bg-bear/10 text-bear border-bear/20';
      default:
        return 'bg-muted/10 text-muted-foreground border-muted/20';
    }
  };

  const getSentimentTrend = () => {
    if (len(history) < 2) return null;
    
    const currentItem = history[history.length - 1];
    const prevItem = history[history.length - 2];
    
    if (!currentItem || !prevItem) return null;
    
    const currentScore = currentItem.score || 0;
    const prevScore = prevItem.score || 0;
    const diff = currentScore - prevScore;
    
    if (Math.abs(diff) < 0.1) {
      return (
        <Badge variant="outline" className="text-xs bg-gray-100">
          <Meh size={12} className="mr-1" />
          Stable
        </Badge>
      );
    } else if (diff > 0) {
      return (
        <Badge variant="outline" className="text-xs bg-green-100 text-green-800 border-green-200">
          <TrendingUp size={12} className="mr-1" />
          Improving
        </Badge>
      );
    } else {
      return (
        <Badge variant="outline" className="text-xs bg-red-100 text-red-800 border-red-200">
          <TrendingDown size={12} className="mr-1" />
          Deteriorating
        </Badge>
      );
    }
  };

  const renderSentimentChart = () => {
    const maxChartItems = 7; // Show last 7 items
    const chartItems = history.slice(-maxChartItems);
    
    if (len(chartItems) === 0) {
      return (
        <div className="mt-3">
          <ZeroState
            title="No trend data available"
            message="Waiting for historical sentiment data."
          />
        </div>
      );
    }
    
    return (
      <div className="mt-3">
        <div className="flex justify-between text-xs mb-2">
          <span>Sentiment Trend (Last {len(chartItems)} days)</span>
        </div>
        <div className="h-16 flex items-end justify-between">
          {chartItems.map((item: SentimentHistoryItem, i: number) => {
            // Convert -1 to 1 scale to 0-100%
            const barHeight = Math.round((item.score + 1) * 50); 
            const barColor = item.sentiment === 'positive' 
              ? 'bg-bull' 
              : item.sentiment === 'negative' 
                ? 'bg-bear' 
                : 'bg-gray-400';
              
            const isAnomaly = anomalies.some((a: SentimentAnomaly) => 
              new Date(a.timestamp).toDateString() === new Date(item.timestamp).toDateString()
            );
            
            return (
              <div key={i} className="flex flex-col items-center">
                <Tooltip content={`${new Date(item.timestamp).toLocaleDateString()} - ${item.sentiment} (${barHeight}%)`}>
                  <div className="relative">
                    {isAnomaly && (
                      <AlertTriangle 
                        size={14} 
                        className="absolute -top-5 left-1/2 transform -translate-x-1/2 text-amber-500" 
                      />
                    )}
                    <div 
                      className={`${barColor} ${styles.historyBar} ${styles[`barHeight${Math.max(5, Math.round(barHeight / 5) * 5)}`]}`} 
                    />
                  </div>
                </Tooltip>
                <span className="text-[10px] mt-1">
                  {new Date(item.timestamp).toLocaleDateString(undefined, { day: 'numeric' })}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const renderAnomalies = () => {
    if (anomalies.length === 0) {
      return (
        <div className="px-2 py-4 text-center">
          <Info size={18} className="mx-auto mb-2 text-muted-foreground" />
          <p className="text-sm text-muted-foreground">No sentiment anomalies detected</p>
        </div>
      );
    }
    
    return (
      <div className="space-y-2 mt-2">
        {anomalies.slice(0, 5).map((anomaly: SentimentAnomaly, i: number) => (
          <div key={i} className="border-l-2 border-amber-400 pl-2 py-1">
            <div className="flex items-center">
              <AlertTriangle size={14} className="text-amber-500 mr-1.5" />
              <span className="text-xs font-medium">
                {new Date(anomaly.timestamp).toLocaleDateString()}
              </span>
            </div>
            <p className="text-xs mt-0.5">{anomaly.description}</p>
            <div className="flex gap-1 mt-1">
              {anomaly.tags?.map((tag: string, idx: number) => (
                <Badge key={idx} variant="outline" className="text-[10px] py-0 px-1">
                  {tag}
                </Badge>
              ))}
            </div>
          </div>
        ))}
      </div>
    );
  };

  return (
    <Card className={className}>
      <CardHeader className="pb-2">
        <CardTitle className="text-base flex items-center justify-between">
          <span>Market Sentiment</span>
          <div className="flex items-center gap-2">
            {getSentimentTrend()}
            <span className="text-xs flex items-center text-muted-foreground">
              <Clock size={12} className="mr-1" /> 
              {formatRelativeTime(new Date(sentiment.timestamp))}
            </span>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="p-4 pt-2">
        <Tabs value={activeView} onValueChange={setActiveView} className="w-full">
          <TabsList className="grid grid-cols-3 mb-3">
            <TabsTrigger value="current">Current</TabsTrigger>
            <TabsTrigger value="trend">Trend</TabsTrigger>
            <TabsTrigger value="anomalies">
              Anomalies
              {anomalies.length > 0 && (
                <Badge variant="destructive" className="ml-1.5 h-4 w-4 p-0 flex items-center justify-center">
                  {anomalies.length}
                </Badge>
              )}
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="current">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-lg font-bold capitalize">{sentiment.market_sentiment}</p>
              </div>
              {getSentimentIcon(sentiment.market_sentiment)}
            </div>
            
            <div className="mt-2">
              <div className="flex justify-between text-xs mb-1">
                <span>Sentiment Score: {scorePercent}%</span>
              </div>
              <div className={styles.sentimentBar}>
                <div className={styles.sentimentBarWrapper}>
                  <div 
                    className={`${safeSplit(getSentimentColor(sentiment.market_sentiment), ' ')[0] || ''} ${styles.sentimentFill} ${styles[`fillPercent${Math.round(scorePercent / 5) * 5}`] || styles.fillPercent50}`}
                  ></div>
                </div>
              </div>
            </div>
            
            <div className="mt-3 space-y-2">
              {len(sentiment?.positive_factors) > 0 && (
                <div>
                  <h4 className="text-xs font-medium mb-1 text-bull">Positive Factors:</h4>
                  <ul className="text-xs list-disc list-inside space-y-1">
                    {toArray(sentiment.positive_factors).slice(0, 3).map((factor, i) => (
                      <li key={i}>{factor}</li>
                    ))}
                  </ul>
                </div>
              )}
              
              {len(sentiment?.negative_factors) > 0 && (
                <div>
                  <h4 className="text-xs font-medium mb-1 text-bear">Negative Factors:</h4>
                  <ul className="text-xs list-disc list-inside space-y-1">
                    {toArray(sentiment.negative_factors).slice(0, 3).map((factor, i) => (
                      <li key={i}>{factor}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
            
            <p className="text-xs text-muted-foreground mt-2">Source: {sentiment.source}</p>
          </TabsContent>
          
          <TabsContent value="trend">
            {renderSentimentChart()}
            <div className="flex justify-between mt-3 text-xs text-muted-foreground">
              <div>
                <span className="inline-block w-3 h-3 bg-bull rounded-sm mr-1"></span>
                Positive
              </div>
              <div>
                <span className="inline-block w-3 h-3 bg-gray-400 rounded-sm mr-1"></span>
                Neutral
              </div>
              <div>
                <span className="inline-block w-3 h-3 bg-bear rounded-sm mr-1"></span>
                Negative
              </div>
              <div>
                <AlertTriangle size={12} className="inline text-amber-500 mr-1" />
                Anomaly
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="anomalies">
            {renderAnomalies()}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default EnhancedSentimentAnalysis;
