import React from 'react';
import { useNews } from '@/hooks/useNews';
import { Spinner } from '@/components/ui/Spinner';
import { ErrorMessage } from '@/components/ui/ErrorMessage';
import { NewsItem } from './NewsItem';

interface SidebarProps {}

const Sidebar: React.FC<SidebarProps> = () => {
  const { data: positiveNews, isLoading: isLoadingPositive, error: errorPositive } = useNews('positive', 3);
  const { data: neutralNews, isLoading: isLoadingNeutral, error: errorNeutral } = useNews('neutral', 3);
  const { data: negativeNews, isLoading: isLoadingNegative, error: errorNegative } = useNews('negative', 2);

  return (
    <div className="sidebar w-64 h-full border-r border-border bg-card p-4 flex flex-col">
      <div className="sidebar-header mb-4">
        <h2 className="text-xl font-bold">Realtime</h2>
      </div>
      
      {/* Realtime section */}
      <div className="realtime-section mb-6">
        {isLoadingPositive ? (
          <Spinner size="sm" />
        ) : errorPositive ? (
          <ErrorMessage title="Error loading data" message="Could not load realtime data" />
        ) : positiveNews && positiveNews.length > 0 ? (
          positiveNews.map(item => (
            <NewsItem key={item.id} item={item} />
          ))
        ) : (
          <div className="bg-muted rounded-md p-3 text-center text-sm text-muted-foreground">
            No realtime data available
          </div>
        )}
      </div>
      
      <div className="divider h-px bg-border w-full my-4"></div>
      
      {/* News val section */}
      <div className="news-section mb-6">
        <h2 className="text-xl font-bold mb-3">News val</h2>
        {isLoadingNeutral ? (
          <Spinner size="sm" />
        ) : errorNeutral ? (
          <ErrorMessage title="Error loading news" message="Could not load news data" />
        ) : neutralNews && neutralNews.length > 0 ? (
          neutralNews.map(item => (
            <NewsItem key={item.id} item={item} />
          ))
        ) : (
          <div className="bg-muted rounded-md p-3 text-center text-sm text-muted-foreground">
            No news available
          </div>
        )}
      </div>
      
      <div className="divider h-px bg-border w-full my-4"></div>
      
      {/* Negative section */}
      <div className="negative-section">
        <h2 className="text-xl font-bold mb-3">Negative</h2>
        {isLoadingNegative ? (
          <Spinner size="sm" />
        ) : errorNegative ? (
          <ErrorMessage title="Error loading alerts" message="Could not load negative alerts" />
        ) : negativeNews && negativeNews.length > 0 ? (
          negativeNews.map(item => (
            <NewsItem key={item.id} item={item} />
          ))
        ) : (
          <div className="bg-muted rounded-md p-3 text-center text-sm text-muted-foreground">
            No negative alerts
          </div>
        )}
      </div>
    </div>
  );
};

export default Sidebar; 