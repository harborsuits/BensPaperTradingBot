import { useState } from "react";
import { useNewsSentiment } from "@/hooks/useNewsSentiment";
import { RefreshCw, ExternalLink, AlertTriangle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { ZeroState } from "@/components/ui/ZeroState";
import { fmtNum } from "@/utils/number";

// Available news categories
const CATEGORIES = ["markets", "macro", "politics", "tech", "crypto"];

/**
 * Renders a sentiment indicator pill based on a value from -1 to 1
 */
function SentimentPill({ value, title }: { value: number, title: string }) {
  // Map -1..1 to 0..100 for percentage width
  const pct = Math.round((value + 1) * 50);
  
  return (
    <div title={title} className="w-28 h-2 bg-white/10 rounded overflow-hidden">
      <div 
        style={{ width: `${pct}%` }} 
        className={`h-full ${value >= 0 ? 'bg-emerald-400' : 'bg-rose-400'}`} 
      />
    </div>
  );
}

/**
 * News Sentiment Board component that shows clustered news with sentiment analysis
 */
export default function NewsSentimentBoard() {
  const [category, setCategory] = useState<string>("markets");
  const { data, isLoading, isError, refetch } = useNewsSentiment(category, "", 5);

  return (
    <Card className="overflow-hidden">
      <CardHeader className="pb-2">
        <CardTitle className="text-base flex items-center justify-between">
          <span>News Sentiment Analysis</span>
          <button
            onClick={() => refetch()}
            className="p-1 rounded-full bg-primary/10 hover:bg-primary/20 text-primary transition-colors"
            title="Refresh news sentiment"
          >
            <RefreshCw size={14} />
          </button>
        </CardTitle>
      </CardHeader>

      <CardContent className="p-4">
        {/* Category selector */}
        <div className="flex gap-2 mb-4 overflow-x-auto pb-2">
          {CATEGORIES.map(cat => (
            <button
              key={cat}
              className={`px-3 py-1 rounded-full text-sm whitespace-nowrap ${
                cat === category
                  ? 'bg-primary text-white'
                  : 'bg-white/10 hover:bg-white/20'
              }`}
              onClick={() => setCategory(cat)}
            >
              {cat}
            </button>
          ))}
        </div>

        {/* Loading state */}
        {isLoading && (
          <div className="space-y-4">
            {[1, 2, 3].map(i => (
              <div key={i} className="p-3 rounded border border-white/10 bg-white/5">
                <Skeleton className="h-5 w-3/4 mb-2" />
                <Skeleton className="h-4 w-1/2" />
                <div className="flex justify-between mt-2">
                  <Skeleton className="h-3 w-1/4" />
                  <Skeleton className="h-3 w-1/4" />
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Error state */}
        {isError && (
          <ZeroState
            icon={<AlertTriangle className="h-8 w-8 text-amber-500" />}
            title="Unable to load news sentiment"
            message="There was an error fetching news sentiment data. Please try again later."
            action={{
              label: "Retry",
              onClick: () => refetch()
            }}
          />
        )}

        {/* Empty state */}
        {data?.clusters?.length === 0 && !isLoading && !isError && (
          <ZeroState
            title="No news found"
            message={`No news articles found for category "${category}". Try another category or check back later.`}
          />
        )}

        {/* News clusters */}
        {data?.clusters && data.clusters.length > 0 && (
          <div className="space-y-4">
            {data.clusters.slice(0, 6).map((cluster, i) => (
              <div key={i} className="p-3 rounded border border-white/10 bg-white/5">
                <div className="flex items-center justify-between gap-2">
                  <a 
                    className="font-medium hover:underline flex items-center gap-1" 
                    href={cluster.url} 
                    target="_blank" 
                    rel="noreferrer"
                  >
                    {cluster.headline}
                    <ExternalLink size={14} className="text-gray-400" />
                  </a>
                </div>
                
                <div className="flex flex-wrap items-center gap-x-4 gap-y-2 mt-3 text-xs">
                  <div className="flex items-center gap-1">
                    <span className="text-gray-400">Sentiment:</span>
                    <SentimentPill value={cluster.sentiment} title="Sentiment score (-1 to 1)" />
                    <span className="ml-1">{fmtNum(cluster.sentiment, 2)}</span>
                  </div>
                  
                  <div className="flex items-center gap-1">
                    <span className="text-gray-400">Info:</span>
                    <span>{fmtNum(cluster.informational, 2)}</span>
                  </div>
                  
                  <div className="flex items-center gap-1">
                    <span className="text-gray-400">Partisan Δ:</span>
                    <span>{fmtNum(cluster.partisan_spread, 2)}</span>
                  </div>
                  
                  <div className="flex items-center gap-1">
                    <span className="text-gray-400">Finance:</span>
                    <span>{fmtNum(cluster.finance, 2)}</span>
                  </div>
                </div>
                
                <div className="mt-2 text-xs text-gray-400">
                  Sources: {cluster.sources.join(" · ")}
                </div>
              </div>
            ))}
          </div>
        )}
        
        {/* Outlet metrics */}
        {data?.outlets && Object.keys(data.outlets).length > 0 && (
          <div className="mt-4 pt-4 border-t border-white/10">
            <h3 className="text-sm font-medium mb-2">Source Analysis</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-xs">
              {Object.entries(data.outlets).map(([outlet, metrics]) => (
                <div key={outlet} className="p-2 rounded bg-white/5">
                  <div className="font-medium">{outlet}</div>
                  <div className="flex justify-between mt-1">
                    <span className="text-gray-400">Articles: {metrics.count}</span>
                    <span className="text-gray-400">Sent: {fmtNum(metrics.avg_sent, 2)}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
