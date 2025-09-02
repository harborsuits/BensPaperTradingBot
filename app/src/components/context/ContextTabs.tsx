import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/Tabs";
import { Card } from "@/components/ui/Card";
import { useMarketContext } from "@/hooks/useMarketContext";
import MetaLine from "@/components/ui/MetaLine";
import { getMeta } from "@/lib/meta";

export default function ContextTabs() {
  const { regime, volatility, sentiment, news, newsSentiment } = useMarketContext();

  return (
    <Tabs defaultValue="regime" className="w-full">
      <TabsList className="grid grid-cols-4">
        <TabsTrigger value="regime">Regime</TabsTrigger>
        <TabsTrigger value="vol">Volatility</TabsTrigger>
        <TabsTrigger value="sent">Sentiment</TabsTrigger>
        <TabsTrigger value="news">News</TabsTrigger>
      </TabsList>

                    <TabsContent value="regime">
                <Card className="card">
                  <div className="card-header">
                    <div className="card-title">Market Regime</div>
                    <div className="card-subtle">{regime?.asOf ?? "—"}</div>
                    <MetaLine meta={getMeta(regime as any)} />
                  </div>
                  <div className="card-content">
                    <div className="text-xl font-semibold">
                      {regime?.type ?? regime?.label ?? regime?.regime ?? "Neutral"}
                    </div>
                    <div className="text-sm text-gray-500">
                      Confidence: {Math.round((regime?.confidence ?? 0.5)*100)}%
                    </div>
                    {regime?.description && (
                      <div className="text-sm text-gray-600 mt-2">
                        {regime.description}
                      </div>
                    )}
                  </div>
                </Card>
              </TabsContent>

      <TabsContent value="vol">
        <Card className="card">
          <div className="card-header">
            <div className="card-title">Volatility</div>
            <div className="card-subtle">{volatility?.asOf ?? "—"}</div>
            <MetaLine meta={getMeta(volatility as any)} />
          </div>
          <div className="card-content">
            <div className="text-xl font-semibold">{volatility?.value ?? volatility?.vix ?? "—"}</div>
            <div className="text-sm text-gray-500">Δ {(volatility?.change ?? volatility?.delta ?? 0)}%</div>
          </div>
        </Card>
      </TabsContent>

      <TabsContent value="sent">
        <Card className="card">
          <div className="card-header"><div className="card-title">Sentiment</div></div>
          <div className="card-content">
            <div className="text-2xl font-semibold">
              {Math.round((sentiment?.confidence ?? 0.5)*100)}%
              <span className="ml-2 text-sm text-gray-500">({sentiment?.label ?? "Neutral"})</span>
            </div>
            <div className="mt-3 flex gap-2 flex-wrap">
              {Array.isArray(newsSentiment) ? newsSentiment.slice(0,6).map((s:any) => (
                <span key={s.source} className="chip">{s.source}: {s.score > 0 ? "+" : ""}{s.score.toFixed(2)}</span>
              )) : null}
            </div>
          </div>
        </Card>
      </TabsContent>

      <TabsContent value="news">
        <Card className="card">
          <div className="card-header"><div className="card-title">Market News</div></div>
          <div className="card-content space-y-2">
            {Array.isArray(news) ? news.slice(0,10).map((n:any) => (
              <div key={n.id || n.headline} className="border-b pb-2">
                <div className="font-medium">{n.headline}</div>
                <div className="text-xs text-gray-500">{n.source} · {n.timeAgo ?? n.publishedAt}</div>
              </div>
            )) : <div className="text-sm text-gray-500">Loading news...</div>}
          </div>
        </Card>
      </TabsContent>
    </Tabs>
  );
}
