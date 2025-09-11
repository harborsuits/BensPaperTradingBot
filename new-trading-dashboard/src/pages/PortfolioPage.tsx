import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/Tabs";
import { Card } from "@/components/ui/Card";
import { usePaperAccount } from "@/hooks/usePaperAccount";
import { usePaperPositions } from "@/hooks/usePaperPositions";
import MetaLine from "@/components/ui/MetaLine";
import { getMeta } from "@/lib/meta";
import { usePortfolioHistory } from "@/hooks/usePortfolioHistory";
import { useQuotesQuery } from "@/hooks/useQuotes";

function PaperView() {
  const acct = usePaperAccount();
  const pos = usePaperPositions();
  const hist = usePortfolioHistory("paper", 90);
  
  // Get quotes for all positions to calculate live P/L
  const symbols = (pos.data ?? []).map((p:any) => p.symbol).filter(Boolean);
  const { data: quotes } = useQuotesQuery(symbols);

  return (
    <div className="dashboard-container">
      {/* Paper Account Summary */}
      <div className="dashboard-section">
        <Card className="card">
          <div className="card-header">
            <div className="card-title">Paper Account</div>
            <div className="card-subtle">{acct.data?.asOf ?? "—"}</div>
          </div>
          <div className="card-content">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-card/50 rounded-lg border border-border/50">
                <p className="text-sm text-muted-foreground mb-1">Total Equity</p>
                <p className="text-2xl font-bold num">${(acct.data?.equity ?? 0).toLocaleString()}</p>
              </div>
              <div className="text-center p-4 bg-card/50 rounded-lg border border-border/50">
                <p className="text-sm text-muted-foreground mb-1">Cash</p>
                <p className="text-2xl font-semibold num">${(acct.data?.cash ?? 0).toLocaleString()}</p>
              </div>
              <div className="text-center p-4 bg-card/50 rounded-lg border border-border/50">
                <p className="text-sm text-muted-foreground mb-1">Buying Power</p>
                <p className="text-2xl font-semibold num">${(acct.data?.buyingPower ?? 0).toLocaleString()}</p>
              </div>
              <div className="text-center p-4 bg-card/50 rounded-lg border border-border/50">
                <p className="text-sm text-muted-foreground mb-1">Day P/L</p>
                <p className={`text-2xl font-semibold num ${(acct.data?.dayPL ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  ${(acct.data?.dayPL ?? 0).toLocaleString()}
                </p>
              </div>
            </div>
            <div className="mt-4">
              <MetaLine meta={getMeta(acct.data as any)} />
            </div>
          </div>
        </Card>
      </div>

      {/* Positions Table */}
      <div className="dashboard-section">
        <Card className="card">
          <div className="card-header">
            <div className="card-title">Positions</div>
            <div className="card-subtle">{(pos.data ?? []).length} open positions</div>
          </div>
          <div className="card-content">
            {(pos.data ?? []).length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <p className="text-lg font-medium">No open positions</p>
                <p className="text-sm mt-2">Positions will appear here when you have active trades.</p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="border-b border-border">
                    <tr>
                      <th className="text-left py-3 px-2 font-medium">Symbol</th>
                      <th className="text-right py-3 px-2 font-medium">Quantity</th>
                      <th className="text-right py-3 px-2 font-medium">Avg Price</th>
                      <th className="text-right py-3 px-2 font-medium">Unrealized P/L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(pos.data ?? []).map((p:any) => {
                      // Calculate unrealized P/L from live quotes if not provided
                      const quote = Array.isArray(quotes) ? quotes.find(q => q.symbol === p.symbol) : null;
                      const currentPrice = quote?.last ?? p.currentPrice ?? p.markPrice ?? 0;
                      const avgPrice = Number(p.avgPrice ?? 0);
                      const qty = Number(p.qty ?? 0);
                      const unrealizedPL = p.unrealizedPL ?? ((currentPrice - avgPrice) * qty);

                      return (
                        <tr key={p.symbol} className="border-b border-border/50 hover:bg-muted/30 transition-colors">
                          <td className="py-3 px-2 font-medium">{p.symbol}</td>
                          <td className="text-right py-3 px-2 num">{qty}</td>
                          <td className="text-right py-3 px-2 num">${avgPrice.toFixed(2)}</td>
                          <td className={`text-right py-3 px-2 num font-medium ${unrealizedPL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                            ${Number(unrealizedPL).toFixed(2)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </Card>
      </div>

      {/* Equity Curve */}
      <div className="dashboard-section">
        <Card className="card">
          <div className="card-header">
            <div className="card-title">Equity Curve</div>
            <div className="card-subtle">Performance over time</div>
          </div>
          <div className="card-content">
            <div className="min-h-[200px] flex items-center justify-center bg-card/30 rounded-lg border border-dashed border-border/50">
              <div className="text-center text-muted-foreground">
                <p className="text-lg font-medium mb-2">Chart Coming Soon</p>
                <p className="text-sm">Equity curve visualization will be displayed here.</p>
                {/* Temporary data display for debugging */}
                {hist.data?.points?.slice(-5) && (
                  <div className="mt-4 p-3 bg-muted/50 rounded text-xs font-mono max-w-md mx-auto">
                    <p className="text-muted-foreground mb-2">Recent Data Points:</p>
                    {hist.data.points.slice(-5).map((point: any, i: number) => (
                      <div key={i} className="flex justify-between">
                        <span>{point.date || point.timestamp || `Point ${i + 1}`}</span>
                        <span className="num">${Number(point.value || point.equity || 0).toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}

export default function PortfolioPage(){
  return (
    <div className="w-full min-h-screen bg-background">
      <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Header Section */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <h1 className="text-3xl font-bold text-foreground">Portfolio</h1>
          </div>
        </div>

        {/* Tabs Section */}
        <div className="mb-8">
          <Tabs defaultValue="paper">
            <TabsList className="grid w-full max-w-md grid-cols-2">
              <TabsTrigger value="paper">Paper Trading</TabsTrigger>
              <TabsTrigger value="live" disabled>Live Trading</TabsTrigger>
            </TabsList>
            <TabsContent value="paper" className="mt-6">
              <PaperView/>
            </TabsContent>
            <TabsContent value="live" className="mt-6">
              <div className="dashboard-section">
                <Card className="card">
                  <div className="card-content">
                    <div className="text-center py-12 text-muted-foreground">
                      <p className="text-lg font-medium mb-2">Live Trading Coming Soon</p>
                      <p className="text-sm">Configure broker credentials to enable live trading.</p>
                    </div>
                  </div>
                </Card>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}