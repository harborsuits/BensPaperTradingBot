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
    <div className="grid gap-4 md:grid-cols-2">
      <Card className="card">
        <div className="card-header">
          <div className="card-title">Paper Account</div>
          <div className="card-subtle">{acct.data?.asOf ?? "—"}</div>
        </div>
        <div className="card-content space-y-1">
          <div>Total Equity: <b className="num">${(acct.data?.equity ?? 0).toLocaleString()}</b></div>
          <div>Cash: <span className="num">${(acct.data?.cash ?? 0).toLocaleString()}</span></div>
          <div>Buying Power: <span className="num">${(acct.data?.buyingPower ?? 0).toLocaleString()}</span></div>
          <div>Day P/L: <span className="num">${(acct.data?.dayPL ?? 0).toLocaleString()}</span></div>
          <MetaLine meta={getMeta(acct.data as any)} />
        </div>
      </Card>

      <Card className="card">
        <div className="card-header">
          <div className="card-title">Positions</div>
          <div className="card-subtle">{(pos.data ?? []).length} open</div>
        </div>
        <div className="card-content">
          <table className="w-full text-sm">
            <thead><tr><th>Symbol</th><th className="text-right">Qty</th><th className="text-right">Avg</th><th className="text-right">uP/L</th></tr></thead>
            <tbody>
              {(pos.data ?? []).map((p:any) => {
                // Calculate unrealized P/L from live quotes if not provided
                const quote = Array.isArray(quotes) ? quotes.find(q => q.symbol === p.symbol) : null;
                const currentPrice = quote?.last ?? p.currentPrice ?? p.markPrice ?? 0;
                const avgPrice = Number(p.avgPrice ?? 0);
                const qty = Number(p.qty ?? 0);
                const unrealizedPL = p.unrealizedPL ?? ((currentPrice - avgPrice) * qty);
                
                return (
                  <tr key={p.symbol}>
                    <td>{p.symbol}</td>
                    <td className="text-right num">{qty}</td>
                    <td className="text-right num">${avgPrice.toFixed(2)}</td>
                    <td className={`text-right num ${unrealizedPL >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      ${Number(unrealizedPL).toFixed(2)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <Card className="card md:col-span-2">
        <div className="card-header"><div className="card-title">Equity Curve</div></div>
        <div className="card-content">
          {/* plug your sparkline here; placeholder shows latest points for now */}
          <pre className="text-xs bg-gray-50 p-2 rounded">{JSON.stringify(hist.data?.points?.slice(-10), null, 2)}</pre>
        </div>
      </Card>
    </div>
  );
}

export default function PortfolioPage(){
  return (
    <Tabs defaultValue="paper">
      <TabsList>
        <TabsTrigger value="paper">Paper</TabsTrigger>
        <TabsTrigger value="live" disabled>Live</TabsTrigger>
      </TabsList>
      <TabsContent value="paper"><PaperView/></TabsContent>
      <TabsContent value="live"><div className="text-sm text-gray-500 p-4">Enable once broker keys are set.</div></TabsContent>
    </Tabs>
  );
}