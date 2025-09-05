import { useState } from "react";
import { Card } from "@/components/ui/Card";
import { useQuotesQuery } from "@/hooks/useQuotes";
import { useBars } from "@/hooks/useBars";

export default function MarketDataPage(){
  const [symbols, setSymbols] = useState(["SPY","QQQ","AAPL"]);
  const { data: q, isLoading: quotesLoading, isError: quotesError, refetch: refetchQuotes } = useQuotesQuery(symbols);
  const { data: bars, isLoading: barsLoading, isError: barsError, refetch: refetchBars } = useBars(symbols[0], "1Day", 30);

  return (
    <div className="grid gap-3 md:grid-cols-2">
      <Card className="card">
        <div className="card-header">
          <div className="card-title">Quotes</div>
          <div className="card-subtle">
            {quotesLoading ? "Loading..." : quotesError ? "Error" : "Live"}
          </div>
        </div>
        <div className="card-content">
          {quotesError ? (
            <div className="text-center py-4">
              <p className="text-red-500 text-sm mb-2">Unable to load quotes</p>
              <button className="btn text-xs" onClick={() => refetchQuotes()}>Retry</button>
            </div>
          ) : quotesLoading ? (
            <div className="text-center py-4">
              <div className="animate-pulse">Loading quotes...</div>
            </div>
          ) : !q || q.length === 0 ? (
            <div className="text-center py-4 text-gray-500">
              No quotes available
            </div>
          ) : (
            <table className="w-full text-sm">
              <thead><tr><th>Sym</th><th className="text-right">Last</th><th className="text-right">Δ%</th></tr></thead>
              <tbody>
                {q.map((r:any)=>(
                  <tr key={r.symbol} className="cursor-pointer hover:bg-gray-50" onClick={()=>setSymbols([r.symbol, ...symbols.filter(s=>s!==r.symbol)])}>
                    <td>{r.symbol}</td>
                    <td className="text-right num">${Number(r.last).toFixed(2)}</td>
                    <td className={`text-right num ${r.pct >= 0 ? "text-green-600":"text-red-600"}`}>
                      {r.pct >= 0 ? '+' : ''}{Number(r.pct).toFixed(2)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </Card>

      <Card className="card">
        <div className="card-header">
          <div className="card-title">{symbols[0]} · 30 bars</div>
          <div className="card-subtle">
            {barsLoading ? "Loading..." : barsError ? "Error" : "Historical"}
          </div>
        </div>
        <div className="card-content">
          {barsError ? (
            <div className="text-center py-4">
              <p className="text-red-500 text-sm mb-2">Unable to load bars</p>
              <button className="btn text-xs" onClick={() => refetchBars()}>Retry</button>
            </div>
          ) : barsLoading ? (
            <div className="text-center py-4">
              <div className="animate-pulse">Loading bars...</div>
            </div>
          ) : !bars?.bars || bars.bars.length === 0 ? (
            <div className="text-center py-4 text-gray-500">
              No bar data available
            </div>
          ) : (
            <div>
              <div className="text-xs text-gray-600 mb-2">
                Latest {bars.bars.length} bars (showing last 5):
              </div>
              <pre className="text-xs bg-gray-50 p-2 rounded overflow-auto">
                {JSON.stringify(bars.bars.slice(-5), null, 2)}
              </pre>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}
