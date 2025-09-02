import { useState } from "react";
import { useScannerCandidates } from "@/hooks/useScanner";
import { usePlacePaperOrder } from "@/hooks/usePlacePaperOrder";
import { Card, CardHeader, CardContent } from "@/components/ui/Card";

const money = (n:number)=>new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:2}).format(n);
const pct = (x:number)=>`${(x*100).toFixed(2)}%`;

export default function ScannerPage() {
  const [list, setList] = useState('small_caps_liquid');
  const { data = [], isLoading } = useScannerCandidates(list, 30);
  const { mutate: place, isLoading: placing } = usePlacePaperOrder();

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <h1 className="text-xl font-semibold">Catalyst Scanner</h1>
        <select className="border rounded px-2 py-1" value={list} onChange={e=>setList(e.target.value)}>
          <option value="small_caps_liquid">Small Caps (Liquid)</option>
          <option value="default">Default</option>
          <option value="etfs_top">ETFs</option>
          <option value="news_movers_today">News Movers</option>
        </select>
        <span className="text-sm text-gray-500">{isLoading ? 'Loading…' : `${data.length} results`}</span>
      </div>

      <Card>
        <CardHeader title="Ranked Candidates" />
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left border-b">
                  <th>Sym</th><th className="text-right">Last</th><th className="text-right">Score</th>
                  <th className="text-right">Conf</th><th>Plan</th><th>Explain</th><th className="text-right">Action</th>
                </tr>
              </thead>
              <tbody>
                {data.map(row => (
                  <tr key={row.symbol} className="border-b hover:bg-muted/40">
                    <td className="font-medium">{row.symbol}</td>
                    <td className="text-right">{money(row.last)}</td>
                    <td className={`text-right ${row.score>=0?'text-green-600':'text-red-600'}`}>{row.score.toFixed(3)}</td>
                    <td className="text-right">{Math.round(row.confidence*100)}%</td>
                    <td>
                      <div className="text-xs">
                        {row.side.toUpperCase()} @{money(row.plan.entry)} · stop {money(row.plan.stop)} · take {money(row.plan.take)}
                      </div>
                    </td>
                    <td>
                      <div className="text-xs text-gray-600">
                        RVOL {row.explain.rvol.toFixed(2)} · gap {pct(row.explain.gapPct)} · spread {row.explain.spreadPct?.toFixed?.(2) ?? row.explain.spreadPct}%
                        {row.explain.outlets?.length ? <> · {row.explain.outlets.join(', ')}</> : null}
                      </div>
                    </td>
                    <td className="text-right">
                      <button
                        className="px-3 py-1 rounded bg-black text-white disabled:opacity-50"
                        disabled={placing || !row.risk.spreadOK}
                        onClick={()=>place({ symbol: row.symbol, side: row.side==='buy'?'buy':'sell', qty: row.risk.suggestedQty, type: 'market' })}
                      >
                        {placing ? 'Ordering…' : `Paper ${row.side==='buy'?'Buy':'Sell'}`}
                      </button>
                      {!row.risk.spreadOK && <div className="text-[10px] text-amber-600">Spread too wide</div>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
