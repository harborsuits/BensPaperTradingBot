import { useQuotes } from '@/hooks/useQuotes';
import { useQuery } from '@tanstack/react-query';

export default function PriceTape({ symbols }: { symbols?: string[] }) {
  // Use provided symbols, fallback to active roster, then comprehensive default basket
  const roster = useQuery({
    queryKey: ['roster','active'],
    queryFn: async () => {
      try { const r = await fetch('/api/roster/active'); return await r.json(); } catch { return { items: [] }; }
    },
    refetchInterval: 15_000,
    staleTime: 10_000,
  });
  const active = Array.isArray((roster.data as any)?.items) ? (roster.data as any).items : [];
  const derived = (symbols && symbols.length ? symbols : active.map((x:any)=>x.symbol)).filter(Boolean);
  // Comprehensive default basket covering major asset classes
  const wanted = derived.length ? derived.slice(0, 25) : [
    // Large Caps
    'SPY','AAPL','QQQ','MSFT','NVDA','TSLA','AMD','META','GOOGL','AVGO','COST','CRM',
    // Mid/Small Caps & Growth
    'PLTR','IWM','SMH','AMZN','NFLX','DIS','JNJ','JPM','V','PG','KO','XOM','BAC',
    // Crypto & Commodities
    'BTC','ETH','GLD','SLV'
  ];
  const { quotes, asOf, error } = useQuotes(wanted, 5000);
  const items = Object.values(quotes);

  // Create ticker data with duplicates for seamless scrolling
  const tickerItems = items.length > 0 ? [...items, ...items, ...items] : [];

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-3 w-full overflow-x-hidden">
      <div className="flex items-center justify-between mb-2">
        <div className="font-semibold">
          Live Prices
        </div>
        <div className="text-xs opacity-60 truncate">
          {error ? 'stream issue' : asOf ? `as of ${new Date(asOf).toLocaleTimeString()}` : ''}
        </div>
      </div>
      {tickerItems.length === 0 ? (
        <div className="text-sm opacity-70">Waiting for prices…</div>
      ) : (
        <div className="relative overflow-hidden bg-slate-800/30 rounded-md w-full">
          <div className="flex animate-ticker hover:pause-ticker w-max">
            {tickerItems.map((q:any, index) => (
              <div key={`${q.symbol}-${index}`} className="flex-shrink-0 whitespace-nowrap px-3 py-2 text-sm">
                <span className="font-medium mr-1 text-slate-300">{q.symbol}</span>
                <span className="mr-2 text-white">{Number(q.last ?? q.price ?? q.close ?? 0).toFixed(2)}</span>
                {typeof (q.pct ?? q.changePct) === 'number' && (
                  <span className={`mr-4 ${Number(q.pct ?? q.changePct) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {Number(q.pct ?? q.changePct).toFixed(2)}%
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

    </div>
  );
}


