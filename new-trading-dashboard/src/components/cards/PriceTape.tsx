import { useQuotes } from '@/hooks/useQuotes';
import { useQuery } from '@tanstack/react-query';
import ProvenanceChip from '@/components/ProvenanceChip';
import StalePill from '@/components/StalePill';

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

  // Detect live provider status from backend
  const quotesStatus = useQuery({
    queryKey: ['quotes','status'],
    queryFn: async () => {
      try {
        const r = await fetch('/api/quotes/status');
        return await r.json();
      } catch {
        return {} as any;
      }
    },
    refetchInterval: 30_000,
    staleTime: 20_000,
  });
  const provider = (quotesStatus.data as any)?.provider as string | undefined;
  const source = (quotesStatus.data as any)?.source as string | undefined;
  const asof_ts = (quotesStatus.data as any)?.asof_ts as string | undefined;
  const latency_ms = (quotesStatus.data as any)?.latency_ms as number | undefined;
  const hasLiveProvider = provider && provider !== 'none' && provider !== 'synthetic';

  // Single-pass list to avoid duplicated look in dev
  const tickerItems = items;

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-3 w-full overflow-x-hidden">
      <div className="flex items-center justify-between mb-2">
        <div className="font-semibold">
          {hasLiveProvider ? 'Live Prices' : 'Prices (no live provider)'}
        </div>
        <div className="text-xs opacity-60 truncate" style={{ display:'flex', gap:8, alignItems:'center' }}>
          {error ? 'stream issue' : asOf ? `as of ${new Date(asOf).toLocaleTimeString()}` : ''}
          <ProvenanceChip source={source} provider={provider} asof_ts={asof_ts} latency_ms={latency_ms} warnIfNotBroker={false} />
        </div>
      </div>
      {tickerItems.length === 0 ? (
        <div className="text-sm opacity-70">{hasLiveProvider ? 'Waiting for prices…' : 'No feed available'}</div>
      ) : (
        <div className="relative bg-slate-800/30 rounded-md w-full">
          <div className="flex flex-wrap items-center">
            {tickerItems.map((q:any, index) => {
              const stale = Boolean((q as any).stale);
              const ageMs = (q as any).cache_age_ms;
              const ttlMs = (q as any).ttl_ms;
              return (
              <div key={`${q.symbol}-${index}`} className="whitespace-nowrap px-3 py-2 text-sm flex items-center">
                <span className="font-medium mr-1 text-slate-300">{q.symbol}</span>
                <span className="mr-2 text-white" style={{opacity: stale ? 0.6 : 1}}>{Number(q.last ?? q.price ?? q.close ?? 0).toFixed(2)}</span>
                <StalePill stale={stale} ageMs={ageMs} ttlMs={ttlMs} />
                {typeof (q.pct ?? q.changePct) === 'number' && (
                  <span className={`${Number(q.pct ?? q.changePct) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {(Number(q.pct ?? q.changePct)).toFixed(2)}%
                  </span>
                )}
              </div>
            )})}
          </div>
        </div>
      )}

    </div>
  );
}


