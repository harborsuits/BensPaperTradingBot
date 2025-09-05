import { useQuotes } from '@/hooks/useQuotes';

export default function PriceTape({ symbols }: { symbols?: string[] }) {
  const { quotes, asOf, error } = useQuotes(symbols, 5000);
  const items = Object.values(quotes);

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-3">
      <div className="flex items-center justify-between mb-2">
        <div className="font-semibold">Live Prices</div>
        <div className="text-xs opacity-60">
          {error ? 'stream issue' : asOf ? `as of ${new Date(asOf).toLocaleTimeString()}` : ''}
        </div>
      </div>
      {items.length === 0 ? (
        <div className="text-sm opacity-70">Waiting for prices…</div>
      ) : (
        <div className="flex gap-6 overflow-x-auto no-scrollbar text-sm">
          {items.map((q) => (
            <div key={q.symbol} className="whitespace-nowrap">
              <span className="font-medium mr-2">{q.symbol}</span>
              <span>{q.price?.toFixed?.(2)}</span>
              {typeof q.changePct === 'number' && (
                <span className={`ml-2 ${q.changePct >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                  {q.changePct.toFixed(2)}%
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}


