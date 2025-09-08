import { useScannerCandidatesQuery } from '@/hooks/useScanner';
import { useNavigate } from 'react-router-dom';
import { useQuotesQuery } from '@/hooks/useQuotes';

export default function ScannerCandidatesCard() {
  const nav = useNavigate();

  // Scan across all available symbols - no universe filtering
  const { data: items, isLoading, error } = useScannerCandidatesQuery('all', 50);

  // Get unified quotes for consistent pricing
  const symbols = items?.map(item => item.symbol) || [];
  const { data: quotesData } = useQuotesQuery(symbols);

  if (isLoading) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-3">
        <div className="flex items-center justify-between mb-2">
          <div className="font-semibold">Scanner Candidates</div>
          <div className="text-xs opacity-60">loading...</div>
        </div>
        <div className="text-sm opacity-70">Scanning for opportunities...</div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-3 w-full max-w-full overflow-x-hidden">
      <div className="flex items-center justify-between mb-2">
        <div className="font-semibold">Scanner Candidates</div>
        <div className="text-xs opacity-60">
          {error ? 'no scanner data' : items?.length ? `${items.length} found` : 'scanning...'}
        </div>
      </div>

      {!items || items.length === 0 ? (
        <div className="text-sm opacity-70">
          {error ? `Error: ${error.message}` : 'No candidates found across all symbols.'}
        </div>
      ) : (
        <ul className="divide-y divide-slate-800 max-h-[300px] overflow-y-auto">
          {items.slice(0, 20).map((c, i) => {
            // Get unified price from quotes data for consistency
            const quote = Array.isArray(quotesData) ? quotesData.find(q => q.symbol === c.symbol) : null;
            const unifiedPrice = quote?.last || quote?.price || c.last || 0;

            return (
              <li
                key={`${c.symbol}-${i}`}
                className="py-2 cursor-pointer hover:bg-slate-800/40 rounded-md px-2 -mx-2"
                onClick={() => nav(`/symbol/${encodeURIComponent(c.symbol)}`)}
              >
                <div className="flex items-center justify-between">
                  <div className="font-medium flex items-center gap-2">
                    {c.symbol}
                    <span className={`text-xs px-2 py-0.5 rounded-full ${
                      c.side === 'buy' ? 'bg-green-800/60 text-green-300' : 'bg-red-800/60 text-red-300'
                    }`}>
                      {c.side?.toUpperCase()}
                    </span>
                  </div>
                  <div className="text-xs opacity-70">
                    {c.confidence ? `${(c.confidence * 100).toFixed(0)}%` : c.score?.toFixed(2) || ''}
                  </div>
                </div>
                <div className="text-xs opacity-70 mt-1 flex items-center justify-between">
                  <span>{c.explain?.impact24h ? `${c.explain.impact24h.toFixed(1)}% 24h` : ''}</span>
                  <span>${unifiedPrice ? unifiedPrice.toFixed(2) : '—'}</span>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}


