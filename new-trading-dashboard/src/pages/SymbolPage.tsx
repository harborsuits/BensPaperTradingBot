import { useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { useQuotes } from '@/hooks/useQuotes';
import { pricesStream } from '@/lib/streams';

export default function SymbolPage() {
  const { symbol = '' } = useParams();
  const sym = String(symbol || '').toUpperCase();
  const { quotes, asOf } = useQuotes([sym], 5000);
  const q = quotes[sym];
  
  // Explicitly subscribe to this symbol with higher priority
  useEffect(() => {
    if (sym) {
      pricesStream.send({ 
        type: 'subscribe', 
        symbols: [sym], 
        ttlSec: 300 // longer TTL for focused symbol
      });
    }
  }, [sym]);

  return (
    <div className="container py-6 space-y-4">
      <div className="text-2xl font-semibold">{sym}</div>
      <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
        {!q ? (
          <div className="text-sm opacity-70">Loading price…</div>
        ) : (
          <div className="flex items-center gap-6">
            <div className="text-3xl font-bold">{q.price?.toFixed?.(2)}</div>
            {typeof q.changePct === 'number' && (
              <div className={`text-lg ${q.changePct >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                {q.changePct.toFixed(2)}%
              </div>
            )}
          </div>
        )}
        <div className="text-xs opacity-60 mt-2">{asOf ? `as of ${new Date(asOf).toLocaleTimeString()}` : ''}</div>
      </div>
    </div>
  );
}


