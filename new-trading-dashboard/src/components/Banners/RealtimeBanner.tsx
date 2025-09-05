import { useEffect } from 'react';
import { useLiveStatus } from '@/stores/liveStatus';

export default function RealtimeBanner({ demoQuotes }: { demoQuotes?: boolean }) {
  const { prices, decisions, setTick } = useLiveStatus();
  useEffect(() => {
    const t = setInterval(setTick, 1000);
    return () => clearInterval(t);
  }, [setTick]);

  const noPrices = (prices.state !== 'open') || (prices.staleMs > 20_000);
  const noDecisions = (decisions.state !== 'open') || (decisions.staleMs > 20_000);

  if (!noPrices && !noDecisions && !demoQuotes) return null;

  return (
    <div className="rounded-md bg-amber-500/10 border border-amber-500/30 text-amber-600 px-3 py-2 text-sm">
      {!demoQuotes && (noPrices || noDecisions) ? (
        <>Not receiving real-time updates{noPrices ? ' (prices)' : ''}{noPrices && noDecisions ? ' & ' : ''}{noDecisions ? ' (decisions)' : ''}. Reconnecting…</>
      ) : demoQuotes ? (
        <>Quotes running in demo mode. Add <code>TRADIER_TOKEN</code> (or set <code>QUOTES_PROVIDER</code>) for live prices.</>
      ) : null}
    </div>
  );
}
