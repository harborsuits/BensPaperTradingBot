import { useScannerCandidates } from '@/hooks/useScanner';
import { useNavigate } from 'react-router-dom';

export default function ScannerCandidatesCard() {
  const { items, asOf, error } = useScannerCandidates(30000);
  const nav = useNavigate();

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-3">
      <div className="flex items-center justify-between mb-2">
        <div className="font-semibold">Scanner Candidates</div>
        <div className="text-xs opacity-60">
          {error ? 'no scanner data' : asOf ? `as of ${new Date(asOf).toLocaleTimeString()}` : ''}
        </div>
      </div>

      {!items || items.length === 0 ? (
        <div className="text-sm opacity-70">No candidates right now.</div>
      ) : (
        <ul className="divide-y divide-slate-800">
          {items.map((c, i) => (
            <li
              key={`${c.symbol}-${i}`}
              className="py-2 cursor-pointer hover:bg-slate-800/40 rounded-md px-2 -mx-2"
              onClick={() => nav(`/symbol/${encodeURIComponent(c.symbol)}`)}
            >
              <div className="flex items-center justify-between">
                <div className="font-medium">{c.symbol}</div>
                {typeof c.score === 'number' && (
                  <div className="text-xs opacity-70">score {c.score.toFixed(2)}</div>
                )}
              </div>
              {(c.reason || c.notes) && (
                <div className="text-xs opacity-70 mt-1">{c.reason || c.notes}</div>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}


