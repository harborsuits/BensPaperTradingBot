import React from 'react';
import { useDecisionsRecent } from '@/hooks/useDecisionsRecent';

const TradeDecisionsPage: React.FC = () => {
  const { data: decisions } = useDecisionsRecent(100);

  return (
    <div className="container py-6">
      <div className="flex items-baseline justify-between">
        <h1 className="text-2xl font-bold">Trade Decisions</h1>
        <div className="text-sm text-muted-foreground">live via /ws/decisions</div>
      </div>

      {(!decisions || decisions.length === 0) ? (
        <div className="mt-6 text-sm text-muted-foreground">No decisions yet.</div>
      ) : (
        <div className="mt-4 space-y-3">
          {decisions.map((d: any) => (
            <div key={String(d.trace_id || d.id)} className="border rounded-xl p-4">
              <div className="flex items-center justify-between">
                <div className="font-semibold">{d.symbol} • {String(d.action || d.side).toUpperCase()}</div>
                <div className="text-xs text-muted-foreground">{d.createdAt || d.decidedAt || d.timestamp}</div>
              </div>
              <div className="text-sm mt-1">{d.one_liner || d.reason || '—'}</div>
              <div className="flex flex-wrap gap-2 mt-2">
                {(Array.isArray(d.reasons) ? d.reasons : []).slice(0,4).map((r: string) => (
                  <span key={r} className="px-2 py-0.5 rounded text-xs bg-zinc-200 text-zinc-800">{r}</span>
                ))}
                {(Array.isArray(d.sources) ? d.sources : []).slice(0,4).map((s: string) => (
                  <span key={s} className="px-2 py-0.5 rounded text-xs bg-sky-200 text-sky-800">#{s}</span>
                ))}
                {d?.gates?.passed === false && (
                  <span className="px-2 py-0.5 rounded text-xs bg-red-600 text-white">blocked</span>
                )}
                {d?.gates?.passed === true && (
                  <span className="px-2 py-0.5 rounded text-xs bg-green-600 text-white">approved</span>
                )}
              </div>
              {d?.plan && (
                <div className="mt-2 text-xs text-muted-foreground">
                  plan: {String(d.plan.orderType || d.plan.type || '—')} • qty {Number(d.plan.qty || d.size || 0)} {d.plan.limit ? `• limit ${d.plan.limit}` : ''} {d.plan.stop ? `• stop ${d.plan.stop}` : ''}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default TradeDecisionsPage;