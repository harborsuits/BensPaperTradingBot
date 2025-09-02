import React from 'react';

export default function HealthBadge({ state, asOf }: { state: 'GREEN'|'AMBER'|'RED', asOf?: string }) {
  const tone = state === 'GREEN' ? 'text-emerald-400 border-emerald-700 bg-emerald-900/30'
             : state === 'AMBER' ? 'text-amber-400 border-amber-700 bg-amber-900/30'
             : 'text-rose-400 border-rose-700 bg-rose-900/30';
  return (
    <div className={`inline-flex items-center gap-2 px-2 py-1 rounded border ${tone}`}>
      <span className="h-2 w-2 rounded-full bg-current" />
      <span className="text-xs">Health: {state}</span>
      {asOf && <span className="text-[11px] opacity-70">as of {new Date(asOf).toLocaleTimeString()}</span>}
    </div>
  );
}
