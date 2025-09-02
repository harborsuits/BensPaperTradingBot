import React from 'react';

type StageKey =
  | 'FETCH' | 'PARSE' | 'ANALYZE' | 'SCORE'
  | 'GATE'  | 'ROUTE' | 'EXECUTE' | 'WATCHDOG';

type StageMetric = {
  ok: number; warn: number; err: number;
  p50_ms?: number; p95_ms?: number; ewma_ms?: number;
  ts?: number;
};

type MetricsPayload = {
  stages: Partial<Record<StageKey, StageMetric>>;
  ts?: number;
};

type IngestEvent = {
  ts: number;
  stage: StageKey | string;
  symbol?: string;
  strategy_id?: string;
  step?: string;
  message?: string;
  ms?: number;
  trace_id?: string;
  level?: 'ok' | 'warn' | 'err';
};

const STAGES: { key: StageKey; label: string }[] = [
  { key: 'FETCH',    label: 'Fetch' },
  { key: 'PARSE',    label: 'Parse' },
  { key: 'ANALYZE',  label: 'Analyze' },
  { key: 'SCORE',    label: 'Score' },
  { key: 'GATE',     label: 'Gate' },
  { key: 'ROUTE',    label: 'Route' },
  { key: 'EXECUTE',  label: 'Execute' },
  { key: 'WATCHDOG', label: 'Watchdog' },
];

function ms(x?: number) { return x == null ? '—' : `${Math.round(x)}ms`; }

export default function DataIngestionCard() {
  const [metrics, setMetrics] = React.useState<MetricsPayload>({ stages: {} });
  const [events, setEvents] = React.useState<IngestEvent[]>([]);
  const [since, setSince] = React.useState<number | null>(null);
  const [filter, setFilter] = React.useState('');
  const [paused, setPaused] = React.useState(false);

  async function fetchSummary() {
    try {
      const r = await fetch('/api/data/status');
      if (!r.ok) return;
      const j = await r.json();
      setMetrics(j as MetricsPayload);
    } catch {}
  }

  async function fetchEvents() {
    try {
      const params = new URLSearchParams({ limit: '200' });
      if (since) params.set('since', String(since));
      const r = await fetch(`/api/ingestion/events?${params.toString()}`);
      if (!r.ok) return;
      const j = await r.json() as IngestEvent[];
      if (Array.isArray(j) && j.length) {
        const newest = Math.max(...j.map(e => e.ts));
        setSince(prev => Math.max(prev ?? 0, newest));
        if (!paused) setEvents(prev => [...j.reverse(), ...prev].slice(0, 500));
      }
    } catch {}
  }

  React.useEffect(() => {
    fetchSummary();
    fetchEvents();
    const a = setInterval(fetchSummary, 30000);
    const b = setInterval(fetchEvents, 4000);
    return () => { clearInterval(a); clearInterval(b); };
  }, [since, paused]);

  const filtered = React.useMemo(() => {
    const f = filter.trim().toLowerCase();
    if (!f) return events;
    return events.filter(e =>
      (e.symbol ?? '').toLowerCase().includes(f) ||
      (e.stage ?? '').toLowerCase().includes(f) ||
      (e.strategy_id ?? '').toLowerCase().includes(f) ||
      (e.trace_id ?? '').toLowerCase().includes(f) ||
      (e.message ?? '').toLowerCase().includes(f)
    );
  }, [events, filter]);

  const summaryTs = metrics.ts ?? Object.values(metrics.stages ?? {}).reduce<number>((m, s) => Math.max(m, s?.ts ?? 0), 0);

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        {STAGES.map(({ key, label }) => {
          const s = metrics.stages?.[key];
          const status = s?.err ? 'err' : s?.warn ? 'warn' : 'ok';
          const dot = status === 'err' ? 'bg-rose-500' : status === 'warn' ? 'bg-amber-400' : 'bg-emerald-400';
          return (
            <div key={key} className="min-w-0 border border-neutral-800 rounded-xl p-2">
              <div className="flex items-center gap-2">
                <span className={`inline-block w-2 h-2 rounded-full ${dot}`} />
                <h3 className="font-medium text-sm">{label}</h3>
              </div>
              <div className="mt-1 grid grid-cols-3 gap-1 text-[11px]">
                <div className="text-neutral-400">ok</div>
                <div className="text-neutral-400">warn</div>
                <div className="text-neutral-400">err</div>
                <div className="tabular-nums">{s?.ok ?? 0}</div>
                <div className="tabular-nums">{s?.warn ?? 0}</div>
                <div className="tabular-nums">{s?.err ?? 0}</div>
              </div>
              <div className="mt-1 text-[11px]">
                <div className="text-neutral-400">latency</div>
                <div className="tabular-nums truncate">p50 {ms(s?.p50_ms)} · p95 {ms(s?.p95_ms)} · ewma {ms(s?.ewma_ms)}</div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <input
          placeholder="Filter (symbol, stage, strategy, trace, text)…"
          className="px-2 py-1 text-sm rounded border border-neutral-700 bg-transparent min-w-[220px]"
          value={filter} onChange={e => setFilter(e.target.value)}
        />
        <button onClick={() => setPaused(p => !p)} className="px-2 py-1 text-xs rounded border border-neutral-700 hover:bg-neutral-800">{paused ? 'Resume' : 'Pause'}</button>
        <button onClick={() => setEvents([])} className="px-2 py-1 text-xs rounded border border-neutral-700 hover:bg-neutral-800">Clear</button>
        <div className="ml-auto text-xs text-neutral-400">as of {summaryTs ? new Date(summaryTs).toLocaleTimeString() : '—'}</div>
      </div>

      <div className="table-wrap max-h-[180px] overflow-y-auto">
        <table className="table text-xs">
          <thead>
            <tr>
              <th>Time</th>
              <th>Stage</th>
              <th>Symbol</th>
              <th className="hidden sm:table-cell">Strategy</th>
              <th>Step</th>
              <th className="hidden md:table-cell">Trace</th>
              <th className="hidden lg:table-cell">Latency</th>
              <th className="w-[1%]"></th>
            </tr>
          </thead>
          <tbody>
            {filtered.slice(0, 50).map((e, i) => {
              const t = new Date((e.ts < 1e12 ? e.ts * 1000 : e.ts));
              const level = e.level === 'err' ? 'border-rose-700 bg-rose-900/30' : e.level === 'warn' ? 'border-amber-600 bg-amber-900/30' : 'border-neutral-700 bg-neutral-800/30';
              return (
                <tr key={e.trace_id ?? i}>
                  <td className="tabular-nums">{t.toLocaleTimeString()}</td>
                  <td><span className={`inline-block px-1.5 py-0.5 rounded-full text-[10px] border ${level}`}>{e.stage}</span></td>
                  <td className="font-medium">{e.symbol ?? '—'}</td>
                  <td className="hidden sm:table-cell">{e.strategy_id ?? '—'}</td>
                  <td className="max-w-[20ch] truncate" title={(e.message ?? e.step) || undefined}>{e.step ?? e.message ?? '—'}</td>
                  <td className="hidden md:table-cell">{e.trace_id?.slice(0, 8) ?? '—'}</td>
                  <td className="hidden lg:table-cell tabular-nums">{e.ms != null ? `${Math.round(e.ms)}ms` : '—'}</td>
                  <td>{e.trace_id ? (<button className="px-1.5 py-0.5 text-[10px] rounded border border-neutral-700 hover:bg-neutral-800" onClick={() => setFilter(e.trace_id!)}>Focus</button>) : (<span className="text-[10px] text-neutral-500">—</span>)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="text-right">
        <a href="/logs" className="text-xs text-neutral-400 hover:underline">Open full timeline →</a>
      </div>
    </div>
  );
}

