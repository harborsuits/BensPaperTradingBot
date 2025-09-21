type Props = { source?: string; provider?: string; asof_ts?: string; latency_ms?: number; warnIfNotBroker?: boolean };

export default function ProvenanceChip({ source='unknown', provider='unknown', asof_ts, latency_ms, warnIfNotBroker=true }: Props) {
  const isWarn = warnIfNotBroker && source !== 'broker';
  const title = `source=${source} provider=${provider} asof=${asof_ts || '-'} latency=${latency_ms ?? '-'}ms`;
  return (
    <span
      title={title}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        fontSize: 12,
        padding: '2px 6px',
        borderRadius: 999,
        border: `1px solid ${isWarn ? '#b45309' : '#94a3b8'}`,
        color: isWarn ? '#b45309' : '#475569',
        background: isWarn ? 'rgba(251,191,36,0.1)' : 'rgba(148,163,184,0.15)',
      }}
    >
      <span>{provider}</span>
      <span>·</span>
      <span>{asof_ts ? new Date(asof_ts).toLocaleTimeString() : '—'}</span>
      {isWarn && <span style={{ marginLeft: 4 }}>⚠️</span>}
    </span>
  );
}


