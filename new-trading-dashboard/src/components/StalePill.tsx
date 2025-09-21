type Props = { stale?: boolean; ageMs?: number; ttlMs?: number; className?: string };
export default function StalePill({ stale, ageMs, ttlMs, className }: Props) {
  if (!stale) return null;
  return (
    <span
      title={`stale (${ageMs ?? '-'} ms > ${ttlMs ?? '-'} ms)`}
      className={className}
      style={{
        marginLeft: 8, fontSize: 11, padding: '1px 6px', borderRadius: 999,
        background: 'rgba(251,191,36,.15)', color: '#b45309', border: '1px solid #facc15'
      }}
    >stale</span>
  );
}


