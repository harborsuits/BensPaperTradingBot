import React from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { ingestionApi } from '@/services/api';

function formatAge(seconds: number): string {
  if (seconds < 60) return `${Math.max(0, Math.floor(seconds))}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}m${s}s`;
}

export default function DataAgeBadge() {
  const qc = useQueryClient();
  const { data } = useQuery({
    queryKey: ['data-status-header'],
    queryFn: () => ingestionApi.getDataStatus(),
    refetchInterval: 5000,
    staleTime: 4000,
  });

  // Prefer explicit freshness seconds if provided; otherwise derive from timestamp
  const freshSecs = (data as any)?.data?.metrics?.dataFreshSeconds;
  const ts = data?.success && (data as any).data?.timestamp ? new Date((data as any).data.timestamp) : null;
  let ageSec = typeof freshSecs === 'number' ? Math.max(0, freshSecs) : ts ? (Date.now() - ts.getTime()) / 1000 : null;
  if (ageSec == null) {
    const ps = qc.getQueryState(['portfolio']);
    if (ps?.dataUpdatedAt) ageSec = (Date.now() - ps.dataUpdatedAt) / 1000;
  }
  // Show em dash when unknown or zero to keep UI calm until data flows
  if (!ageSec || ageSec <= 0) {
    return (
      <span className="text-xs px-2 py-0.5 rounded bg-gray-200 text-gray-700">—</span>
    );
  }
  const color = ageSec < 15 ? 'bg-green-100 text-green-700' : ageSec < 60 ? 'bg-yellow-100 text-yellow-800' : 'bg-red-100 text-red-700';
  const label = `Data: ${formatAge(ageSec)}`;

  return (
    <span className={`text-xs px-2 py-0.5 rounded ${color}`}>{label}</span>
  );
}


