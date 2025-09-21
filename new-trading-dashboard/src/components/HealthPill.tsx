import React from 'react';
import { useHealth } from '@/hooks/useHealth';
import { computeStaleness, formatAsOf } from '@/lib/staleness';

export const HealthPill: React.FC = () => {
  const { data, isLoading, isError } = useHealth();
  const asOf = (data as any)?.asOf || (data as any)?.meta?.asOf || (data as any)?.ts;
  const state = asOf ? computeStaleness(asOf) : isError ? 'stale' : 'active';
  const classes =
    state === 'active'
      ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300'
      : state === 'degraded'
      ? 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300'
      : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300';
  const label = isLoading
    ? 'Checking…'
    : isError
    ? 'Unhealthy'
    : `${(data as any)?.env ?? 'env?'}${(data as any)?.gitSha ? ` • ${String((data as any).gitSha).slice(0, 7)}` : ''}`;

  return (
    <div
      className={`px-3 py-1 rounded-full text-xs font-semibold ${classes}`}
      title={asOf ? `as of ${formatAsOf(asOf)}` : undefined}
    >
      {label}
    </div>
  );
};

export default HealthPill;


