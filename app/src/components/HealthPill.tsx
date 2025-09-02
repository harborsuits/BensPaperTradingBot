import React from 'react';
import { useHealth } from '@/hooks/useHealth';
import { computeStaleness, formatAsOf } from '@/lib/staleness';

export const HealthPill: React.FC = () => {
  const { data, isLoading, isError } = useHealth();
  const state = data?.asOf ? computeStaleness(data.asOf) : isError ? 'stale' : 'active';
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
    : `${data?.env ?? 'env?'}${data?.gitSha ? ` • ${String(data.gitSha).slice(0, 7)}` : ''}`;

  return (
    <div
      className={`px-3 py-1 rounded-full text-xs font-semibold ${classes}`}
      title={data?.asOf ? `as of ${formatAsOf(data.asOf)}` : undefined}
    >
      {label}
    </div>
  );
};

export default HealthPill;


