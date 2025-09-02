import React from 'react';
import type { Staleness } from '@/lib/staleness';

export const StalenessBadge: React.FC<{ state: Staleness } & React.HTMLAttributes<HTMLSpanElement>> = ({ state, ...rest }) => {
  const styles: Record<Staleness, string> = {
    active: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300',
    degraded: 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300',
    stale: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300',
  };
  const label = state === 'active' ? 'Live' : state === 'degraded' ? 'Degraded' : 'Stale';
  return (
    <span className={`px-2 py-0.5 rounded text-[11px] font-semibold ${styles[state]}`} {...rest}>{label}</span>
  );
};

export default StalenessBadge;


