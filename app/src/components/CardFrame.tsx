import React from 'react';
import { computeStaleness, formatAsOf, type Staleness } from '@/lib/staleness';
import { StalenessBadge } from '@/components/StalenessBadge';

type Props = {
  title: string;
  asOf?: string | number | Date;
  warnAfterMs?: number;
  staleAfterMs?: number;
  right?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
};

export const CardFrame: React.FC<Props> = ({ title, asOf, warnAfterMs, staleAfterMs, right, children, className }) => {
  const state: Staleness = computeStaleness(asOf, Date.now(), warnAfterMs, staleAfterMs);
  return (
    <div className={`rounded-2xl border border-border bg-card p-4 ${className || ''}`}>
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-baseline gap-2">
          <h3 className="text-sm font-semibold opacity-80 truncate pr-2">{title}</h3>
          <span className="text-xs text-muted-foreground">as of {formatAsOf(asOf)}</span>
        </div>
        <div className="flex items-center gap-2">
          <StalenessBadge state={state} />
          {right}
        </div>
      </div>
      {children}
    </div>
  );
};

export default CardFrame;


