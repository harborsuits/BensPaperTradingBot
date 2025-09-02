import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Lock, RefreshCw } from 'lucide-react';
import { useHealth } from '@/hooks/useHealth';
import { fmtShortHand } from '@/utils/formatters';
import { useCandidates } from '@/queries/useCandidates';
import EmptyState from '@/components/ui/EmptyState';
import { RowSkeleton } from '@/components/ui/CardSkeleton';

const CandidatesQuickList: React.FC = () => {
  const navigate = useNavigate();
  const { data: health } = useHealth();
  const { data = [], isLoading, error, refetch } = useCandidates();

  const isHealthy = health?.breaker === 'GREEN';

  const handleCandidateClick = (symbol: string) => {
    navigate(`/decisions?symbol=${encodeURIComponent(symbol)}`);
  };

  if (isLoading) {
    return (
      <div className="space-y-1 mb-4">
        <div className="text-xs text-muted-foreground mb-2">Scanning candidates...</div>
        <RowSkeleton rows={3} />
      </div>
    );
  }

  if (error) {
    return (
      <EmptyState
        title="Couldn't load candidates"
        description={error.message}
        icon="error"
        className="mb-4"
        action={
          <Button size="sm" variant="outline" onClick={() => refetch()}>
            <RefreshCw className="mr-2 h-4 w-4" /> Retry
          </Button>
        }
      />
    );
  }

  if (data.length === 0) {
    return (
      <EmptyState
        title="No candidates"
        description="We'll show them here when signals fire."
        icon="empty"
        showHealth={true}
        className="mb-4"
      />
    );
  }

  const top = data.slice(0, 6);
  const more = Math.max(0, data.length - top.length);

  return (
    <div className="space-y-1 mb-4">
      <div className="text-xs text-muted-foreground mb-2">
        Quick scan {!isHealthy && '(read-only mode)'} • {data.length} candidates
      </div>
      <ul aria-label="Candidates quick list" className="space-y-1">
        {top.map((candidate) => {
          const shorthand = fmtShortHand(candidate);
          return (
            <li key={candidate.symbol}>
              <Button
                variant="ghost"
                size="sm"
                className="w-full justify-start h-auto p-2 text-xs font-mono hover:bg-muted/50"
                onClick={() => handleCandidateClick(candidate.symbol)}
                disabled={!isHealthy}
                aria-label={`Open Decisions for ${candidate.symbol}`}
              >
                <div className="flex items-center gap-2 w-full min-w-0">
                  <span className="truncate flex-1 text-left">
                    {shorthand}
                  </span>
                  {!isHealthy && <Lock size={12} className="text-muted-foreground flex-shrink-0" />}
                </div>
              </Button>
            </li>
          );
        })}
        {more > 0 && (
          <li className="text-xs text-muted-foreground text-center py-1">
            +{more} more candidates
          </li>
        )}
      </ul>
    </div>
  );
};

export default CandidatesQuickList;
