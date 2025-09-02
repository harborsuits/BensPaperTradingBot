import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { ArrowRight } from 'lucide-react';

interface LoopStripProps {
  className?: string;
}

const stages = [
  { id: 'ingest', label: 'Ingest' },
  { id: 'context', label: 'Context' },
  { id: 'candidates', label: 'Candidates' },
  { id: 'gates', label: 'Gates' },
  { id: 'plan', label: 'Plan' },
  { id: 'route', label: 'Route' },
  { id: 'manage', label: 'Manage' },
  { id: 'learn', label: 'Learn' }
];

const activityNotes = [
  "Pulled fresh prices; market looks risk-off; scanning SPY + tech names…",
  "Checking news sentiment; tariff headlines affecting tech sector…",
  "Evaluating SPY for potential entry; volatility elevated…",
  "Verifying position sizing; ensuring cash is sufficient…",
];

const LoopStripBanner: React.FC<LoopStripProps> = ({ className = '' }) => {
  const [activeStage, setActiveStage] = useState(0);
  const [note, setNote] = useState(activityNotes[0]);

  const { data: events } = useQuery({
    queryKey: ['ingestion', 'events'],
    queryFn: async () => {
      try {
        const res = await fetch('/api/ingestion/events?limit=10');
        if (!res.ok) return [];
        return await res.json();
      } catch (error) {
        console.error('Failed to fetch ingestion events:', error);
        return [];
      }
    },
    refetchInterval: 5000,
  });

  useEffect(() => {
    if (events && events.length > 0) {
      const latestEvent = events[0];
      const stageIndex = stages.findIndex(s =>
        latestEvent.stage?.toLowerCase().includes(s.id.toLowerCase())
      );
      if (stageIndex >= 0) {
        setActiveStage(stageIndex);
        setNote(activityNotes[stageIndex % activityNotes.length]);
      }
    }
  }, [events]);

  return (
    <div className={`bg-card border border-border rounded-md p-3 ${className}`}>
      <div className="flex flex-col space-y-2">
        <div className="flex items-center space-x-1 overflow-x-auto pb-2 scrollbar-thin">
          {stages.map((stage, index) => (
            <React.Fragment key={stage.id}>
              <div
                className={`px-2 py-1 rounded text-sm whitespace-nowrap transition-colors ${
                  index === activeStage
                    ? 'bg-primary/20 text-primary font-medium'
                    : 'text-muted-foreground'
                }`}
              >
                {stage.label}
              </div>
              {index < stages.length - 1 && (
                <ArrowRight size={14} className="text-muted-foreground flex-shrink-0" />
              )}
            </React.Fragment>
          ))}
        </div>
        <div className="text-sm text-muted-foreground italic">
          <span className="font-medium text-foreground">Note:</span> {note}
        </div>
      </div>
    </div>
  );
};

export default LoopStripBanner;
