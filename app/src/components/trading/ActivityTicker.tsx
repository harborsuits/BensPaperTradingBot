import { useQuery } from '@tanstack/react-query';
import { Terminal } from 'lucide-react';
import { Link } from 'react-router-dom';

const ActivityTicker = () => {
  const { data: events } = useQuery({
    queryKey: ['ingestion', 'events'],
    queryFn: async () => {
      try {
        const res = await fetch('/api/ingestion/events?limit=50');
        if (!res.ok) return [];
        return await res.json();
      } catch (error) {
        console.error('Failed to fetch ingestion events:', error);
        return [];
      }
    },
    refetchInterval: 4000,
  });

  const tickerEvents = events || [];

  return (
    <div className="bg-card border border-border rounded-md p-2 overflow-hidden relative h-10">
      <div className="absolute inset-0 flex items-center">
        <div className="flex animate-marquee whitespace-nowrap">
          {tickerEvents.map((event, i) => (
            <Link to={`/decisions?symbol=${encodeURIComponent(event.symbol)}&trace=${encodeURIComponent(event.trace_id)}`} key={`${event.id}-${i}`} className="flex items-center mx-4 text-sm hover:text-primary">
              <span className="text-muted-foreground">{new Date(event.timestamp).toLocaleTimeString()}</span>
              <span className="mx-2 text-muted-foreground">•</span>
              <span className="font-medium">{event.stage}</span>
              <span className="mx-2 text-muted-foreground">•</span>
              <span>{event.symbol}</span>
              <span className="mx-2 text-muted-foreground">•</span>
              <span className="italic text-muted-foreground">"{event.note}"</span>
              <span className="mx-2 text-muted-foreground">•</span>
              <span>{event.latency_ms}ms</span>
              <span className="mx-2 text-muted-foreground">•</span>
              <span className={event.status === 'ok' ? 'text-green-500' : 'text-red-500'}>{event.status}</span>
            </Link>
          ))}
        </div>
        <div className="flex animate-marquee2 whitespace-nowrap">
                    {tickerEvents.map((event, i) => (
            <Link to={`/decisions?symbol=${encodeURIComponent(event.symbol)}&trace=${encodeURIComponent(event.trace_id)}`} key={`${event.id}-${i}-clone`} className="flex items-center mx-4 text-sm hover:text-primary">
              <span className="text-muted-foreground">{new Date(event.timestamp).toLocaleTimeString()}</span>
              <span className="mx-2 text-muted-foreground">•</span>
              <span className="font-medium">{event.stage}</span>
              <span className="mx-2 text-muted-foreground">•</span>
              <span>{event.symbol}</span>
              <span className="mx-2 text-muted-foreground">•</span>
              <span className="italic text-muted-foreground">"{event.note}"</span>
              <span className="mx-2 text-muted-foreground">•</span>
              <span>{event.latency_ms}ms</span>
              <span className="mx-2 text-muted-foreground">•</span>
              <span className={event.status === 'ok' ? 'text-green-500' : 'text-red-500'}>{event.status}</span>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ActivityTicker;
