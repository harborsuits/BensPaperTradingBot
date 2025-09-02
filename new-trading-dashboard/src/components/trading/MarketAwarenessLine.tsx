import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Badge } from '@/components/ui/badge';
import { format } from 'date-fns';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface MarketAwarenessLineProps {
  className?: string;
}

const MarketAwarenessLine: React.FC<MarketAwarenessLineProps> = ({ className = '' }) => {
  const { data: context } = useQuery({
    queryKey: ['context'],
    queryFn: async () => {
      const res = await fetch('/api/context');
      if (!res.ok) {
        throw new Error(`API error: ${res.status} ${res.statusText}`);
      }
      return res.json();
    },
    refetchInterval: 45000,
    retry: (failureCount, error) => {
      // Don't retry on 4xx errors (client errors)
      if (error.message.includes('4')) {
        return false;
      }
      return failureCount < 3;
    },
  });

  const mood = context?.regime?.type || '...';
  const vol = context?.volatility?.classification || '...';
  const volChange = context?.volatility?.change || 0;
  const newsTone = context?.sentiment?.score > 0.6 ? 'positive' : context?.sentiment?.score < 0.4 ? 'negative' : 'neutral';

  const VolIcon = volChange > 0 ? TrendingUp : volChange < 0 ? TrendingDown : Minus;
  const Separator = () => <div className="h-4 w-[1px] bg-border" />;

  return (
    <div className={`flex items-center text-xs text-muted-foreground space-x-2 ${className}`}>
      <span>Market mood: <span className="font-semibold text-foreground">{mood}</span></span>
      <Separator />
      <span>Volatility: <span className="font-semibold text-foreground">{vol}</span> <VolIcon size={14} className={`inline-block ${volChange > 0 ? 'text-red-500' : 'text-green-500'}`} /></span>
      <Separator />
      <span>News tone: <span className="font-semibold text-foreground">{newsTone}</span></span>
      <Separator />
      <span>As of {context?.timestamp ? format(new Date(context.timestamp), 'HH:mm:ss') : '...'}</span>
    </div>
  );
};

export default MarketAwarenessLine;
