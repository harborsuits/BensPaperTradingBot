import React from 'react';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { SimpleCard } from '@/components/ui/SimpleCard';

export default function DecisionJournalCard(){
  const { data, isLoading, isError } = useQuery({
    queryKey: ['decision-latest'],
    queryFn: async () => (await axios.get('/api/decisions/latest')).data,
    refetchInterval: 15_000,
    staleTime: 10_000,
    retry: 1,
  });

  if (isLoading) return <SimpleCard title="Latest Agent Decision"><div className="p-4 text-sm text-muted-foreground">Loading…</div></SimpleCard>;
  if (isError || !data) return <SimpleCard title="Latest Agent Decision"><div className="p-4 text-sm text-muted-foreground">No decision</div></SimpleCard>;

  const when = (()=>{ try{ return data.createdAt ? new Date(data.createdAt).toLocaleString() : '—'; } catch { return '—'; }})();
  const features = data.features && typeof data.features === 'object' ? Object.entries(data.features) : [];
  const actions = Array.isArray(data.actions) ? data.actions : [];

  return (
    <SimpleCard title="Latest Agent Decision">
      <div className="p-4 space-y-3">
        <div className="text-sm text-muted-foreground">{when}</div>
        <div className="text-sm">{data.rationale ?? '—'}</div>
        {features.length>0 && (
          <div>
            <div className="text-xs font-medium mb-1">Features</div>
            <ul className="text-xs text-muted-foreground space-y-0.5">
              {features.map(([k,v]: any)=> (
                <li key={k}><span className="font-medium text-foreground mr-1">{k}:</span>{typeof v==='number'? v.toFixed(2): String(v)}</li>
              ))}
            </ul>
          </div>
        )}
        {actions.length>0 && (
          <div>
            <div className="text-xs font-medium mb-1">Proposed Actions</div>
            <ul className="text-xs text-muted-foreground space-y-0.5">
              {actions.map((a:any,i:number)=> (
                <li key={i}>{a.type} {a.symbol} {a.qty ?? ''} {a.tp?`TP ${a.tp}`:''} {a.sl?`SL ${a.sl}`:''}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </SimpleCard>
  );
}
