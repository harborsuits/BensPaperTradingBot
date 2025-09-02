import React from 'react';
import { useActiveStrategies } from '@/hooks/useActiveStrategies';
import CardFrame from '@/components/CardFrame';

export const ActiveStrategiesCard: React.FC = () => {
  const { data, isLoading, isError } = useActiveStrategies();
  const asOf = data && data.length ? Math.max(...data.map((s) => (s.asOf ? Date.parse(s.asOf) : 0))) : undefined;

  return (
    <CardFrame title="Active Strategies" asOf={asOf}>
      {isLoading && <div>Loading…</div>}
      {isError && <div className="text-red-500">Failed to load strategies.</div>}
      {!isLoading && !isError && (!data || data.length === 0) && <div>No active strategies.</div>}
      {!isLoading && !isError && !!data?.length && (
        <table className="w-full text-[13px] border-collapse">
          <thead className="text-muted-foreground">
            <tr>
              <th className="text-left">Name</th>
              <th className="text-left">Signal</th>
              <th className="text-left">Conf.</th>
              <th className="text-right">Pos</th>
              <th className="text-right">P&L (Day)</th>
              <th className="text-left">Health</th>
            </tr>
          </thead>
          <tbody>
            {data.map((s) => (
              <tr key={s.id} className="border-t border-border">
                <td className="py-1.5">{s.name}</td>
                <td>{s.signal}</td>
                <td>{Math.round(s.confidence * 100)}%</td>
                <td className="text-right">{s.positions}</td>
                <td className={`text-right ${Number(s.pnlDay ?? s.pnl_day ?? 0)>=0 ? 'text-green-600':'text-red-600'}`}>
                  {new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:2}).format(Number(s.pnlDay ?? s.pnl_day ?? 0))}
                </td>
                <td>{s.health}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </CardFrame>
  );
};

export default ActiveStrategiesCard;


