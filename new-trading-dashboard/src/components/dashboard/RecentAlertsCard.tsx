import { useQuery } from "@tanstack/react-query";
import type { AlertRow } from "@/contracts/types";

export default function RecentAlertsCard(){
  const { data } = useQuery<AlertRow[]>({
    queryKey:["alerts","recent"],
    queryFn: async ()=> (await fetch("/api/alerts?limit=50")).json(),
    refetchInterval: 30000, staleTime: 15000,
  });

  const important = (data ?? [])
    .sort((a,b)=> new Date(b.ts).getTime() - new Date(a.ts).getTime())
    .slice(0,5);

  return (
    <div className="border rounded-2xl p-4">
      <div className="flex items-baseline justify-between">
        <h3 className="text-lg font-semibold">Recent Alerts</h3>
        <a href="/logs" className="text-xs underline">View all logs</a>
      </div>
      <div className="mt-2 space-y-1">
        {important.map(a=> (
          <div key={(a.id ?? "")+a.ts} className="border rounded-xl p-2">
            <div className="flex justify-between"><span className="font-medium text-xs">{a.level}</span><span className="text-xs text-muted-foreground">{new Date(a.ts).toLocaleTimeString()}</span></div>
            <div className="text-xs text-muted-foreground">{a.source ?? "system"}</div>
            <div className="text-sm">{a.message}</div>
          </div>
        ))}
        {important.length===0 && <div className="text-sm text-muted-foreground">No recent alerts.</div>}
      </div>
    </div>
  );
}


