import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import axios from "axios";

type Decision = any;

function normalize(decision: any): Decision {
  const created = decision.createdAt || decision.created_at || decision.timestamp || decision.as_of || new Date().toISOString();
  const action = (decision.action || decision.direction || "").toString().toUpperCase();
  return {
    ...decision,
    id: decision.id || decision.trace_id || decision.traceId || decision.symbol + "-" + created,
    trace_id: decision.trace_id || decision.id,
    action,
    score: typeof decision.score === "number" ? decision.score : undefined,
    one_liner: decision.one_liner || decision.reason,
    strategy: decision.strategy || decision.strategy_name || decision.strategyName || decision.strategy_id,
    createdAt: created,
    decidedAt: decision.decidedAt || created,
    reasons: Array.isArray(decision.reasons) ? decision.reasons : decision.reason ? [decision.reason] : ["No reason provided"],
    plan: decision.plan || { sizePct: 0, slPct: 0, tpPct: 0 },
  };
}

export function useDecisionsRecent(limit = 20) {
  const [live, setLive] = useState<Decision[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  const { data: polled } = useQuery({
    queryKey: ["decisionsRecent", limit],
    queryFn: async () => (await axios.get(`/api/decisions/recent?limit=${limit}`)).data,
    refetchInterval: 7_000,
    staleTime: 5_000,
  });

  // Merge polled results into live state
  useEffect(() => {
    const items = Array.isArray(polled) ? polled : polled?.items || [];
    const normalized = items.filter((d: any) => d?.symbol).map(normalize);
    setLive((prev) => {
      const map = new Map<string, Decision>();
      [...normalized, ...prev].forEach((d: any) => map.set(String(d.trace_id || d.id), d));
      return Array.from(map.values())
        .sort((a: any, b: any) => new Date(b.createdAt || 0).getTime() - new Date(a.createdAt || 0).getTime())
        .slice(0, limit);
    });
  }, [polled, limit]);

  // WS stream for real-time updates
  useEffect(() => {
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const base = (import.meta as any).env?.VITE_WS_BASE_URL;
    const url = (base ? String(base).replace(/\/$/, '') : `${proto}://${window.location.host}`) + `/ws/decisions`;
    const ws = new WebSocket(url);
    wsRef.current = ws;
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        const arr = Array.isArray(msg) ? msg : [msg];
        const normalized = arr.filter((d: any) => d?.symbol).map(normalize);
        setLive((prev) => {
          const map = new Map<string, Decision>();
          [...normalized, ...prev].forEach((d: any) => map.set(String(d.trace_id || d.id), d));
          return Array.from(map.values())
            .sort((a: any, b: any) => new Date(b.createdAt || 0).getTime() - new Date(a.createdAt || 0).getTime())
            .slice(0, limit);
        });
      } catch (e) {
        // ignore bad frames
      }
    };
    return () => {
      try {
        if (wsRef.current === ws) {
          // Avoid closing while still connecting to reduce console errors in StrictMode
          if (ws.readyState === WebSocket.OPEN) {
            ws.close(1000);
          } else {
            ws.onclose = null as any;
            ws.onerror = null as any;
          }
        }
      } catch {}
    };
  }, [limit]);

  const data = useMemo(() => live, [live]);
  return { data } as const;
}


