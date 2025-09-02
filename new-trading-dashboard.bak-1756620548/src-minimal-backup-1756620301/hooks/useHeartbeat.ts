import { useEffect, useRef, useState } from "react";

export type HeartbeatState = {
  connected: boolean;
  latencyMs: number | null;
  lastTickIso: string | null;
};

export function useHeartbeat(pingFn: () => Promise<void>, intervalMs = 5000) {
  const [state, setState] = useState<HeartbeatState>({
    connected: false,
    latencyMs: null,
    lastTickIso: null,
  });
  const timer = useRef<number | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function tick() {
      const start = performance.now();
      try {
        await pingFn(); // e.g., GET /api/health or ws ping
        if (cancelled) return;
        const latency = Math.round(performance.now() - start);
        setState(s => ({ ...s, connected: true, latencyMs: latency, lastTickIso: new Date().toISOString() }));
      } catch {
        if (cancelled) return;
        setState(s => ({ ...s, connected: false }));
      }
    }

    tick();
    timer.current = window.setInterval(tick, intervalMs) as unknown as number;

    return () => {
      cancelled = true;
      if (timer.current) clearInterval(timer.current);
    };
  }, [pingFn, intervalMs]);

  return state;
}
