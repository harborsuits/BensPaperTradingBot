import axios from 'axios';
import { roster } from './symbolRoster';

const API_BASE = process.env.INTERNAL_API_BASE || 'http://localhost:4000'; // same process ok
const TAKE = (arr: any[], n: number) => arr.slice(0, n);

export async function refreshRosterFromBackend() {
  // pull from your existing endpoints (already shipped):
  // /api/portfolio, /api/paper/orders/open, /api/strategies/active, /api/scanner/candidates, /api/universe, /api/watchlists
  const [portfolio, orders, activeStrats, candidates, universe] = await Promise.allSettled([
    axios.get(`${API_BASE}/api/portfolio`),
    axios.get(`${API_BASE}/api/paper/orders/open`),
    axios.get(`${API_BASE}/api/strategies/active`),
    axios.get(`${API_BASE}/api/scanner/candidates`),
    axios.get(`${API_BASE}/api/universe`)
  ]);

  const tier1 = new Set<string>();
  const tier2 = new Set<string>();
  const tier3 = new Set<string>();
  const pins: string[] = [];

  // Tier1: portfolio + open orders + active strategy required symbols
  if (portfolio.status === 'fulfilled') {
    (portfolio.value.data?.positions || []).forEach((p: any) => {
      const s = String(p.symbol).toUpperCase();
      tier1.add(s);
      pins.push(s);
    });
  }
  if (orders.status === 'fulfilled') {
    (orders.value.data || []).forEach((o: any) => {
      const s = String(o.symbol).toUpperCase();
      tier1.add(s);
      pins.push(s);
    });
  }
  if (activeStrats.status === 'fulfilled') {
    (activeStrats.value.data || []).forEach((s: any) => (s.symbols || []).forEach((sym: string) => tier1.add(sym.toUpperCase())));
  }

  // Tier2: scanner candidates (top N by score) + ingest into scoring
  if (candidates.status === 'fulfilled') {
    const items = (candidates.value.data?.items || []).sort((a: any,b: any)=> (b.score??0)-(a.score??0));
    TAKE(items, 50).forEach((c: any) => {
      const sym = String(c.symbol).toUpperCase();
      tier2.add(sym);
      const score = Math.max(0, Math.min(1, Number(c.score ?? 0.5)));
      roster.ingest({ symbol: sym, reason: 'scanner', score, ttlSec: 3600 });
    });
  }

  // Tier3: rest of universe (cap it)
  if (universe.status === 'fulfilled') {
    const u = (universe.value.data?.symbols || universe.value.data || []);
    (u as any[]).forEach((sym: any) => {
      const v = String(sym).toUpperCase();
      if (!tier1.has(v) && !tier2.has(v)) tier3.add(v);
    });
  }

  roster.setTier('tier1', Array.from(tier1));
  roster.setTier('tier2', Array.from(tier2));
  // keep tier3 modest; cap later in the governor
  roster.setTier('tier3', Array.from(tier3));
  roster.setPins(pins);
}
