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

  // Tier1: portfolio + open orders + active strategy required symbols
  if (portfolio.status === 'fulfilled') {
    (portfolio.value.data?.positions || []).forEach((p: any) => tier1.add(String(p.symbol).toUpperCase()));
  }
  if (orders.status === 'fulfilled') {
    (orders.value.data || []).forEach((o: any) => tier1.add(String(o.symbol).toUpperCase()));
  }
  if (activeStrats.status === 'fulfilled') {
    (activeStrats.value.data || []).forEach((s: any) => (s.symbols || []).forEach((sym: string) => tier1.add(sym.toUpperCase())));
  }

  // Tier2: scanner candidates (top N by score) + first watchlist(s)
  if (candidates.status === 'fulfilled') {
    const items = (candidates.value.data?.items || []).sort((a: any,b: any)=> (b.score??0)-(a.score??0));
    TAKE(items, 50).forEach((c: any) => tier2.add(String(c.symbol).toUpperCase()));
  }

  // Tier3: rest of universe (cap it)
  if (universe.status === 'fulfilled') {
    (universe.value.data || []).forEach((sym: any) => {
      const u = String(sym).toUpperCase();
      if (!tier1.has(u) && !tier2.has(u)) tier3.add(u);
    });
  }

  roster.setTier('tier1', Array.from(tier1));
  roster.setTier('tier2', Array.from(tier2));
  // keep tier3 modest; cap later in the governor
  roster.setTier('tier3', Array.from(tier3));
}
