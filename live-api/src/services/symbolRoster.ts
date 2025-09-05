import { EventEmitter } from 'events';

type Tier = 'tier1'|'tier2'|'tier3';
type Sub = { symbol: string; expiresAt: number };

const subs = new Map<string, Sub>(); // UI subscriptions w/ TTL
const em = new EventEmitter();

export const roster = {
  // computed sets
  tier1: new Set<string>(), // portfolio/open orders/active strategies
  tier2: new Set<string>(), // scanner candidates + key watchlists
  tier3: new Set<string>(), // full universe tail

  getAll(): string[] {
    const out = new Set<string>();
    [this.tier1, this.tier2, this.tier3].forEach(s => s.forEach(x => out.add(x)));
    // include valid subscriptions
    const now = Date.now();
    subs.forEach((v, k) => { if (v.expiresAt > now) out.add(k); else subs.delete(k); });
    return Array.from(out);
  },

  // external setters (called by a small loader below)
  setTier(tier: Tier, syms: string[]) {
    const tgt = tier === 'tier1' ? this.tier1 : tier === 'tier2' ? this.tier2 : this.tier3;
    tgt.clear();
    syms.forEach(s => tgt.add(s.toUpperCase()));
    em.emit('updated');
  },

  subscribe(symbols: string[], ttlSec: number) {
    const exp = Date.now() + Math.max(5, ttlSec) * 1000;
    symbols.map(s => s.toUpperCase()).forEach(sym => subs.set(sym, { symbol: sym, expiresAt: exp }));
    em.emit('updated');
  },

  onUpdated(cb: () => void) { em.on('updated', cb); return () => em.off('updated', cb); }
};
