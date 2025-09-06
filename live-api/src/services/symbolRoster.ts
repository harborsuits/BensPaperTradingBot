import { EventEmitter } from 'events';

type Tier = 'tier1'|'tier2'|'tier3';
type Sub = { symbol: string; expiresAt: number };

type Source = 'news'|'earnings'|'position'|'order'|'gap'|'volume'|'scanner'|'subscription'|'tier1'|'tier2'|'tier3'|'pin';

type ScoreRecord = { score: number; updated: number; reasons: Record<string, number> };

type Hit = { symbol: string; reason: Source; score: number; ttlSec?: number; meta?: any };

const subs = new Map<string, Sub>(); // UI subscriptions w/ TTL
const em = new EventEmitter();

// Scoring store and pins
const scores = new Map<string, ScoreRecord>();
const pins = new Set<string>();

const HALFLIFE_MS = 30 * 60 * 1000; // 30 minutes
const LIMIT_RTH = 150;
const LIMIT_OOH = 50;

function decayFactor(sinceMs: number) {
  return Math.exp(-(sinceMs) / HALFLIFE_MS);
}

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
    // include scored and pinned symbols
    scores.forEach((_, k) => out.add(k));
    pins.forEach((k) => out.add(k));
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
    // Boost subscription symbols slightly in scoring so they are likely included
    symbols.map(s => s.toUpperCase()).forEach(sym => this.ingest({ symbol: sym, reason: 'subscription', score: 0.3, ttlSec }));
    em.emit('updated');
  },

  onUpdated(cb: () => void) { em.on('updated', cb); return () => em.off('updated', cb); },

  // Scoring ingest API
  ingest(hit: Hit) {
    const now = Date.now();
    const sym = hit.symbol.toUpperCase();
    const rec: ScoreRecord = scores.get(sym) || { score: 0, updated: now, reasons: {} };
    const dec = decayFactor(now - rec.updated);
    const base = rec.score * dec;
    const added = Math.min(1, Math.max(0, Number(hit.score) || 0));
    const next = Math.min(5, base + added);
    rec.score = next;
    rec.updated = now;
    rec.reasons[hit.reason] = (rec.reasons[hit.reason] || 0) + added;
    scores.set(sym, rec);
    em.emit('updated');
  },

  pin(symbol: string, on: boolean) {
    const sym = symbol.toUpperCase();
    if (on) pins.add(sym); else pins.delete(sym);
    em.emit('updated');
  },

  setPins(symbols: string[]) {
    pins.clear();
    symbols.forEach((s) => pins.add(s.toUpperCase()));
    em.emit('updated');
  },

  // Compute active roster with scoring + tiers + subscriptions + pins
  activeRoster(now = Date.now(), isRTH = true) {
    const limit = isRTH ? LIMIT_RTH : LIMIT_OOH;
    const aggregate = new Map<string, { symbol: string; score: number; reasons: Record<string, number> }>();

    const add = (symbol: string, score: number, reason: Source) => {
      const sym = symbol.toUpperCase();
      const cur = aggregate.get(sym) || { symbol: sym, score: 0, reasons: {} };
      cur.score += score;
      cur.reasons[reason] = (cur.reasons[reason] || 0) + score;
      aggregate.set(sym, cur);
    };

    // decayed scores
    scores.forEach((rec, sym) => {
      const s = rec.score * decayFactor(now - rec.updated);
      if (s > 0.01) add(sym, s, 'scanner');
    });

    // tiers contribute baseline priority
    this.tier1.forEach((sym) => add(sym, 3.0, 'tier1'));
    this.tier2.forEach((sym) => add(sym, 1.5, 'tier2'));
    this.tier3.forEach((sym) => add(sym, 0.5, 'tier3'));

    // subscriptions
    subs.forEach((sub, sym) => { if (sub.expiresAt > now) add(sym, 1.0, 'subscription'); });

    // pins dominate
    pins.forEach((sym) => add(sym, 999, 'pin'));

    const items = Array.from(aggregate.values())
      .filter(x => x.score > 0.05)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);

    return items;
  }
};
