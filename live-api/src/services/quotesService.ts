import EventEmitter from 'events';
import { resolveQuoteProvider } from '../providers/quotes';
import type { Quote } from '../providers/quotes/QuoteProvider';
import { roster } from './symbolRoster';
import { refreshRosterFromBackend } from './symbolRosterLoader';

const emitter = new EventEmitter();
let cache: Record<string, Quote> = {};
let lastRefresh = 0;
let lastError: string | null = null;

const BASE_MS   = Number(process.env.QUOTES_REFRESH_MS || 5000);
const OFF_MS    = Number(process.env.QUOTES_REFRESH_MS_OFFHOURS || 15000);
const QPM       = Number(process.env.QUOTES_RATE_QPM || 60);
const BATCH     = Number(process.env.QUOTES_BATCH_SIZE || 50);
const MAX_SYMS  = Number(process.env.QUOTES_MAX_SYMBOLS || 400);
const T1_MULT   = Number(process.env.QUOTES_TIER1_REFRESH_MULT || 1);
const T2_MULT   = Number(process.env.QUOTES_TIER2_REFRESH_MULT || 3);
const T3_MULT   = Number(process.env.QUOTES_TIER3_REFRESH_MULT || 6);

const provider  = resolveQuoteProvider();

// Simple token bucket for QPM
let tokens = QPM;
setInterval(()=> { tokens = QPM; }, 60_000);

function inMarketHours(now = new Date()) {
  // simple guard; if you have a market-hours util, use it
  const day = now.getUTCDay(); // 0 Sun ... 6 Sat
  if (day === 0 || day === 6) return false;
  
  // Check US market hours (9:30 AM - 4:00 PM ET)
  const hour = now.getUTCHours();
  const minute = now.getUTCMinutes();
  
  // Convert to ET (UTC-4 or UTC-5 depending on DST)
  // Simplified: just subtract 4 hours (assuming EDT)
  const etHour = (hour - 4 + 24) % 24;
  
  // Market hours: 9:30 AM - 4:00 PM ET
  if (etHour < 9 || etHour >= 16) return false;
  if (etHour === 9 && minute < 30) return false;
  
  return true;
}

let tickIdx = 0;

async function refreshBatch(symbols: string[]) {
  if (!symbols.length) return;
  try {
    const data = await provider.getQuotes(symbols);
    Object.assign(cache, data);
    lastRefresh = Date.now();
    emitter.emit('quotes', { quotes: data, time: new Date(lastRefresh).toISOString() });
  } catch (e: any) {
    lastError = e?.message || String(e);
    throw e; // Let caller handle it
  }
}

async function governorTick() {
  const all = roster.getAll().slice(0, MAX_SYMS);
  if (!all.length) return;

  // slice into tiers with staggered cadence
  const t1 = all.filter(s => roster.tier1.has(s));
  const t2 = all.filter(s => roster.tier2.has(s));
  const t3 = all.filter(s => roster.tier3.has(s));

  const wants: string[] = [];
  if (tickIdx % T1_MULT === 0) wants.push(...t1);
  if (tickIdx % T2_MULT === 0) wants.push(...t2);
  if (tickIdx % T3_MULT === 0) wants.push(...t3);
  tickIdx++;

  // dedupe + chunk
  const wanted = Array.from(new Set(wants));
  const chunks: string[][] = [];
  for (let i=0; i<wanted.length; i+=BATCH) chunks.push(wanted.slice(i, i+BATCH));

  for (const chunk of chunks) {
    if (tokens <= 0) break;
    tokens--; // 1 API call per chunk
    try { 
      await refreshBatch(chunk); 
      console.log(`[QuotesService] Refreshed ${chunk.length} symbols, ${tokens} tokens left`);
    }
    catch (e:any) { 
      lastError = e?.message || String(e);
      console.error(`[QuotesService] Error refreshing quotes: ${lastError}`);
    }
    // tiny delay between chunks to be gentle
    await new Promise(r => setTimeout(r, 50));
  }
}

export function onQuotes(cb: (p: {quotes: Record<string, Quote>, time: string}) => void) {
  emitter.on('quotes', cb); return () => emitter.off('quotes', cb);
}

export function getQuotesCache() { 
  return { quotes: cache, asOf: new Date(lastRefresh).toISOString() }; 
}

export function getQuotesStatus() {
  return {
    provider: process.env.QUOTES_PROVIDER || 'auto',
    autorefresh: process.env.AUTOREFRESH_ENABLED === '1',
    refreshMs: inMarketHours() ? BASE_MS : OFF_MS,
    qpmBudget: QPM,
    batchSize: BATCH,
    symbolsCached: Object.keys(cache).length,
    tier1: roster.tier1.size,
    tier2: roster.tier2.size,
    tier3: roster.tier3.size,
    lastRefresh: lastRefresh ? new Date(lastRefresh).toISOString() : null,
    lastError,
    marketHours: inMarketHours(),
  };
}

let loop: NodeJS.Timeout | null = null;
let rosterTimer: NodeJS.Timeout | null = null;

export function startQuotesLoop() {
  if (String(process.env.AUTOREFRESH_ENABLED) !== '1') {
    console.log('[QuotesService] Autorefresh disabled, not starting quotes loop');
    return;
  }

  console.log('[QuotesService] Starting quotes loop');

  // initial roster load + periodic refresh (rate-safe)
  refreshRosterFromBackend().catch(()=>{});
  rosterTimer = setInterval(() => refreshRosterFromBackend().catch(()=>{}), 90_000);

  const tick = async () => {
    await governorTick().catch(()=>{});
    const next = inMarketHours() ? BASE_MS : OFF_MS;
    loop = setTimeout(tick, next);
  };
  tick();
}

export function stopQuotesLoop() {
  if (loop) clearTimeout(loop);
  if (rosterTimer) clearInterval(rosterTimer);
  loop = null; rosterTimer = null;
  console.log('[QuotesService] Stopped quotes loop');
}
