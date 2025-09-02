// src/services/candidates.ts
import { z } from "zod";

// Be permissive: keep unknown fields to avoid breaking UI if backend adds more.
export const Candidate = z.object({
  symbol: z.string().min(1).transform(s => s.toUpperCase()),
  score: z.number().optional(),
  confidence: z.number().optional(),
  side: z.enum(["buy", "sell"]).optional(),
  last: z.number().optional(),
  plan: z.object({
    entry: z.number().optional(),
    atr: z.number().optional(),
  }).optional(),
  explain: z.object({
    feeBps: z.number().optional(),
    slipBps: z.number().optional(),
    spreadPct: z.number().optional(),
    relativeVolume: z.number().optional(),
    tone: z.string().optional(),
    noveltyScore: z.number().optional(),
  }).optional(),
  risk: z.object({
    heatDelta: z.number().optional(),
    suggestedQty: z.number().optional(),
    spreadOK: z.boolean().optional(),
  }).optional(),
  // Add schema_version and asOf for watermark
  schema_version: z.string().optional(),
  asOf: z.string().optional(),
}).passthrough();

export type Candidate = z.infer<typeof Candidate>;

const CandArray = z.array(Candidate);
const CandEnvelopeItems = z.object({ items: CandArray });
const CandEnvelopeCandidates = z.object({ candidates: CandArray });

const AnySupported = z.union([CandArray, CandEnvelopeItems, CandEnvelopeCandidates]);

function logUnexpectedShape(json: unknown, reason: string) {
  try {
    // keep it light; you can route to Sentry/console endpoint later
    // @ts-ignore
    console.warn("candidates:unexpected_shape", { reason, keys: json && typeof json === "object" ? Object.keys(json) : null });
  } catch {}
}

function dedupeBySymbol(arr: Candidate[]): Candidate[] {
  const map = new Map<string, Candidate>();
  for (const c of arr) {
    if (!c?.symbol) continue;
    if (!map.has(c.symbol)) map.set(c.symbol, c);
  }
  return [...map.values()];
}

export function normalizeCandidatesResponse(json: unknown): Candidate[] {
  const parsed = AnySupported.safeParse(json);
  if (parsed.success) {
    const v = parsed.data as unknown;
    const arr = Array.isArray(v)
      ? v as Candidate[]
      : "items" in (v as any)
        ? (v as any).items as Candidate[]
        : (v as any).candidates as Candidate[];
    return dedupeBySymbol(arr);
  }

  // Salvage mode: if backend sends { data: [...] } or any object containing an array of objects with 'symbol'
  if (json && typeof json === "object") {
    for (const val of Object.values(json as Record<string, unknown>)) {
      if (Array.isArray(val) && val.every(x => x && typeof x === "object" && "symbol" in (x as any))) {
        const salvage = CandArray.safeParse(val);
        if (salvage.success) return dedupeBySymbol(salvage.data);
      }
    }
  }

  logUnexpectedShape(json, "no-supported-shape");
  return [];
}
