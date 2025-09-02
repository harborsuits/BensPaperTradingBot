import React from "react";
import type { Candidate } from "@/types/candidate";
import { buildCandidateNarrative } from "@/utils/candidateNarrative";
import { SkipLabels } from "@/utils/labels";

type Props = { c: Candidate; onProbe?: (id: string)=>void; onTrade?: (id: string)=>void };

export default function CandidateCard({ c, onProbe, onTrade }: Props) {
  const n = buildCandidateNarrative(c);
  const ts = c.timestamp ? new Date(c.timestamp * (c.timestamp < 2e10 ? 1000 : 1)) : null;

  const pass = c.decision === "PASS" || c.decision === "PROBE";
  const border =
    c.decision === "SKIP" ? "border-rose-700/60" :
    c.decision === "PROBE" ? "border-amber-600/60" : "border-emerald-700/60";

  return (
    <div className={`card min-w-0 p-4 md:p-5 border ${border} rounded-2xl bg-neutral-900/60`}>
      <div className="flex items-center justify-between gap-3">
        <div className="text-sm opacity-70">{c.symbol} · {c.kind ?? "asset"}</div>
        <div className="text-xs tabular-nums opacity-60">{ts ? ts.toLocaleTimeString() : ""}</div>
      </div>

      <div className="mt-3 space-y-2">
        <Row label="Why it's on the list" text={n.why} />
        <Row label="Fit with market" text={n.fit} />
        <Row label="Costs & risk" text={n.costs} />
        <Row label="Plan suggestion" text={n.plan} />
        <Row label="Go / No-go" text={n.go} strong />
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        {c.decision === "PROBE" && onProbe && (
          <button
            className="px-3 py-1.5 rounded-lg border border-amber-600 hover:bg-amber-900/30 text-sm"
            onClick={() => onProbe(c.id)}
            title="Open a tiny probe and wait for confirm signal"
          >
            Start probe
          </button>
        )}
        {pass && onTrade && (
          <button
            className="px-3 py-1.5 rounded-lg border border-emerald-700 hover:bg-emerald-900/30 text-sm"
            onClick={() => onTrade(c.id)}
          >
            Submit paper order
          </button>
        )}
        {!pass && c.skip_codes?.length ? (
          <div className="flex flex-wrap gap-1">
            {c.skip_codes.map(code => (
              <span key={code} className="inline-block px-2 py-0.5 rounded-full text-xs bg-neutral-800 border border-neutral-700">
                {SkipLabels[code] ?? code.replace(/_/g," ").toLowerCase()}
              </span>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function Row({ label, text, strong=false }: {label:string; text:string; strong?:boolean}) {
  return (
    <div className="grid grid-cols-[160px_1fr] gap-3 items-start">
      <div className="text-muted-foreground">{label}:</div>
      <div className={strong ? "font-medium" : ""}>{text}</div>
    </div>
  );
}
