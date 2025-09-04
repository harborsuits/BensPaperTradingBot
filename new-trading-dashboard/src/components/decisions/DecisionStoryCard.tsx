import { useMemo } from "react";
import { buildEvidenceFromUi, enrichWithWhy } from "@/lib/evidence/builders";
import StageChips from "@/components/common/StageChips";
import { headline as mkHeadline, whyBullets, contribs, risks } from "@/lib/evidence/humanize";
import type { DecisionRow, ContextRow } from "@/contracts/types";

function ContribBar({ label, value, note }:{ label:string; value:number; note?:string }){
  const width = Math.max(0, Math.min(1, value))*100;
  return (
    <div className="text-xs">
      <div className="flex justify-between"><span className="font-medium">{label}</span><span>{width.toFixed(0)}%</span></div>
      <div className="h-1.5 bg-slate-800 rounded"><div className="h-1.5 bg-blue-600 rounded" style={{width: `${width}%`}}/></div>
      {note && <div className="text-[11px] text-slate-400 mt-0.5">{note}</div>}
    </div>
  );
}

function RiskPill({ label, severity }:{ label:string; severity:"low"|"med"|"high" }){
  const cls = severity === "high" ? "bg-red-700/50" : severity === "med" ? "bg-amber-600/40" : "bg-slate-700/40";
  return <span className={`px-2 py-0.5 rounded text-[11px] ${cls}`}>{label}</span>;
}

export default function DecisionStoryCard({ decision, context, onOpenEvidence }:{ decision: (DecisionRow & { stageIndex?:number; stageStatus?:string }); context?: ContextRow; onOpenEvidence:(p:any)=>void; }){
  const anyGateRed = (decision.gates ?? []).some((g:any)=> !g?.passed);
  const failedGates = (decision.gates ?? []).filter((g:any)=> !g?.passed).map((g:any)=> g?.name || "gate");

  const packet = useMemo(()=> enrichWithWhy(buildEvidenceFromUi({ decision, context })), [decision, context]);

  const head = mkHeadline(packet);
  const bullets = whyBullets(packet);
  const contributions = contribs(packet);
  const riskChips = risks(packet);

  return (
    <div className="border rounded-2xl p-4 bg-slate-900/40">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="text-lg font-semibold">{decision.symbol}</div>
          <span className="text-xs rounded bg-slate-700/60 px-2 py-0.5">{decision.strategy ?? "Strategy"}</span>
          <span className="text-xs rounded bg-slate-700/60 px-2 py-0.5">{Math.round((packet.confidence ?? 0)*100)}% conf</span>
        </div>
        <div className="text-xs text-muted-foreground">{decision.createdAt ? new Date(decision.createdAt).toLocaleTimeString() : "now"}</div>
      </div>

      {typeof decision.stageIndex === "number" && (
        <div className="mt-2"><StageChips currentIndex={decision.stageIndex!} currentStatus={decision.stageStatus as any} /></div>
      )}

      <div className="mt-3">
        <div className="text-sm text-slate-100">{head}</div>
        <ul className="list-disc pl-5 mt-1 space-y-1 text-sm">
          {bullets.slice(0,3).map((b,i)=>(<li key={i}>{b}</li>))}
        </ul>
      </div>

      {contributions.length>0 && (
        <div className="mt-3 space-y-1">
          {contributions.map((c)=> (<ContribBar key={c.key} label={c.display ?? c.key} value={c.weight} note={c.rationale} />))}
        </div>
      )}

      <div className="flex flex-wrap gap-2 mt-3 text-xs">
        {decision.entry!==undefined && <span className="rounded bg-slate-800 px-2 py-0.5">Entry {decision.entry}</span>}
        {decision.target!==undefined && <span className="rounded bg-slate-800 px-2 py-0.5">Target {decision.target}</span>}
        {decision.stop!==undefined && <span className="rounded bg-slate-800 px-2 py-0.5">Stop {decision.stop}</span>}
        {decision.spreadPct!==undefined && <span className="rounded bg-slate-800 px-2 py-0.5">Spread {decision.spreadPct}%</span>}
      </div>
      {riskChips.length>0 && (
        <div className="flex flex-wrap gap-2 mt-2">{riskChips.map((r,i)=> <RiskPill key={i} label={r.label} severity={r.severity} />)}</div>
      )}

      <div className="mt-3 flex flex-wrap gap-2 text-[11px] text-slate-300">
        {(packet.sources ?? []).slice(0,3).map((s)=> (
          <span key={s.id} className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700/60">{s.publisher} • {s.credibility}/5</span>
        ))}
      </div>

      <div className="flex gap-2 mt-4 items-center">
        <button onClick={()=> onOpenEvidence(packet)} className="px-2 py-1 text-xs rounded bg-slate-200 text-slate-900">Open Evidence</button>
        <button
          disabled={anyGateRed}
          title={anyGateRed ? `Blocked: ${failedGates.join(', ')}` : ''}
          className="px-2 py-1 text-xs rounded bg-blue-600 text-white disabled:opacity-50"
        >
          Send to Paper
        </button>
      </div>
    </div>
  );
}
