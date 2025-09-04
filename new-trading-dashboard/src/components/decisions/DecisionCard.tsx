import StageChips from "@/components/common/StageChips";
import ReasonChips from "@/components/common/ReasonChips";
import { buildEvidenceFromUi } from "@/lib/evidence/builders";
import type { DecisionRow, ContextRow } from "@/contracts/types";

export default function DecisionCard({ d, context, onOpenEvidence }:{ d: (DecisionRow & { stageIndex?:number; stageStatus?:string }); context?: ContextRow; onOpenEvidence:(p:any)=>void; }){
  const anyGateRed = (d.gates ?? []).some((g:any)=> !g?.passed);
  const reasons = [d.one_liner ?? d.reason ?? "", d.rr ? `R/R ${d.rr}` : ""].filter(Boolean);

  const handleOpen = ()=> onOpenEvidence(buildEvidenceFromUi({ decision: d, context }));

  return (
    <div className="border rounded-2xl p-4 bg-slate-900/40">
      <div className="flex items-baseline justify-between">
        <div className="flex items-center gap-2">
          <div className="text-lg font-semibold">{d.symbol}</div>
          <span className="text-xs rounded bg-slate-700/60 px-2 py-0.5">{d.strategy ?? "Strategy"}</span>
        </div>
        <div className="text-xs text-muted-foreground">{d.createdAt ? new Date(d.createdAt).toLocaleTimeString() : "now"}</div>
      </div>

      {typeof d.stageIndex === "number" && (
        <div className="mt-2"><StageChips currentIndex={d.stageIndex!} currentStatus={d.stageStatus as any} /></div>
      )}

      <ReasonChips reasons={reasons as any} />

      <div className="flex flex-wrap gap-3 text-xs mt-3">
        {d.entry!==undefined && <span className="rounded bg-slate-800 px-2 py-0.5">Entry {d.entry}</span>}
        {d.target!==undefined && <span className="rounded bg-slate-800 px-2 py-0.5">Target {d.target}</span>}
        {d.stop!==undefined && <span className="rounded bg-slate-800 px-2 py-0.5">Stop {d.stop}</span>}
        {d.size!==undefined && <span className="rounded bg-slate-800 px-2 py-0.5">Size {d.size}%</span>}
        {d.spreadPct!==undefined && <span className="rounded bg-slate-800 px-2 py-0.5">Spread {d.spreadPct}%</span>}
        {d.maxLoss!==undefined && <span className="rounded bg-slate-800 px-2 py-0.5">Max loss ${d.maxLoss}</span>}
      </div>

      <div className="flex gap-2 mt-4">
        <button onClick={handleOpen} className="px-2 py-1 text-xs rounded bg-slate-200 text-slate-900">Open Evidence</button>
        <button disabled={anyGateRed} className="px-2 py-1 text-xs rounded bg-blue-600 text-white disabled:opacity-50">Send to Paper</button>
        {anyGateRed && <span className="text-[11px] text-amber-400">Blocked by gates</span>}
      </div>
    </div>
  );
}


