export default function ReasonChips({ reasons }:{ reasons: Array<string|number> }){
  if (!reasons?.length) return null;
  return (
    <div className="flex flex-wrap gap-2 mt-2">
      {reasons.map((r,i)=> (
        <span key={i} className="px-2 py-0.5 rounded-full text-[11px] bg-slate-700/40 text-slate-200 border border-slate-600/40">{String(r)}</span>
      ))}
    </div>
  );
}


