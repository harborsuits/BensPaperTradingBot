import { STAGES } from "@/lib/flow/utils";

export default function StageChips({ currentIndex, currentStatus }:{ currentIndex:number; currentStatus?:string }){
  return (
    <div className="flex flex-wrap gap-2">
      {STAGES.map((s,i)=>{
        const color = i<currentIndex
          ? "bg-green-600 text-white"
          : i===currentIndex
          ? (currentStatus==="ok"?"bg-amber-500 text-white":"bg-red-600 text-white")
          : "bg-slate-200 text-slate-700";
        return <span key={s} className={`px-2 py-0.5 rounded text-[10px] ${color}`}>{s}</span>;
      })}
    </div>
  );
}


