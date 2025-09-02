import axios from "axios";
import { useSafetyStatus } from "@/hooks/useSafetyStatus";
import { Card } from "@/components/ui/Card";

export default function SafetyPage(){
  const { data, refetch, isFetching } = useSafetyStatus();
  const act = async (f:()=>Promise<any>) => { await f(); await refetch(); };

  return (
    <div className="grid gap-3 md:grid-cols-2">
      <Card className="card">
        <div className="card-header"><div className="card-title">Safety Status</div><div className="card-subtle">{isFetching ? "…" : "live"}</div></div>
        <div className="card-content space-y-1">
          <div>Mode: <b>{data?.mode ?? "PAPER"}</b></div>
          <div>Kill Switch: <b>{data?.killSwitch ?? "READY"}</b></div>
          <div>Circuit Breaker: <b>{data?.circuitBreaker ?? "NORMAL"}</b></div>
          <div>Cooldown: <b>{data?.cooldown ?? "READY"}</b></div>
        </div>
      </Card>
      <Card className="card">
        <div className="card-header"><div className="card-title">Controls</div></div>
        <div className="card-content flex gap-2">
          <button className="btn btn-danger" onClick={()=>act(()=>axios.post("/api/safety/emergency-stop"))}>Emergency Stop</button>
          <button className="btn" onClick={()=>act(()=>axios.post("/api/safety/trading-mode",{mode:"paper"}))}>Set PAPER</button>
          <button className="btn" onClick={()=>act(()=>axios.post("/api/safety/trading-mode",{mode:"live"}))}>Set LIVE</button>
        </div>
      </Card>
    </div>
  );
}
