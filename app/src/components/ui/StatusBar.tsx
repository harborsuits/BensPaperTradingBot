import React from "react";
import { HeartbeatState } from "@/hooks/useHeartbeat";

type Props = {
  mode: "Paper" | "Live";
  marketOpen: boolean;
  heartbeat: HeartbeatState;
  onPauseToggle?: () => void;
  paused?: boolean;
};

const dot = (ok: boolean) => (
  <span aria-hidden className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: ok ? "#22c55e" : "#ef4444" }} />
);

export default function StatusBar({ mode, marketOpen, heartbeat, onPauseToggle, paused }: Props) {
  const last = heartbeat.lastTickIso ? new Date(heartbeat.lastTickIso).toLocaleTimeString() : "—";
  return (
    <div className="statusbar">
      <div className="cell">
        <strong>Market</strong>
        <div className="row">{dot(marketOpen)} <span>{marketOpen ? "Open" : "Closed"}</span></div>
      </div>

      <div className="cell">
        <strong>Account</strong>
        <div className="row">
          <span className={`badge ${mode === "Live" ? "live" : "paper"}`}>{mode}</span>
        </div>
      </div>

      <div className="cell">
        <strong>WebSocket</strong>
        <div className="row">{dot(heartbeat.connected)} <span>{heartbeat.connected ? "Connected" : "Disconnected"}</span></div>
      </div>

      <div className="cell">
        <strong>Latency</strong>
        <div className="row"><span>{heartbeat.latencyMs ?? "—"} ms</span></div>
      </div>

      <div className="cell">
        <strong>Last Tick</strong>
        <div className="row"><span>{last}</span></div>
      </div>

      {!!onPauseToggle && (
        <div className="cell">
          <button
            className={`btn ${paused ? "btn-danger" : "btn-ghost"}`}
            onClick={onPauseToggle}
            aria-pressed={paused}
            aria-label={paused ? "Resume trading" : "Pause trading"}
          >
            {paused ? "Resume" : "Pause"}
          </button>
        </div>
      )}

      {/* styles */}
      <style>{`
        .statusbar {
          display:flex; gap:16px; align-items:center;
          padding:10px 16px; border:1px solid rgba(255,255,255,0.08);
          background:#0f172a; border-radius:12px; position:sticky; top:0; z-index:30;
        }
        .cell { display:flex; flex-direction:column; gap:4px; min-width:110px; }
        .row { display:flex; align-items:center; gap:8px; opacity:0.9; }
        .badge { padding:2px 8px; border-radius:999px; font-size:12px; line-height:18px; }
        .badge.paper { background:rgba(59,130,246,.15); color:#93c5fd; border:1px solid rgba(59,130,246,.35);}
        .badge.live { background:rgba(34,197,94,.15); color:#86efac; border:1px solid rgba(34,197,94,.35);}
        .btn { padding:6px 10px; border-radius:10px; border:1px solid rgba(255,255,255,0.1); background:transparent; color:#e5e7eb; }
        .btn:hover { background:rgba(255,255,255,0.06); }
        .btn-danger { background:rgba(239,68,68,.15); border-color:rgba(239,68,68,.35); color:#fecaca; }
      `}</style>
    </div>
  );
}
