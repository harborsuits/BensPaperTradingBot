import React, { useState } from "react";

interface StrategyRowProps {
  name: string;
  status: "active" | "idle" | "stopped" | "error";
  priority: number;
  onChangePriority: (n: number) => Promise<void>;
  onStart: () => Promise<void>;
  onStop: () => Promise<void>;
}

export default function StrategyRow({ 
  name, 
  status, 
  priority, 
  onChangePriority, 
  onStart, 
  onStop 
}: StrategyRowProps) {
  const [p, setP] = useState(priority);
  const [busy, setBusy] = useState<null | "start" | "stop" | "prio">(null);

  async function step(delta: number) {
    const next = Math.max(0, p + delta);
    setP(next);
    setBusy("prio");
    try { 
      await onChangePriority(next); 
    } finally { 
      setBusy(null); 
    }
  }

  async function handle(action: "start" | "stop", fn: () => Promise<void>) {
    setBusy(action);
    try { 
      await fn(); 
    } finally { 
      setBusy(null); 
    }
  }

  // Convert status string to Title Case for display
  const statusDisplay = status.charAt(0).toUpperCase() + status.slice(1);

  return (
    <div className="srow" role="row" aria-label={`${name} strategy`}>
      <div className="left">
        <div className={`dot ${status}`} aria-hidden />
        <div className="name">{name}</div>
        <div className="status" aria-live="polite">{statusDisplay}</div>
      </div>

      <div className="right">
        <div className="prio" aria-label="Priority controls">
          <button 
            onClick={() => step(-1)} 
            disabled={busy === "prio" || p === 0} 
            aria-label="Decrease priority"
          >−</button>
          <span aria-live="polite" className="pv">{p}</span>
          <button 
            onClick={() => step(+1)} 
            disabled={busy === "prio"} 
            aria-label="Increase priority"
          >+</button>
        </div>
        
        {status !== "active" ? (
          <button 
            className="act start" 
            onClick={() => handle("start", onStart)} 
            disabled={busy !== null} 
            aria-label={`Start ${name}`}
          >▶</button>
        ) : (
          <button 
            className="act stop" 
            onClick={() => handle("stop", onStop)} 
            disabled={busy !== null} 
            aria-label={`Stop ${name}`}
          >⏹</button>
        )}
      </div>

      <style jsx>{`
        .srow {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 12px 14px;
          border: 1px solid rgba(255,255,255,.08);
          border-radius: 12px;
          background: #0b1223;
        }
        .srow:hover {
          border-color: rgba(255,255,255,.16);
          background: #0d1428;
        }
        .left {
          display: flex;
          align-items: center;
          gap: 10px;
        }
        .dot {
          width: 8px;
          height: 8px;
          border-radius: 999px;
        }
        .dot.active {
          background: #22c55e;
        }
        .dot.idle {
          background: #a3a3a3;
        }
        .dot.stopped {
          background: #ef4444;
        }
        .dot.error {
          background: #f97316;
        }
        .name {
          font-weight: 600;
        }
        .status {
          opacity: .7;
        }
        .right {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .prio {
          display: flex;
          align-items: center;
          gap: 8px;
          border: 1px solid rgba(255,255,255,.08);
          border-radius: 10px;
          padding: 2px 8px;
        }
        .prio button {
          border: none;
          background: transparent;
          color: #e5e7eb;
          font-size: 18px;
          line-height: 20px;
        }
        .pv {
          font-variant-numeric: tabular-nums;
        }
        .act {
          border: 1px solid rgba(255,255,255,.12);
          background: transparent;
          color: #e5e7eb;
          border-radius: 10px;
          padding: 6px 10px;
        }
        .act.start:hover {
          background: rgba(34,197,94,.12);
          border-color: rgba(34,197,94,.35);
        }
        .act.stop:hover {
          background: rgba(239,68,68,.12);
          border-color: rgba(239,68,68,.35);
        }
      `}</style>
    </div>
  );
}
