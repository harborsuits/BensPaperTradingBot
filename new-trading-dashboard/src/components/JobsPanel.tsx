import { useState, useEffect } from "react";
import { useDashboardData } from "@/hooks/useDashboardData";
import { JobsSvc } from "@/services/trading";

export default function JobsPanel() {
  const { startBacktest } = useDashboardData();
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("IDLE");

  useEffect(() => {
    let t: any;
    if (jobId) {
      t = setInterval(async () => {
        const s = await JobsSvc.status(jobId);
        setProgress(s.progress);
        setStatus(s.status);
        if (s.status === "DONE" || s.status === "ERROR") clearInterval(t);
      }, 1000);
    }
    return () => clearInterval(t);
  }, [jobId]);

  return (
    <div>
      <button
        className="px-3 py-2 bg-black text-white rounded"
        onClick={async () => {
          const r = await startBacktest.mutateAsync();
          setJobId(r.job_id);
          setProgress(0); setStatus("RUNNING");
        }}
      >Start Backtest</button>
      {jobId && (
        <div className="mt-3">
          <div className="text-sm text-gray-500">Job {jobId} • {status}</div>
          <div className="w-full bg-gray-200 h-2 rounded overflow-hidden">
            <div className="bg-green-500 h-2" style={{ width: `${progress}%` }} />
          </div>
        </div>
      )}
    </div>
  );
}
