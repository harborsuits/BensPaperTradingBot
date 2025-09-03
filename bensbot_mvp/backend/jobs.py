import threading, uuid, time
from fastapi import APIRouter
from .models import JobStartResponse, JobStatus
from .persistence import upsert_job, get_job

router = APIRouter(prefix="/jobs", tags=["jobs"])

def _simulate_long_backtest(job_id:str):
    # replace with real backtest entrypoint; keep structure identical
    for p in range(0, 101, 5):
        upsert_job(job_id, "backtest", "RUNNING", p)
        time.sleep(0.2)
    # store a file or ref id if needed
    upsert_job(job_id, "backtest", "DONE", 100, result_ref=f"report:{job_id}")

@router.post("/backtests", response_model=JobStartResponse)
def start_backtest():
    job_id = uuid.uuid4().hex
    upsert_job(job_id, "backtest", "QUEUED", 0)
    t = threading.Thread(target=_simulate_long_backtest, args=(job_id,), daemon=True)
    t.start()
    return JobStartResponse(job_id=job_id)

@router.get("/{job_id}", response_model=JobStatus)
def job_status(job_id:str):
    row = get_job(job_id)
    if not row:
        return JobStatus(job_id=job_id, status="ERROR", progress=0, error="not_found")
    _id, kind, status, progress, result_ref, error = row
    return JobStatus(job_id=_id, status=status, progress=progress, result_ref=result_ref, error=error)
