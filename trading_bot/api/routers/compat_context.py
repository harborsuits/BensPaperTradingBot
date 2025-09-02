from fastapi import APIRouter
from trading_bot.metrics import benbot_context_requests_total, benbot_context_latency_seconds


router = APIRouter(tags=["compat"])


@router.get("/api/context")
def get_context() -> dict:
    """
    Minimal compatibility endpoint for the UI. Returns an empty object
    which the frontend normalizes safely.
    """
    benbot_context_requests_total.labels(path="/api/context").inc()
    with benbot_context_latency_seconds.time():
        return {}


