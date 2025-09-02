from fastapi import APIRouter, Query
from typing import Any, Dict, List
from trading_bot.metrics import benbot_logs_total


router = APIRouter(tags=["compat"])


@router.get("/api/logs")
def get_logs(
    level: str = Query("INFO"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Canonical UI contract: return {"items": [...]}.
    Replace the stub with your real source later.
    """
    benbot_logs_total.labels(source="rest").inc()
    return {"items": []}


