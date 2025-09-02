from fastapi import APIRouter
from datetime import datetime, timezone


router = APIRouter(tags=["compat"])


@router.get("/metrics")
def metrics_json() -> dict:
    """
    Minimal JSON metrics for UI compatibility when Prometheus isn't used.
    """
    return {
        "benbot_ws_connected": True,
        "benbot_data_fresh_seconds": 0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


