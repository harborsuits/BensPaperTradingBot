import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

_SOURCE_STATE: Dict[str, Dict[str, Any]] = {}

def update_source(source: str, latency_ms: Optional[float] = None, error: bool = False) -> None:
    now = datetime.utcnow().isoformat()
    state = _SOURCE_STATE.setdefault(source, {"last_seen_at": None, "latency_ms": None, "error_count": 0})
    state["last_seen_at"] = now
    if latency_ms is not None:
        state["latency_ms"] = float(latency_ms)
    if error:
        state["error_count"] = int(state.get("error_count", 0)) + 1

def get_freshness(status_thresholds: Dict[str, int] = None) -> Dict[str, Any]:
    """
    Return freshness info for each source with a status:
      green: < 5m, yellow: < 30m, red: >= 30m or never.
    """
    thresholds = status_thresholds or {"green": 300, "yellow": 1800}  # seconds
    result: Dict[str, Any] = {}
    now = datetime.utcnow()
    for source, state in _SOURCE_STATE.items():
        last = state.get("last_seen_at")
        if not last:
            status = "red"
            age_s = None
        else:
            try:
                dt = datetime.fromisoformat(last)
            except Exception:
                dt = now
            age = (now - dt)
            age_s = int(age.total_seconds())
            if age_s < thresholds["green"]:
                status = "green"
            elif age_s < thresholds["yellow"]:
                status = "yellow"
            else:
                status = "red"
        result[source] = {
            "last_seen_at": state.get("last_seen_at"),
            "latency_ms": state.get("latency_ms"),
            "error_count": state.get("error_count", 0),
            "age_seconds": age_s,
            "status": status,
        }
    return result


