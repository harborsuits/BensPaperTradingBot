from typing import Dict, Any
import os
from fastapi import APIRouter, Request
from trading_bot.core.safety.pause_state import is_paused

router = APIRouter(prefix="", tags=["metrics"])


@router.get("/metrics")
async def get_metrics(request: Request) -> Dict[str, Any]:
    app = request.app
    exec_ = getattr(app.state, "trade_executor", None)
    orders_counts = {}
    open_trades = 0

    if exec_ is not None:
        orders_counts = getattr(exec_, "metrics_counts", {}) or {}
        active = getattr(exec_, "active_trades", {}) or {}
        open_trades = len(active)

    return {
        "entries_paused": is_paused(),
        "trading_mode": os.environ.get("TRADING_MODE", "paper"),
        "reconcile_runs": getattr(app.state, "reconcile_runs", 0),
        "reconcile_mismatches_total": getattr(app.state, "reconcile_mismatches_total", 0),
        "orders": orders_counts,
        "open_trades": open_trades,
    }


