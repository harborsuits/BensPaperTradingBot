from fastapi import APIRouter
from datetime import datetime, timezone

router = APIRouter(prefix="/api", tags=["compat"])


@router.get("/context")
async def context_alias():
    return {"items": []}


@router.get("/data/status")
async def data_status_alias():
    return {"last_data_ts": datetime.now(timezone.utc).isoformat()}


@router.get("/strategies/active")
async def strategies_active_alias():
    return []


@router.get("/decisions/latest")
async def decisions_latest_alias(limit: int = 10):
    return {"items": []}


@router.get("/decisions")
async def decisions_alias(limit: int = 50):
    return {"items": []}


def _empty_portfolio():
    return {"data": {"total_equity": 0.0, "cash": 0.0, "positions": []}}


@router.get("/portfolio/paper")
async def portfolio_paper_alias():
    return _empty_portfolio()


@router.get("/portfolio/live")
async def portfolio_live_alias():
    return _empty_portfolio()


