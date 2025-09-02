from fastapi import APIRouter, Response, status
from typing import Any, Dict

from trading_bot.diagnostics.data_sources_check import check_all

try:
    from trading_bot.services.metrics import snapshot_usage
except Exception:
    def snapshot_usage() -> Dict[str, Any]:
        return {}

router = APIRouter(prefix="/api/health", tags=["health"])

@router.get("")
async def root_health(resp: Response):
    payload = await check_all()
    ok = all(v.get("ok") for v in payload.values())
    resp.status_code = status.HTTP_200_OK if ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return payload

@router.get("/data-sources")
async def data_sources():
    return await check_all()

@router.get("/usage")
async def usage():
    return snapshot_usage()


