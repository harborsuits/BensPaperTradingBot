from fastapi import APIRouter, HTTPException, Query
import httpx
import os
import asyncio
import time
from aiolimiter import AsyncLimiter
from cachetools import TTLCache
from typing import List, Dict, Any, Optional

router = APIRouter(prefix="/api", tags=["market"])

# API endpoints
DATA = os.environ.get("ALPACA_DATA_BASE", "https://data.alpaca.markets/v2")
PAPER = os.environ.get("ALPACA_PAPER_BASE", "https://paper-api.alpaca.markets/v2")
HEADERS = {
    "APCA-API-KEY-ID": os.environ.get("ALPACA_KEY_ID", ""),
    "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET_KEY", ""),
}

# Rate limits & caches
quote_limiter = AsyncLimiter(180, 60)     # 180/min (buffer below 200/min)
quote_cache = TTLCache(maxsize=5000, ttl=0.9)  # ~1s TTL for quotes
bars_cache = TTLCache(maxsize=2000, ttl=5)    # 5s TTL for intraday bars

async def _get_quote(symbol: str):
    """Get a quote for a single symbol with caching and rate limiting"""
    key = symbol.upper()
    now = time.time()

    if key in quote_cache:
        return {**quote_cache[key], "stale": False}

    async with quote_limiter:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(f"{DATA}/stocks/{key}/quotes/latest", headers=HEADERS)
        if r.status_code == 200:
            payload = r.json()
            res = {"symbol": key, "quote": payload.get("quote")}
            quote_cache[key] = res
            return {**res, "stale": False}

        if r.status_code == 404:
            # return empty but cache briefly to avoid hammering
            res = {"symbol": key, "quote": None}
            quote_cache[key] = res
            return {**res, "stale": True}

        raise HTTPException(status_code=r.status_code, detail=r.text)


@router.get("/quotes")   # GET /api/quotes?symbols=AAPL,MSFT,SPY
async def quotes(symbols: str = Query(..., description="CSV of symbols")):
    """Get quotes for multiple symbols in a single batched request"""
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not syms:
        return {"quotes": {}}
    
    results = await asyncio.gather(*[_get_quote(s) for s in syms], return_exceptions=True)

    out = {}
    for s, r in zip(syms, results):
        if isinstance(r, Exception):
            # serve stale if present instead of failing outright
            cached = quote_cache.get(s)
            if cached:
                out[s] = {**cached, "stale": True, "error": "upstream_error"}
            else:
                out[s] = {"symbol": s, "quote": None, "stale": True, "error": "upstream_error"}
        else:
            out[s] = r
    return {"quotes": out}


@router.get("/bars")  # passthrough w/ small cache
async def bars(symbol: str, timeframe: str = "1Min", start: str = None, end: str = None, limit: int = 500):
    """Get bars for a symbol with caching"""
    ck = f"{symbol}|{timeframe}|{start}|{end}|{limit}"
    if ck in bars_cache:
        return {**bars_cache[ck], "stale": False}
    
    params = {"timeframe": timeframe, "limit": limit}
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{DATA}/stocks/{symbol}/bars", headers=HEADERS, params=params)
    
    if r.status_code == 200:
        data = r.json()
        bars_cache[ck] = data
        return {**data, "stale": False}
    
    if r.status_code == 404:
        return {"bars": [], "stale": True}
    
    raise HTTPException(status_code=r.status_code, detail=r.text)


@router.get("/health/market")
async def health_check():
    """Health check for market data services"""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{DATA}/stocks/SPY/quotes/latest", headers=HEADERS)
        return {
            "status": "ok" if r.status_code == 200 else "degraded",
            "code": r.status_code,
            "latency_ms": round(r.elapsed.total_seconds() * 1000)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }