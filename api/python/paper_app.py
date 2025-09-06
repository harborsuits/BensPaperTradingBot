#!/usr/bin/env python3
import os
from datetime import datetime
from typing import Optional

import requests
try:
    import dotenv; dotenv.load_dotenv()
except Exception:
    pass
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel

# Canonical envs with alias fallbacks
TRADIER_API_KEY = os.getenv("TRADIER_API_KEY") or os.getenv("TRADIER_TOKEN") or ""
TRADIER_ACCOUNT_ID = os.getenv("TRADIER_ACCOUNT_ID") or ""
TRADIER_BASE_URL = os.getenv("TRADIER_BASE_URL") or os.getenv("TRADIER_API_URL") or "https://sandbox.tradier.com/v1"

COMMON_HEADERS = {
    "Accept": "application/json",
    "Authorization": f"Bearer {TRADIER_API_KEY}",
}

app = FastAPI(title="Paper Broker Shim", version="1.0.0")


class OrderIn(BaseModel):
    symbol: str
    side: str
    qty: int
    type: str
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    duration: str = "day"


def _require_env():
    if not TRADIER_API_KEY or not TRADIER_ACCOUNT_ID:
        raise HTTPException(status_code=500, detail="Missing TRADIER_API_KEY or TRADIER_ACCOUNT_ID")


@app.get("/")
def root():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}


@app.post("/paper/orders")
async def paper_place_order(order: OrderIn, idempotency_key: Optional[str] = Header(default=None, convert_underscores=False)):
    _require_env()
    headers = dict(COMMON_HEADERS)
    headers["Content-Type"] = "application/x-www-form-urlencoded"
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key

    payload = {
        "class": "equity",
        "symbol": order.symbol,
        "side": order.side,
        "quantity": str(order.qty),
        "type": order.type,
        "duration": order.duration,
    }
    if order.limit_price is not None:
        payload["price"] = str(order.limit_price)
    if order.stop_price is not None:
        payload["stop"] = str(order.stop_price)

    try:
        url = f"{TRADIER_BASE_URL}/accounts/{TRADIER_ACCOUNT_ID}/orders"
        r = requests.post(url, headers=headers, data=payload, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"paper order failed: {e}")


@app.get("/paper/orders/{oid}")
def paper_get_order(oid: str):
    _require_env()
    try:
        url = f"{TRADIER_BASE_URL}/accounts/{TRADIER_ACCOUNT_ID}/orders/{oid}"
        r = requests.get(url, headers=COMMON_HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"paper get order failed: {e}")


@app.get("/paper/positions")
def paper_positions():
    _require_env()
    try:
        url = f"{TRADIER_BASE_URL}/accounts/{TRADIER_ACCOUNT_ID}/positions"
        r = requests.get(url, headers=COMMON_HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"paper positions failed: {e}")


@app.get("/paper/account")
def paper_account():
    _require_env()
    try:
        url = f"{TRADIER_BASE_URL}/accounts/{TRADIER_ACCOUNT_ID}/balances"
        r = requests.get(url, headers=COMMON_HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"paper account failed: {e}")


