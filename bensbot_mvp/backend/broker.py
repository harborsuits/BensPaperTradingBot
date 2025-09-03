import os, httpx
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential

BROKER = os.getenv("BROKER","tradier").lower()

def _client(base_url:str, api_key:str):
    headers = {}
    if BROKER=="tradier":
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded"
        }
    elif BROKER=="alpaca":
        headers = {
            "APCA-API-KEY-ID": os.getenv("ALPACA_KEY_ID",""),
            "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY","")
        }
    return httpx.Client(base_url=base_url, timeout=6.0, headers=headers)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.2, max=3))
def _get(c:httpx.Client, path:str, params:Dict[str,Any]=None):
    r=c.get(path, params=params or {})
    r.raise_for_status()
    return r.json()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.2, max=3))
def _post(c:httpx.Client, path:str, data:Dict[str,Any]=None):
    r=c.post(path, data=data or {})
    r.raise_for_status()
    return r.json()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.2, max=3))
def _delete(c:httpx.Client, path:str):
    r=c.delete(path)
    r.raise_for_status()
    # Tradier cancel returns 204 w/ no JSON; just return ok
    try:
        return r.json()
    except Exception:
        return {"ok": True}

# ---------------- Tradier helpers ----------------

def _tradier():  # one place to open client
    return _client(os.getenv("TRADIER_BASE_URL",""), os.getenv("TRADIER_API_KEY",""))

def _acct_id():
    return os.getenv("TRADIER_ACCOUNT_ID","")

def _quotes_map(symbols: List[str]) -> Dict[str, float]:
    if not symbols:
        return {}
    with _tradier() as c:
        j = _get(c, "/v1/markets/quotes", params={"symbols": ",".join(sorted(set(symbols)))})
        q = j.get("quotes",{}).get("quote", [])
        if isinstance(q, dict): q = [q]
        out = {}
        for item in q or []:
            # prefer last; fallback to close / bid/ask mid if missing
            last = item.get("last")
            if last is None:
                close = item.get("prevclose") or item.get("close")
                bid = item.get("bid"); ask = item.get("ask")
                last = close if close is not None else ( (bid+ask)/2.0 if (bid is not None and ask is not None) else 0.0 )
            out[item.get("symbol")] = float(last or 0.0)
        return out

# ---------------- Public broker surface ----------------

def account_balance()->Dict[str,Any]:
    if BROKER=="tradier":
        with _tradier() as c:
            j=_get(c, f"/v1/accounts/{_acct_id()}/balances")
            b=j.get("balances",{})
            # Tradier doesn't directly give day P/L; leave 0.0 (UI still useful)
            return {
                "equity": float(b.get("equity", 0.0) or 0.0),
                "cash": float(b.get("cash", 0.0) or 0.0),
                "day_pl_dollar": 0.0,
                "day_pl_pct": 0.0
            }
    # fallback
    return {"equity":0,"cash":0,"day_pl_dollar":0,"day_pl_pct":0}

def open_positions()->List[Dict[str,Any]]:
    if BROKER=="tradier":
        with _tradier() as c:
            j=_get(c, f"/v1/accounts/{_acct_id()}/positions")
            raw = j.get("positions",{}).get("position",[])
            if isinstance(raw, dict): raw=[raw]
            positions=[]
            for p in raw or []:
                sym = p.get("symbol")
                qty = float(p.get("quantity",0))
                # Try average from purchase_price or cost_basis / quantity
                avg = None
                if p.get("purchase_price") is not None:
                    avg = float(p.get("purchase_price"))
                elif p.get("cost_basis") is not None and qty:
                    avg = float(p.get("cost_basis"))/abs(qty)
                else:
                    avg = 0.0
                positions.append({"symbol": sym, "qty": qty, "avg_price": float(avg), "last": 0.0, "pl_dollar":0.0, "pl_pct":0.0})
            # enrich with quotes & P/L
            sym_list = [p["symbol"] for p in positions if p["symbol"]]
            qmap = _quotes_map(sym_list)
            for p in positions:
                last = qmap.get(p["symbol"], p["avg_price"])
                p["last"] = float(last)
                if p["avg_price"] and p["qty"]:
                    p["pl_dollar"] = (last - p["avg_price"]) * p["qty"]
                    p["pl_pct"] = ((last / p["avg_price"]) - 1.0) * 100.0
            return positions
    return []

def open_orders()->List[Dict[str,Any]]:
    if BROKER=="tradier":
        with _tradier() as c:
            j=_get(c, f"/v1/accounts/{_acct_id()}/orders")
            arr = j.get("orders",{}).get("order",[])
            if isinstance(arr, dict): arr=[arr]
            from datetime import datetime
            res=[]
            for o in arr or []:
                res.append({
                    "id": str(o.get("id")),
                    "symbol": o.get("symbol",""),
                    "side": (o.get("side","") or "").upper(),
                    "qty": float(o.get("quantity",0) or 0),
                    "type": o.get("type",""),
                    "limit_price": float(o.get("price",0) or 0),
                    "status": o.get("status",""),
                    "ts": datetime.utcnow().isoformat()
                })
            return res
    return []

def place_order(symbol:str, side:str, qty:float, typ:str="market", limit_price:float|None=None)->str:
    """
    Tradier fields: class, symbol, side, quantity, type, duration, price(optional)
    side: buy|sell|sell_short|buy_to_cover ; type: market|limit|stop|stop_limit
    """
    if BROKER!="tradier":
        raise RuntimeError("place_order: only Tradier implemented in MVP")
    with _tradier() as c:
        data = {
            "class": "equity",
            "symbol": symbol.upper(),
            "side": side.lower(),           # "buy" or "sell"
            "quantity": int(qty),           # Tradier expects integer shares for equities
            "type": typ.lower(),            # "market" or "limit"
            "duration": "day"
        }
        if typ.lower() in ("limit","stop","stop_limit") and limit_price:
            data["price"] = float(limit_price)
        j=_post(c, f"/v1/accounts/{_acct_id()}/orders", data=data)
        # Response: {"order":{"id":1234567,...}} in sandbox
        order = j.get("order") or {}
        oid = str(order.get("id") or order.get("order",""))
        return oid or "unknown"

def cancel_order(order_id:str)->bool:
    if BROKER!="tradier":
        return False
    with _tradier() as c:
        _delete(c, f"/v1/accounts/{_acct_id()}/orders/{order_id}")
        return True
