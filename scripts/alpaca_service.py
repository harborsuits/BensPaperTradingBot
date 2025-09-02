import os
import httpx
import datetime
import logging
from typing import Dict, List, Optional, Union, Any

# Configure logging
logger = logging.getLogger(__name__)

# Environment variables for Alpaca API
ALPACA_KEY_ID = os.getenv("ALPACA_KEY_ID", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_IS_PAPER = os.getenv("ALPACA_IS_PAPER", "true").lower() == "true"

# API endpoints
DATA_BASE = os.getenv("ALPACA_DATA_BASE", "https://data.alpaca.markets/v2")
PAPER_BASE = os.getenv("ALPACA_PAPER_BASE", "https://paper-api.alpaca.markets/v2")
LIVE_BASE = os.getenv("ALPACA_LIVE_BASE", "https://api.alpaca.markets/v2")

# Use paper or live trading based on environment
TRADE_BASE = PAPER_BASE if ALPACA_IS_PAPER else LIVE_BASE

# Headers for authentication
HEADERS = {
    "APCA-API-KEY-ID": ALPACA_KEY_ID,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

# Circuit breaker to prevent API hammering
_circuit_breaker = {
    "state": "CLOSED",  # CLOSED, HALF-OPEN, OPEN
    "failures": 0,
    "last_failure": None,
    "threshold": 5,
    "reset_after": 300,  # seconds
}

def _require_keys():
    """Verify API keys are available"""
    if not (ALPACA_KEY_ID and ALPACA_SECRET_KEY):
        raise ValueError("Missing Alpaca API keys. Set ALPACA_KEY_ID and ALPACA_SECRET_KEY.")

def _check_circuit():
    """Check if circuit breaker is open"""
    if _circuit_breaker["state"] == "OPEN":
        # Check if we should try again
        if _circuit_breaker["last_failure"] is None:
            return True
        
        elapsed = (datetime.datetime.now() - _circuit_breaker["last_failure"]).total_seconds()
        if elapsed > _circuit_breaker["reset_after"]:
            logger.info("Circuit half-open, allowing test request")
            _circuit_breaker["state"] = "HALF-OPEN"
            return True
        return False
    return True

def _record_failure():
    """Record an API failure"""
    _circuit_breaker["failures"] += 1
    _circuit_breaker["last_failure"] = datetime.datetime.now()
    
    if _circuit_breaker["failures"] >= _circuit_breaker["threshold"]:
        logger.warning("Circuit breaker OPEN - too many API failures")
        _circuit_breaker["state"] = "OPEN"

def _record_success():
    """Record a successful API call"""
    if _circuit_breaker["state"] == "HALF-OPEN":
        logger.info("Circuit CLOSED - API recovered")
        _circuit_breaker["state"] = "CLOSED"
    
    _circuit_breaker["failures"] = 0

def reset_circuit_breaker():
    """Reset the circuit breaker (for admin use)"""
    _circuit_breaker["state"] = "CLOSED"
    _circuit_breaker["failures"] = 0
    _circuit_breaker["last_failure"] = None
    return {"reset": True, "ts": datetime.datetime.utcnow().isoformat()}

def get_market_status() -> Dict[str, Any]:
    """Get current market status from Alpaca"""
    _require_keys()
    
    if not _check_circuit():
        return {"error": "Circuit breaker open, API calls temporarily disabled"}
    
    try:
        # Use clock endpoint
        with httpx.Client(timeout=8) as client:
            r = client.get(f"{TRADE_BASE}/clock", headers=HEADERS)
        
        if r.status_code != 200:
            _record_failure()
            return {"error": f"status {r.status_code}"}
        
        _record_success()
        j = r.json()
        return {
            "is_open": j.get("is_open"), 
            "next_open": j.get("next_open"), 
            "next_close": j.get("next_close")
        }
    except Exception as e:
        _record_failure()
        logger.error(f"Error getting market status: {str(e)}")
        return {"error": str(e)}

def get_service_health() -> Dict[str, Any]:
    """Get health status of the Alpaca service"""
    try:
        c = get_market_status()
        circuit_status = _circuit_breaker["state"]
        
        return {
            "ok": "error" not in c,
            "clock": c,
            "circuit_state": circuit_status,
            "failures": _circuit_breaker["failures"],
            "last_failure": _circuit_breaker["last_failure"].isoformat() if _circuit_breaker["last_failure"] else None,
            "is_paper": ALPACA_IS_PAPER
        }
    except Exception as e:
        return {
            "ok": False, 
            "error": str(e),
            "circuit_state": _circuit_breaker["state"]
        }

def get_live_price(symbol: str) -> float:
    """Get the latest price for a symbol"""
    _require_keys()
    
    if not _check_circuit():
        raise ValueError("Circuit breaker open, API calls temporarily disabled")
    
    symbol = symbol.upper()
    try:
        with httpx.Client(timeout=8) as client:
            r = client.get(f"{DATA_BASE}/stocks/{symbol}/quotes/latest", headers=HEADERS)
        
        if r.status_code == 200:
            _record_success()
            q = r.json().get("quote") or {}
            
            # prefer mid if possible
            ap = q.get("ap")
            bp = q.get("bp")
            if ap and bp:
                return (ap + bp) / 2
            
            # fallback to ask or bid
            if ap:
                return ap
            if bp:
                return bp
            
            raise ValueError(f"No price data available for {symbol}")
            
        if r.status_code == 404:
            raise ValueError(f"Symbol not found: {symbol}")
        
        _record_failure()
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        _record_failure()
        raise ValueError(f"HTTP error: {e.response.status_code}")
    except httpx.RequestError as e:
        _record_failure()
        raise ValueError(f"Request error: {str(e)}")
    except Exception as e:
        _record_failure()
        raise ValueError(f"Error fetching price: {str(e)}")

def get_batch_quotes(symbols: List[str]) -> Dict[str, Dict]:
    """Get quotes for multiple symbols in one request"""
    _require_keys()
    
    if not _check_circuit():
        raise ValueError("Circuit breaker open, API calls temporarily disabled")
    
    if not symbols:
        return {}
    
    # Uppercase all symbols and deduplicate
    unique_symbols = list(set(s.upper() for s in symbols if s))
    
    # Split into chunks of 10 symbols (Alpaca limit)
    chunk_size = 10
    results = {}
    
    for i in range(0, len(unique_symbols), chunk_size):
        chunk = unique_symbols[i:i+chunk_size]
        symbols_str = ",".join(chunk)
        
        try:
            with httpx.Client(timeout=10) as client:
                r = client.get(
                    f"{DATA_BASE}/stocks/quotes/latest",
                    params={"symbols": symbols_str},
                    headers=HEADERS
                )
            
            if r.status_code == 200:
                _record_success()
                data = r.json()
                quotes = data.get("quotes", {})
                
                # Process each quote
                for symbol, quote in quotes.items():
                    results[symbol] = {
                        "symbol": symbol,
                        "quote": quote,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
            else:
                _record_failure()
                logger.warning(f"Failed to fetch quotes for {symbols_str}: {r.status_code}")
        except Exception as e:
            _record_failure()
            logger.error(f"Error fetching quotes for {symbols_str}: {str(e)}")
    
    # Add empty entries for symbols we couldn't get
    for symbol in unique_symbols:
        if symbol not in results:
            results[symbol] = {
                "symbol": symbol,
                "quote": None,
                "error": "Failed to fetch quote"
            }
    
    return results

def get_account() -> Dict[str, Any]:
    """Get account information"""
    _require_keys()
    
    if not _check_circuit():
        raise ValueError("Circuit breaker open, API calls temporarily disabled")
    
    try:
        with httpx.Client(timeout=8) as client:
            r = client.get(f"{TRADE_BASE}/account", headers=HEADERS)
        
        if r.status_code == 200:
            _record_success()
            return r.json()
        
        _record_failure()
        r.raise_for_status()
    except Exception as e:
        _record_failure()
        raise ValueError(f"Error fetching account: {str(e)}")

def get_positions() -> List[Dict[str, Any]]:
    """Get current positions"""
    _require_keys()
    
    if not _check_circuit():
        raise ValueError("Circuit breaker open, API calls temporarily disabled")
    
    try:
        with httpx.Client(timeout=8) as client:
            r = client.get(f"{TRADE_BASE}/positions", headers=HEADERS)
        
        if r.status_code == 200:
            _record_success()
            return r.json()
        
        if r.status_code == 404:
            return []
        
        _record_failure()
        r.raise_for_status()
    except Exception as e:
        _record_failure()
        raise ValueError(f"Error fetching positions: {str(e)}")

def get_orders(status: str = "open", limit: int = 50) -> List[Dict[str, Any]]:
    """Get orders with optional filters"""
    _require_keys()
    
    if not _check_circuit():
        raise ValueError("Circuit breaker open, API calls temporarily disabled")
    
    params = {"status": status, "limit": limit}
    
    try:
        with httpx.Client(timeout=8) as client:
            r = client.get(f"{TRADE_BASE}/orders", headers=HEADERS, params=params)
        
        if r.status_code == 200:
            _record_success()
            return r.json()
        
        if r.status_code == 404:
            return []
        
        _record_failure()
        r.raise_for_status()
    except Exception as e:
        _record_failure()
        raise ValueError(f"Error fetching orders: {str(e)}")

def get_order(order_id: str) -> Dict[str, Any]:
    """Get a specific order by ID"""
    _require_keys()
    
    if not _check_circuit():
        raise ValueError("Circuit breaker open, API calls temporarily disabled")
    
    try:
        with httpx.Client(timeout=8) as client:
            r = client.get(f"{TRADE_BASE}/orders/{order_id}", headers=HEADERS)
        
        if r.status_code == 200:
            _record_success()
            return r.json()
        
        _record_failure()
        r.raise_for_status()
    except Exception as e:
        _record_failure()
        raise ValueError(f"Error fetching order {order_id}: {str(e)}")

def submit_order(
    symbol: str,
    side: str,
    qty: float,
    type: str = "market",
    time_in_force: str = "day",
    limit_price: Optional[float] = None
) -> Dict[str, Any]:
    """Submit a new order"""
    _require_keys()
    
    if not _check_circuit():
        raise ValueError("Circuit breaker open, API calls temporarily disabled")
    
    payload = {
        "symbol": symbol.upper(),
        "side": side,               # "buy" | "sell"
        "qty": str(qty),            # API expects string
        "type": type,               # "market" | "limit"
        "time_in_force": time_in_force
    }
    
    if type == "limit" and limit_price is not None:
        payload["limit_price"] = str(float(limit_price))
    
    try:
        with httpx.Client(timeout=10) as client:
            r = client.post(f"{TRADE_BASE}/orders", headers=HEADERS, json=payload)
        
        if r.status_code in (200, 201):
            _record_success()
            return r.json()
        
        _record_failure()
        # Provide readable error upstream
        raise ValueError(f"Alpaca order error {r.status_code}: {r.text}")
    except Exception as e:
        _record_failure()
        raise ValueError(f"Error submitting order: {str(e)}")

def get_bars(
    symbol: str,
    timeframe: str = "1Day",
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """Get historical price bars"""
    _require_keys()
    
    if not _check_circuit():
        raise ValueError("Circuit breaker open, API calls temporarily disabled")
    
    params = {
        "timeframe": timeframe,
        "limit": limit
    }
    
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    
    try:
        with httpx.Client(timeout=10) as client:
            r = client.get(
                f"{DATA_BASE}/stocks/{symbol}/bars", 
                headers=HEADERS,
                params=params
            )
        
        if r.status_code == 200:
            _record_success()
            return r.json()
        
        _record_failure()
        r.raise_for_status()
    except Exception as e:
        _record_failure()
        raise ValueError(f"Error fetching bars for {symbol}: {str(e)}")

# --- HEALTH SURFACE ---
from datetime import datetime, timezone
from typing import Dict, Any

def health(symbol: str = "SPY") -> Dict[str, Any]:
    """
    Returns reachability + basic freshness + paper/live flag.
    Never exposes secrets. Safe for /health surfaces.
    """
    info: Dict[str, Any] = {"service": "alpaca", "ok": False, "paper": bool(ALPACA_IS_PAPER)}
    try:
        acct = get_account()
        # Try price
        try:
            _ = get_live_price(symbol)
        except Exception:
            pass
        # Try bars
        bars_count = 0
        try:
            bars_resp = get_bars(symbol, timeframe="1Min", limit=3) or {}
            bars = bars_resp.get("bars") or bars_resp.get("bars", [])
            bars_count = len(bars or [])
        except Exception:
            bars_count = 0

        info.update({
            "ok": True,
            "equity_usd": float(acct.get("equity") or 0.0),
            "rate_limit": None,
            "quote_age_ms": None,
            "bars": bars_count,
        })
    except Exception as e:
        info.update({"ok": False, "error": str(e)})
    return info
