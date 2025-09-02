from typing import List, Dict, Literal, Any

Mode = Literal["paper", "live"]


def get_cash(mode: Mode) -> float:
    """Return current cash for the given account mode.
    TODO: Replace stub with real broker/journal call.
    """
    return 100_000.0 if mode == "paper" else 25_000.0


def _norm_pos(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize arbitrary broker fields -> UI schema.
    Output keys: symbol, qty, avg_price, mkt_price, unrealized_pl, pl_pct
    """
    qty = float(rec.get("qty") or rec.get("quantity") or 0)
    avg = float(rec.get("avg_price") or rec.get("average_price") or 0)
    last = float(
        rec.get("mkt_price")
        or rec.get("last_price")
        or rec.get("market_price")
        or 0
    )
    upl = (last - avg) * qty if qty and avg and last else float(rec.get("unrealized_pl") or 0)
    pl_pct = (upl / (avg * qty) * 100) if qty and avg else 0.0
    return {
        "symbol": rec.get("symbol") or rec.get("ticker") or "UNK",
        "qty": qty,
        "avg_price": avg,
        "mkt_price": last,
        "unrealized_pl": upl,
        "pl_pct": pl_pct,
    }


def get_positions(mode: Mode) -> List[Dict[str, Any]]:
    """Return normalized open positions list for the given mode.
    TODO: Replace stub with real holdings fetch.
    """
    raw: List[Dict[str, Any]] = [
        # Example (delete when wired):
        # {"symbol": "AAPL", "qty": 10, "avg_price": 190.5, "last_price": 191.2}
    ]
    return [_norm_pos(p) for p in raw]


