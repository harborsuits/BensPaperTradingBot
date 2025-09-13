import os
import json
import csv
from datetime import datetime
from typing import Optional, List, Tuple


def _parse_iso(dt_str: str) -> Optional[datetime]:
    try:
        # Accept both with/without timezone; treat naive as UTC
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return None


def _load_csv(path: str) -> List[Tuple[str, datetime, datetime]]:
    rows: List[Tuple[str, datetime, datetime]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            sym = (r.get("symbol") or "").strip().upper()
            start = _parse_iso((r.get("start_iso") or "").strip())
            end = _parse_iso((r.get("end_iso") or "").strip())
            if sym and start and end:
                rows.append((sym, start, end))
    return rows


def _load_json(path: str) -> List[Tuple[str, datetime, datetime]]:
    rows: List[Tuple[str, datetime, datetime]] = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # support list of objects {symbol,start_iso,end_iso}
        for r in data or []:
            sym = (r.get("symbol") or "").strip().upper()
            start = _parse_iso((r.get("start_iso") or "").strip())
            end = _parse_iso((r.get("end_iso") or "").strip())
            if sym and start and end:
                rows.append((sym, start, end))
    return rows


def _load(path: str) -> List[Tuple[str, datetime, datetime]]:
    if not os.path.exists(path):
        return []
    try:
        if path.endswith(".csv"):
            return _load_csv(path)
        return _load_json(path)
    except Exception:
        return []


def is_earnings_blackout(symbol: str, now_utc: datetime, path: Optional[str] = None) -> bool:
    """Return True if symbol is in a blackout window at `now_utc`.
    Path can be CSV or JSON with fields: symbol,start_iso,end_iso (ISO8601).
    """
    sym = (symbol or "").upper()
    file_path = path or os.environ.get("EARNINGS_BLACKOUT_FILE", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../config/earnings_blackout.csv")))
    windows = _load(file_path)
    for s, start, end in windows:
        if s == sym and start <= now_utc <= end:
            return True
    return False


