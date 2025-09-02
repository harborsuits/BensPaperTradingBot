import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _journal_dir() -> Path:
    base = os.environ.get("JOURNAL_DIR")
    if not base:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/journal"))
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _file_path(kind: str) -> Path:
    day = datetime.utcnow().strftime("%Y-%m-%d")
    return _journal_dir() / f"{day}_{kind}.jsonl"


def append(kind: str, obj: Dict[str, Any]) -> None:
    path = _file_path(kind)
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, default=str) + "\n")
    except Exception:
        # journaling should never break the main flow
        pass


def read_recent(kind: str, limit: int = 100) -> List[Dict[str, Any]]:
    path = _file_path(kind)
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        tail = lines[-limit:]
        result: List[Dict[str, Any]] = []
        for line in tail:
            try:
                result.append(json.loads(line))
            except Exception:
                continue
        return result[::-1]
    except Exception:
        return []


def read_latest(kind: str) -> Optional[Dict[str, Any]]:
    items = read_recent(kind, limit=1)
    return items[0] if items else None


def log_trade(trade: Dict[str, Any]) -> None:
    if "timestamp" not in trade:
        trade["timestamp"] = datetime.utcnow().isoformat()
    append("trades", trade)


def log_portfolio(snapshot: Dict[str, Any]) -> None:
    if "timestamp" not in snapshot:
        snapshot["timestamp"] = datetime.utcnow().isoformat()
    append("portfolio", snapshot)


