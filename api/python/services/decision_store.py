from typing import List, Dict, Any


def list_recent_decisions(limit: int = 50) -> List[Dict[str, Any]]:
    """Return last-N decisions as a list of dicts.
    Expected keys: id, timestamp (ISO), symbol, type, reason(optional), note(optional)
    TODO: Replace stub with DB/journal lookup.
    """
    items: List[Dict[str, Any]] = []
    return items[:limit]


