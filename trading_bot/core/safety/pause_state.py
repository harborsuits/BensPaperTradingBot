_PAUSED: bool = False

def set_paused(paused: bool) -> None:
    global _PAUSED
    _PAUSED = bool(paused)

def is_paused() -> bool:
    return _PAUSED


