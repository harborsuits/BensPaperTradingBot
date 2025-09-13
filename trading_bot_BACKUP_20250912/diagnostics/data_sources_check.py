import asyncio, json, sys
from typing import Dict, Any

from alpaca_service import health as alpaca_health
from trading_bot.services import news_service
from trading_bot.api import websocket_manager

GREEN, RED, YELLOW, RESET = "\x1b[32m", "\x1b[31m", "\x1b[33m", "\x1b[0m"

async def check_all() -> Dict[str, Any]:
    # mix sync/async
    a = alpaca_health("SPY")
    n = await news_service.health()
    w = websocket_manager.health()
    return {"alpaca": a, "news": n, "websocket": w}

def badge(ok: bool) -> str:
    return f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"

def soft_warns(payload: Dict[str, Any]) -> None:
    a = payload.get("alpaca", {})
    if a.get("ok") and a.get("quote_age_ms") is not None and a["quote_age_ms"] > 60_000:
        print(f"{YELLOW}WARN{RESET} Alpaca quote stale: {a['quote_age_ms']}ms")
    w = payload.get("websocket", {})
    if w.get("ok") and w.get("msg_per_min") == 0:
        print(f"{YELLOW}WARN{RESET} WS connected but no flow")

async def main() -> int:
    r = await check_all()
    a, n, w = r.get("alpaca", {}), r.get("news", {}), r.get("websocket", {})
    print(f"Alpaca:    {badge(a.get('ok'))} paper={a.get('paper')} equity=${a.get('equity_usd')}")
    print(f"News:      {badge(n.get('ok'))} up={n.get('summary',{}).get('up')}/{n.get('summary',{}).get('total')}")
    print(f"WebSocket: {badge(w.get('ok'))} connected={w.get('connected')} subs={len(w.get('subs',[]))}")
    soft_warns(r)
    print("\nJSON:\n" + json.dumps(r, indent=2))
    return 0 if all([a.get("ok"), n.get("ok"), w.get("ok")]) else 2

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))


