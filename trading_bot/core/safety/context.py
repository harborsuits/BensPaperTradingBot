from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


@dataclass
class RiskPolicy:
    daily_loss_stop: float = 0.03
    max_dd_stop: float = 0.08
    price_floor: float = 5.0
    trade_window: Tuple[str, str] = ("10:10", "15:30")
    enforce_earnings_blackout: bool = True


@dataclass
class OrderContext:
    symbol: str
    side: str
    qty: int
    strategy: str = "default"
    price: Optional[float] = None
    now: datetime = None
    day_pnl_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None


def parse_hhmm(s: str) -> time:
    h, m = map(int, s.split(":"))
    return time(hour=h, minute=m)


