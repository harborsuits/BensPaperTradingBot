from datetime import datetime
from typing import Callable, Optional, Tuple, List
from zoneinfo import ZoneInfo

from trading_bot.core.safety.pause_state import is_paused
from trading_bot.core.safety.context import OrderContext, RiskPolicy, parse_hhmm, ET

# Injectable helpers
EmergencyStopFn = Callable[[], bool]
GetLastPriceFn = Callable[[str], Optional[float]]
IsEarningsBlackoutFn = Callable[[str, datetime], bool]
IsMarketOpenFn = Callable[[datetime], bool]


def default_is_market_open(now_et: datetime) -> bool:
    if now_et.weekday() >= 5:
        return False
    t = now_et.time()
    return (t >= parse_hhmm("09:30")) and (t <= parse_hhmm("16:00"))


def evaluate_order(
    ctx: OrderContext,
    policy: RiskPolicy = RiskPolicy(),
    is_emergency_stop: EmergencyStopFn = lambda: False,
    get_last_price: GetLastPriceFn = lambda _symbol: None,
    is_earnings_blackout: IsEarningsBlackoutFn = lambda _sym, _now: False,
    is_market_open: IsMarketOpenFn = default_is_market_open,
) -> Tuple[bool, List[str]]:
    failures: List[str] = []

    now_et = (ctx.now or datetime.now(tz=ET)).astimezone(ET)

    if is_paused():
        failures.append("entries_paused_due_to_reconcile")

    if is_emergency_stop():
        failures.append("emergency_stop_enabled")

    if ctx.day_pnl_pct is not None and ctx.day_pnl_pct <= -policy.daily_loss_stop:
        failures.append(f"daily_loss_limit_reached_{ctx.day_pnl_pct:.4f}")

    if ctx.max_drawdown_pct is not None and ctx.max_drawdown_pct <= -policy.max_dd_stop:
        failures.append(f"max_drawdown_limit_reached_{ctx.max_drawdown_pct:.4f}")

    if not is_market_open(now_et):
        failures.append("market_closed")
    start_t, end_t = map(parse_hhmm, policy.trade_window)
    if not (start_t <= now_et.time() <= end_t):
        failures.append("outside_trade_window")

    if policy.enforce_earnings_blackout and is_earnings_blackout(ctx.symbol, now_et):
        failures.append("earnings_blackout")

    px = ctx.price
    if px is None:
        px = get_last_price(ctx.symbol)
    if px is None:
        failures.append("price_unknown")
    elif px < policy.price_floor:
        failures.append(f"price_below_floor_{px:.2f}_<_{policy.price_floor:.2f}")

    return (len(failures) == 0), failures


