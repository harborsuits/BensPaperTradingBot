# trading_bot/execution/gates.py
from dataclasses import dataclass
from typing import Literal, Tuple

Asset = Literal["equity","etf","option"]

@dataclass
class GateConfig:
    max_spread_bps_equity: float = 25
    max_spread_bps_etf: float = 15
    max_quote_age_ms_equity: int = 1000
    max_quote_age_ms_affordable: int = 500   # e.g. <$20 names
    max_participation_pct: float = 0.5       # of 1-min notional

def check_gates(asset: Asset,
                spread_bps: float,
                quote_age_ms: int,
                order_notional: float,
                one_min_notional: float,
                is_affordable: bool,
                cfg: GateConfig = GateConfig()) -> Tuple[bool, list]:
    reasons = []
    max_spread = cfg.max_spread_bps_etf if asset=="etf" else cfg.max_spread_bps_equity
    if spread_bps > max_spread: reasons.append(f"spread_bps>{max_spread}")
    max_age = cfg.max_quote_age_ms_affordable if is_affordable else cfg.max_quote_age_ms_equity
    if quote_age_ms > max_age: reasons.append(f"quote_age>{max_age}ms")
    part_pct = 100.0 * order_notional / max(one_min_notional, 1e-6)
    if part_pct > cfg.max_participation_pct: reasons.append(f"participation>{cfg.max_participation_pct}%")
    return (len(reasons)==0, reasons)
