# trading_bot/risk/cost_model.py
from dataclasses import dataclass
from math import sqrt
from typing import Literal, Tuple, Dict

Asset = Literal["equity","etf","option"]

@dataclass
class CostModelConfig:
    per_share_fee: float = 0.0          # e.g., 0.005
    per_contract_fee: float = 0.65      # options fee per contract
    sec_taf_bps: float = 0.18           # sell-side only (bps of notional)
    base_impact_bps_equity: float = 10  # 7–15 bps typical
    base_impact_bps_etf: float = 5      # 3–7 bps typical
    max_impact_bps: float = 60

def half_spread_bps(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    return ((ask - bid) / mid) * 1e4 / 2.0

def impact_bps(asset: Asset, vix: float, participation_pct: float, cfg: CostModelConfig) -> float:
    base = cfg.base_impact_bps_etf if asset == "etf" else cfg.base_impact_bps_equity
    # Smooth scaling: higher vol & larger participation => more impact
    scale = (max(vix, 10.0) / 20.0)**0.5 * (max(participation_pct, 1e-6) / 0.5)**0.5
    return min(base * scale, cfg.max_impact_bps)

def modeled_fill(bid: float, ask: float, side: Literal["buy","sell"]) -> float:
    mid = (bid + ask) / 2.0
    # Marketable limit at mid ± 0.5 * spread
    price = ask if side == "buy" else bid
    return price  # price you actually expect to pay

def apply_costs(
    asset: Asset,
    side: Literal["buy","sell"],
    qty: int,
    bid: float,
    ask: float,
    vix: float,
    one_min_notional: float,  # traded notional last 1m for the symbol
    order_notional: float,    # your order notional
    cfg: CostModelConfig = CostModelConfig()
) -> Dict:
    mid = (bid + ask) / 2.0
    hsb = half_spread_bps(bid, ask)           # half spread (bps)
    participation_pct = 100.0 * order_notional / max(one_min_notional, 1e-6)
    imp_bps = impact_bps(asset, vix, participation_pct, cfg)
    modeled_slip_bps = hsb + imp_bps

    fill = modeled_fill(bid, ask, side)
    # Fees
    fees = 0.0
    if asset in ("equity","etf"):
        fees += cfg.per_share_fee * qty
        if side == "sell":
            fees += (cfg.sec_taf_bps / 1e4) * order_notional
    else:  # option
        fees += cfg.per_contract_fee * qty

    return {
        "mid": mid,
        "fill": fill,
        "half_spread_bps": hsb,
        "impact_bps": imp_bps,
        "modeled_slippage_bps": modeled_slip_bps,
        "modeled_total_cost_cash": fees + (modeled_slip_bps / 1e4) * order_notional
    }
