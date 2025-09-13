"""
Trading playbooks for different instrument types.

This module implements the order generation logic for each instrument type,
following the policy constraints and using appropriate order types and parameters.
"""

from typing import Dict, List, TypedDict, Literal, Optional, Any

from trading_bot.policy.types import Policy, Instrument


class OrderContext(TypedDict):
    """Context for order generation."""
    px: float  # Current price
    vol: float  # Volume or liquidity
    size_budget_usd: float  # Size budget in USD


class Order(TypedDict, total=False):
    """Trading order with parameters."""
    type: Literal["LIMIT", "IOC", "MARKET"]
    side: Literal["BUY", "SELL"]
    qty: float
    limit: Optional[float]
    ttl_ms: Optional[int]
    slippage_bps_cap: Optional[float]
    meta: Optional[Dict[str, Any]]


def build_orders(instrument: str, policy: Policy, ctx: OrderContext) -> List[Order]:
    """
    Build orders for a given instrument type following the appropriate playbook.
    
    Args:
        instrument: The instrument type (ETF, CRYPTO, OPTIONS)
        policy: The current trading policy
        ctx: Context for order generation including price and size budget
        
    Returns:
        List of orders to execute
    """
    if instrument == Instrument.ETF.value:
        # ETF Trend playbook: Vol-budget sizing with conservative limits
        if not policy["toggles"]["enable_etf_trend"]:
            return []
            
        qty = max(1, int(ctx["size_budget_usd"] / ctx["px"]))
        return [{
            "type": "LIMIT",
            "side": "BUY",
            "qty": qty,
            "limit": ctx["px"] * (1 + 0.0005),  # 0.5 bps above current price
            "meta": {"playbook": "ETFTrend"}
        }]
    
    elif instrument == Instrument.CRYPTO.value:
        # Crypto Burst playbook: Quick IOC orders with tight time constraints
        if not policy["toggles"]["enable_crypto_burst"]:
            return []
            
        qty = max(1, int(ctx["size_budget_usd"] / ctx["px"]))
        max_slippage_bps = policy["risk"]["crypto"]["max_slippage_bps"]
        
        return [{
            "type": "IOC",
            "side": "BUY",
            "qty": qty,
            "limit": ctx["px"] * (1 + max_slippage_bps / 10000),
            "ttl_ms": policy["risk"]["crypto"]["max_dwell_ms"],
            "slippage_bps_cap": max_slippage_bps,
            "meta": {"playbook": "CryptoBurst"}
        }]
    
    elif instrument == Instrument.OPTIONS.value:
        # Options Event playbook: Defined-risk spreads
        if not policy["toggles"]["enable_options_events"]:
            return []
            
        # Example: vertical debit spread (simplified)
        # In a real implementation, this would include strike selection logic
        return [{
            "type": "LIMIT",
            "side": "BUY",
            "qty": 1,
            "limit": 1.00,  # Example price
            "meta": {
                "playbook": "EventSpread",
                "defined_risk": True,
                "strategy": "vertical_debit",
                "width": 5  # Example strike width
            }
        }]
    
    # Unknown instrument type
    return []
