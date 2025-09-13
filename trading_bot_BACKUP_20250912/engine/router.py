"""
Decision router for trading opportunities.

This module implements the core decision-making logic that ranks and routes
trading opportunities based on policy, compliance gates, and scoring.
"""

import time
from typing import Dict, List, TypedDict, Optional, Any

from trading_bot.policy.types import Policy
from trading_bot.engine.compliance import compliance_gates, OpportunityMetadata
from trading_bot.engine.scoring import score_opportunity, ScoringInputs
from trading_bot.engine.playbooks import build_orders, OrderContext, Order
from trading_bot.engine.audit import log_decision, DecisionAudit


class Opportunity(TypedDict):
    """Trading opportunity with all required data for decision-making."""
    id: str
    instrument: str
    symbol: str
    ts: int  # Timestamp in milliseconds
    alpha: float
    regime_align: float
    sentiment_boost: float
    est_cost_bps: float
    risk_penalty: float
    meta: OpportunityMetadata
    ctx: OrderContext


class RoutedOpportunity(TypedDict):
    """Opportunity that passed all gates and has been routed for execution."""
    id: str
    instrument: str
    symbol: str
    score: float
    orders: List[Order]


def rank_and_route(
    opportunities: List[Opportunity],
    policy: Policy,
    now: Optional[int] = None
) -> List[RoutedOpportunity]:
    """
    Rank and route trading opportunities based on policy and compliance gates.
    
    Args:
        opportunities: List of trading opportunities
        policy: The current trading policy
        now: Current timestamp in milliseconds (defaults to current time)
        
    Returns:
        List of routed opportunities sorted by score
    """
    if now is None:
        now = int(time.time() * 1000)
    
    result = []
    
    for opp in opportunities:
        reasons: List[str] = []
        
        # Skip stale opportunities
        if now - opp["ts"] > policy["stale_after_ms"]:
            log_decision({
                "id": opp["id"],
                "ts": now,
                "instrument": opp["instrument"],
                "symbol": opp["symbol"],
                "action": "SKIP",
                "reasons": ["stale"],
                "policy_version": policy["version"],
                "score": None,
                "gates_failed": None
            })
            continue
        
        # Check compliance gates
        gate_fails = compliance_gates(opp["instrument"], opp["meta"], policy)
        if gate_fails:
            log_decision({
                "id": opp["id"],
                "ts": now,
                "instrument": opp["instrument"],
                "symbol": opp["symbol"],
                "action": "SKIP",
                "reasons": ["gates_failed"],
                "policy_version": policy["version"],
                "score": None,
                "gates_failed": [g["code"] for g in gate_fails]
            })
            continue
        
        # Score the opportunity
        scoring_inputs: ScoringInputs = {
            "alpha": opp["alpha"],
            "regime_align": opp["regime_align"],
            "sentiment_boost": opp["sentiment_boost"],
            "est_cost_bps": opp["est_cost_bps"],
            "risk_penalty": opp["risk_penalty"]
        }
        
        score_result = score_opportunity(policy, scoring_inputs)
        score = score_result["score"]
        reasons.extend(score_result["reasons"])
        
        # Skip opportunities with non-positive scores
        if score <= 0:
            log_decision({
                "id": opp["id"],
                "ts": now,
                "instrument": opp["instrument"],
                "symbol": opp["symbol"],
                "action": "SKIP",
                "reasons": [*reasons, "low_score"],
                "policy_version": policy["version"],
                "score": score,
                "gates_failed": None
            })
            continue
        
        # Build orders using the appropriate playbook
        orders = build_orders(opp["instrument"], policy, opp["ctx"])
        
        # Skip if no orders were generated (e.g., disabled playbook)
        if not orders:
            log_decision({
                "id": opp["id"],
                "ts": now,
                "instrument": opp["instrument"],
                "symbol": opp["symbol"],
                "action": "SKIP",
                "reasons": [*reasons, "no_orders_generated"],
                "policy_version": policy["version"],
                "score": score,
                "gates_failed": None
            })
            continue
        
        # Log successful decision
        log_decision({
            "id": opp["id"],
            "ts": now,
            "instrument": opp["instrument"],
            "symbol": opp["symbol"],
            "action": "OPEN",
            "reasons": reasons,
            "policy_version": policy["version"],
            "score": score,
            "gates_failed": None
        })
        
        # Add to result
        result.append({
            "id": opp["id"],
            "instrument": opp["instrument"],
            "symbol": opp["symbol"],
            "score": score,
            "orders": orders
        })
    
    # Sort by score (descending)
    return sorted(result, key=lambda x: x["score"], reverse=True)
