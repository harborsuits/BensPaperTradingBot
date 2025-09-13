"""
Opportunity scoring engine.

This module implements the scoring algorithm for trading opportunities,
combining alpha, regime alignment, sentiment, costs, and risk factors.
"""

from typing import Dict, Tuple, TypedDict

from trading_bot.policy.types import Policy


class ScoringInputs(TypedDict):
    """Inputs for the opportunity scoring algorithm."""
    alpha: float
    regime_align: float
    sentiment_boost: float
    est_cost_bps: float
    risk_penalty: float


class ScoringResult(TypedDict):
    """Result of the opportunity scoring algorithm."""
    score: float
    reasons: list[str]


def clamp_01(value: float) -> float:
    """Clamp a value between 0 and 1."""
    return max(0.0, min(1.0, value))


def score_opportunity(policy: Policy, inputs: ScoringInputs) -> ScoringResult:
    """
    Score a trading opportunity based on multiple factors.
    
    Args:
        policy: The current trading policy with scoring weights
        inputs: Input factors for scoring
        
    Returns:
        Dictionary with final score and explanatory reasons
    """
    # Normalize inputs to 0-1 range
    alpha = clamp_01(inputs["alpha"])
    regime = clamp_01((inputs["regime_align"] + 1) / 2)  # Convert -1..1 to 0..1
    sentiment = clamp_01(inputs["sentiment_boost"])
    cost = clamp_01(inputs["est_cost_bps"] / 1000)  # Normalize bps to 0-1
    risk = clamp_01(inputs["risk_penalty"])
    
    # Calculate weighted score
    score = (
        policy["weights"]["alpha"] * alpha +
        policy["weights"]["regime"] * regime +
        policy["weights"]["sentiment"] * sentiment -
        policy["weights"]["cost"] * cost -
        policy["weights"]["risk"] * risk
    )
    
    # Generate explanation
    reasons = [
        f"alpha={alpha:.2f}",
        f"regime={regime:.2f}",
        f"sent={sentiment:.2f}",
        f"cost={cost:.2f}",
        f"risk={risk:.2f}"
    ]
    
    return {
        "score": score,
        "reasons": reasons
    }
