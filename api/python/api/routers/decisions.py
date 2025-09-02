"""
API router for decision engine endpoints.

This module provides FastAPI endpoints for interacting with the decision engine,
including submitting opportunities, retrieving decisions, and managing policy.
"""

import time
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, Body, Query

from trading_bot.policy.types import Policy, Regime
from trading_bot.policy.service import PolicyService
from trading_bot.engine.decision_engine import DecisionEngine
from trading_bot.engine.audit import recent_decisions, get_decision_stats
from trading_bot.engine.router import RoutedOpportunity


# Initialize services
policy_service = PolicyService()
decision_engine = DecisionEngine(policy_service)

# Create router
router = APIRouter(prefix="/decisions", tags=["decisions"])


@router.post("/opportunities")
async def process_opportunities(
    opportunities: List[Dict[str, Any]] = Body(...),
    regime: Optional[str] = Query(None, description="Current market regime"),
    timestamp: Optional[int] = Query(None, description="Timestamp in milliseconds")
) -> List[RoutedOpportunity]:
    """
    Process a batch of trading opportunities.
    
    Args:
        opportunities: List of trading opportunities
        regime: Current market regime (optional)
        timestamp: Current timestamp in milliseconds (optional)
        
    Returns:
        List of routed opportunities
    """
    # Convert regime string to enum if provided
    regime_enum = None
    if regime:
        try:
            regime_enum = Regime(regime)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid regime: {regime}")
    
    # Process opportunities
    try:
        result = decision_engine.process_opportunities(
            opportunities,
            current_regime=regime_enum,
            now=timestamp
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing opportunities: {str(e)}")


@router.get("/recent")
async def get_recent_decisions(
    limit: int = Query(50, ge=1, le=500, description="Maximum number of decisions to return")
) -> List[Dict[str, Any]]:
    """
    Get recent trading decisions.
    
    Args:
        limit: Maximum number of decisions to return
        
    Returns:
        List of recent decision audit records
    """
    return recent_decisions(limit)


@router.get("/stats")
async def get_decision_statistics(
    time_window_ms: int = Query(
        24 * 60 * 60 * 1000,
        description="Time window in milliseconds"
    )
) -> Dict[str, Any]:
    """
    Get statistics about recent trading decisions.
    
    Args:
        time_window_ms: Time window in milliseconds
        
    Returns:
        Dictionary with decision statistics
    """
    return get_decision_stats(time_window_ms)


@router.get("/engine/status")
async def get_engine_status() -> Dict[str, Any]:
    """
    Get the current status of the decision engine.
    
    Returns:
        Dictionary with engine status
    """
    return decision_engine.get_status()


@router.post("/engine/enable")
async def set_engine_enabled(
    enabled: bool = Body(..., embed=True)
) -> Dict[str, Any]:
    """
    Enable or disable the decision engine.
    
    Args:
        enabled: Whether the engine should be enabled
        
    Returns:
        Updated engine status
    """
    decision_engine.set_enabled(enabled)
    return decision_engine.get_status()


@router.post("/engine/update-loss")
async def update_daily_loss(
    loss_pct: float = Body(..., embed=True)
) -> Dict[str, Any]:
    """
    Update the current daily loss percentage.
    
    Args:
        loss_pct: Daily loss as a percentage of portfolio
        
    Returns:
        Updated engine status
    """
    decision_engine.update_daily_loss(loss_pct)
    return decision_engine.get_status()


@router.get("/policy")
async def get_policy() -> Policy:
    """
    Get the current trading policy.
    
    Returns:
        The current policy
    """
    return policy_service.get_policy()


@router.post("/policy")
async def update_policy(
    policy: Policy = Body(...)
) -> Dict[str, Any]:
    """
    Update the current trading policy.
    
    Args:
        policy: The new policy to apply
        
    Returns:
        Status of the update operation
    """
    success = policy_service.update_policy(policy)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid policy")
    
    return {
        "success": True,
        "message": f"Policy updated to version {policy['version']}",
        "timestamp": int(time.time() * 1000)
    }