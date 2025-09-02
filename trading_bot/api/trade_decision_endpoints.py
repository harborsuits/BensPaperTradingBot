"""
Trade Decision API endpoints.
Provides visibility into the decision-making process for trades.
"""
import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from datetime import datetime
import json
import os

# Import WebSocket manager
from trading_bot.api.websocket_manager import enabled_manager

# Import auth service
from trading_bot.auth.service import AuthService

# Import connection to trading engine (replace with your actual trading engine connection)
from trading_bot.core.engine import trading_engine

logger = logging.getLogger("TradingBotAPI.TradeDecisions")

router = APIRouter()

class TradeCandidate(BaseModel):
    """A potential trade candidate with its associated score and other attributes."""
    id: str
    symbol: str
    action: str  # buy, sell
    score: float
    confidence: float
    strategy_id: str
    strategy_name: str
    reasons: List[Dict[str, Any]]  # Contributing factors to the score
    timestamp: str
    status: str  # pending, executed, rejected
    target_price: Optional[float] = None
    estimated_size: Optional[int] = None
    risk_score: Optional[float] = None

class DecisionCycle(BaseModel):
    """A complete decision cycle with all candidates and market conditions."""
    id: str
    timestamp: str
    market_conditions: Dict[str, Any]
    candidates: List[TradeCandidate]
    executed_trades: List[str]  # IDs of candidates that were executed
    active_strategies: List[str]  # Names of strategies active in this cycle

@router.get("/decisions/latest", response_model=DecisionCycle)
async def get_latest_trade_candidates(
    current_user = Depends(AuthService.get_current_active_user),
):
    """
    Get the latest trade candidates and decision cycle.
    This shows all potential trades considered in the last decision cycle and
    their scores, including which ones were executed and which were rejected.
    """
    try:
        # In a real implementation, query this data from your trading engine
        # trading_engine.get_latest_decision_cycle()
        
        # For now, load from a decisions cache if it exists
        decisions_cache_path = os.path.join(
            os.path.dirname(__file__), 
            "../data/decisions_cache.json"
        )
        
        # Check if the cache exists, if not return an empty structure
        if os.path.exists(decisions_cache_path):
            try:
                with open(decisions_cache_path, 'r') as f:
                    cycle_data = json.load(f)
                    return cycle_data
            except Exception as e:
                logger.error(f"Error reading decisions cache: {str(e)}")
                # Fall through to retrieve from trading engine
        
        # If we're here, either the cache doesn't exist or couldn't be read
        # In a real implementation, this would pull from your trading engine
        # Here we'll connect to your actual trading engine's decision system
        
        try:
            # This should be replaced with your actual trading engine call
            cycle = trading_engine.get_latest_decision_cycle()
            
            # If we have a cycle, format it appropriately for the API
            if cycle:
                decision_data = {
                    "id": cycle.id,
                    "timestamp": cycle.timestamp.isoformat(),
                    "market_conditions": {
                        "regime": cycle.market_regime,
                        "volatility": cycle.volatility,
                        # Add other market conditions as needed
                    },
                    "candidates": [
                        {
                            "id": candidate.id,
                            "symbol": candidate.symbol,
                            "action": candidate.action,
                            "score": candidate.score,
                            "confidence": candidate.confidence,
                            "strategy_id": candidate.strategy_id,
                            "strategy_name": candidate.strategy_name,
                            "reasons": candidate.reasons,
                            "timestamp": candidate.timestamp.isoformat(),
                            "status": candidate.status,
                            "target_price": candidate.target_price,
                            "estimated_size": candidate.estimated_size,
                            "risk_score": candidate.risk_score
                        } for candidate in cycle.candidates
                    ],
                    "executed_trades": [trade.id for trade in cycle.executed_trades],
                    "active_strategies": cycle.active_strategies
                }
                
                # Broadcast decision cycle to clients subscribed to decisions channel
                try:
                    await enabled_manager.broadcast_to_channel("decisions", "decision_update", decision_data)
                except Exception as e:
                    logger.error(f"Error broadcasting decision update: {str(e)}")
                
                return decision_data
            
            # If no cycle is available, return an error
            raise HTTPException(
                status_code=404, 
                detail="No decision cycle data available"
            )
            
        except AttributeError:
            # This suggests trading_engine isn't properly initialized
            # or doesn't have the expected method
            logger.error("Trading engine not properly initialized for decision retrieval")
            raise HTTPException(
                status_code=500,
                detail="Trading engine not properly configured for decision retrieval"
            )
            
    except Exception as e:
        logger.error(f"Error fetching trade candidates: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error fetching trade candidates: {str(e)}"
        )

@router.get("/decisions", response_model=List[DecisionCycle])
async def get_decision_history(
    limit: int = Query(10, gt=0, le=100),
    offset: int = Query(0, ge=0),
    date: Optional[str] = None,
    current_user = Depends(AuthService.get_current_active_user),
):
    """
    Get historical decision cycles.
    Optionally filter by date.
    """
    try:
        # In a real implementation, query historical decision cycles
        # from your trading engine or database
        
        # If a date is provided, parse it
        filter_date = None
        if date:
            try:
                filter_date = datetime.fromisoformat(date)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid date format. Use ISO format (YYYY-MM-DD)"
                )
        
        # This would be replaced with actual fetching of historical data
        # from your trading engine or database
        try:
            # Call trading engine to get historical decisions
            history = trading_engine.get_decision_history(
                limit=limit, 
                offset=offset,
                date=filter_date
            )
            
            if history:
                return [
                    {
                        "id": cycle.id,
                        "timestamp": cycle.timestamp.isoformat(),
                        "market_conditions": {
                            "regime": cycle.market_regime,
                            "volatility": cycle.volatility,
                        },
                        "candidates": [
                            {
                                "id": candidate.id,
                                "symbol": candidate.symbol,
                                "action": candidate.action,
                                "score": candidate.score,
                                "confidence": candidate.confidence,
                                "strategy_id": candidate.strategy_id,
                                "strategy_name": candidate.strategy_name,
                                "reasons": candidate.reasons,
                                "timestamp": candidate.timestamp.isoformat(),
                                "status": candidate.status,
                                "target_price": candidate.target_price,
                                "estimated_size": candidate.estimated_size,
                                "risk_score": candidate.risk_score
                            } for candidate in cycle.candidates
                        ],
                        "executed_trades": [trade.id for trade in cycle.executed_trades],
                        "active_strategies": cycle.active_strategies
                    } for cycle in history
                ]
            
            # If no history available, return an empty list
            return []
            
        except AttributeError:
            logger.error("Trading engine not properly initialized for history retrieval")
            raise HTTPException(
                status_code=500,
                detail="Trading engine not properly configured for decision history"
            )
            
    except Exception as e:
        logger.error(f"Error fetching decision history: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error fetching decision history: {str(e)}"
        )
