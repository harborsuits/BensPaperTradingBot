"""
Market Regime Dashboard Component

This module provides FastAPI endpoints and UI components for visualizing
and controlling the market regime detection system.
"""

import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
import json
import pandas as pd
from datetime import datetime, timedelta

# Import from local modules
from trading_bot.analytics.market_regime.detector import MarketRegimeType
from trading_bot.analytics.market_regime.integration import MarketRegimeManager
from trading_bot.core.event_bus import EventBus
from trading_bot.dashboard.dependencies import get_event_bus, get_regime_manager

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/regimes",
    tags=["regimes"],
    responses={404: {"description": "Not found"}},
)

# Define models for API
class RegimeInfo(BaseModel):
    symbol: str
    timeframe: str
    regime: str
    confidence: float
    features: Dict[str, float]
    timestamp: str

class StrategyInfo(BaseModel):
    strategy_id: str
    score: float
    weight: float
    name: str
    config: Dict[str, Any]

class SymbolRegimeInfo(BaseModel):
    symbol: str
    regimes: Dict[str, str]
    active_strategies: List[str]
    preferred_timeframe: str

class RegimePerformanceInfo(BaseModel):
    regime: str
    symbols_count: int
    symbols: List[str]
    top_strategies: List[Dict[str, Any]]

class ParameterSetRequest(BaseModel):
    strategy_id: str
    symbol: str
    timeframe: str

class TimeframeMapRequest(BaseModel):
    symbol: str
    regime: str
    timeframe: str

class StrategyScoreInfo(BaseModel):
    strategy_id: str
    score: float
    rank: int

@router.get("/current")
async def get_current_regimes(
    symbols: Optional[str] = Query(None, description="Comma-separated list of symbols"),
    regime_manager: MarketRegimeManager = Depends(get_regime_manager)
) -> Dict[str, Dict[str, RegimeInfo]]:
    """
    Get current market regimes for specified symbols.
    
    Args:
        symbols: Optional comma-separated list of symbols (defaults to all tracked symbols)
        
    Returns:
        Dict mapping symbols to timeframes to regime info
    """
    try:
        # Parse symbols
        if symbols:
            symbol_list = symbols.split(',')
        else:
            symbol_list = list(regime_manager.tracked_symbols)
        
        result = {}
        
        for symbol in symbol_list:
            # Get regimes for this symbol
            regimes = regime_manager.detector.get_current_regimes(symbol)
            
            # Convert to response format
            symbol_regimes = {}
            
            for timeframe, regime_data in regimes.items():
                regime_type = regime_data.get('regime')
                
                if regime_type:
                    symbol_regimes[timeframe] = RegimeInfo(
                        symbol=symbol,
                        timeframe=timeframe,
                        regime=regime_type.value,
                        confidence=regime_data.get('confidence', 0.0),
                        features=regime_data.get('features', {}),
                        timestamp=datetime.now().isoformat()
                    )
            
            if symbol_regimes:
                result[symbol] = symbol_regimes
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting current regimes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{symbol}/{timeframe}")
async def get_regime_history(
    symbol: str,
    timeframe: str,
    limit: int = Query(20, description="Max number of history points to return"),
    regime_manager: MarketRegimeManager = Depends(get_regime_manager)
) -> List[Dict[str, Any]]:
    """
    Get regime history for a symbol and timeframe.
    
    Args:
        symbol: Symbol
        timeframe: Timeframe
        limit: Max number of history points
        
    Returns:
        List of regime transitions
    """
    try:
        # Get history
        history = regime_manager.detector.get_regime_history(symbol, timeframe)
        
        # Apply limit
        if limit and len(history) > limit:
            history = history[-limit:]
        
        # Convert to response format
        result = []
        
        for entry in history:
            result.append({
                "timestamp": entry.get("timestamp").isoformat() if hasattr(entry.get("timestamp"), "isoformat") else entry.get("timestamp"),
                "regime": entry.get("regime").value if hasattr(entry.get("regime"), "value") else entry.get("regime"),
                "confidence": entry.get("confidence", 0.0),
                "features": entry.get("features", {})
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting regime history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/transitions/{symbol}")
async def get_regime_transitions(
    symbol: str,
    days: int = Query(30, description="Number of days to look back"),
    regime_manager: MarketRegimeManager = Depends(get_regime_manager)
) -> List[Dict[str, Any]]:
    """
    Get regime transitions for a symbol.
    
    Args:
        symbol: Symbol
        days: Number of days to look back
        
    Returns:
        List of regime transitions
    """
    try:
        # Get history from regime manager
        if symbol not in regime_manager.regime_history:
            return []
        
        # Calculate cutoff time
        cutoff_time = datetime.now() - timedelta(days=days)
        
        transitions = []
        
        for timeframe, history in regime_manager.regime_history[symbol].items():
            for entry in history:
                timestamp = entry.get("timestamp")
                
                # Skip if before cutoff
                if timestamp < cutoff_time:
                    continue
                
                transitions.append({
                    "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else timestamp,
                    "timeframe": timeframe,
                    "old_regime": entry.get("old_regime"),
                    "new_regime": entry.get("new_regime"),
                    "confidence": entry.get("confidence", 0.0)
                })
        
        # Sort by timestamp
        transitions.sort(key=lambda x: x["timestamp"])
        
        return transitions
        
    except Exception as e:
        logger.error(f"Error getting regime transitions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategies/{symbol}")
async def get_symbol_strategies(
    symbol: str,
    regime_manager: MarketRegimeManager = Depends(get_regime_manager)
) -> List[StrategyInfo]:
    """
    Get active strategies for a symbol.
    
    Args:
        symbol: Symbol
        
    Returns:
        List of active strategies
    """
    try:
        # Get active strategies
        strategy_ids = regime_manager.strategy_selector.get_active_strategies(symbol)
        
        result = []
        
        for strategy_id in strategy_ids:
            # Get strategy config
            config = regime_manager.strategy_selector.strategy_configs.get(strategy_id, {})
            
            # Get strategy score and weight
            score = 0.0
            weight = regime_manager.strategy_selector.get_strategy_weight(symbol, strategy_id)
            
            # Get current regime
            current_regime = None
            if symbol in regime_manager.active_regimes and regime_manager.config.get("primary_timeframe") in regime_manager.active_regimes[symbol]:
                current_regime = regime_manager.active_regimes[symbol][regime_manager.config.get("primary_timeframe")]
                
                # Get score for this regime
                if current_regime in regime_manager.strategy_selector.strategy_scores:
                    regime_scores = regime_manager.strategy_selector.strategy_scores[current_regime]
                    score = regime_scores.get(strategy_id, 0.0)
            
            # Create response object
            strategy_info = StrategyInfo(
                strategy_id=strategy_id,
                score=score,
                weight=weight,
                name=config.get("name", strategy_id),
                config=config
            )
            
            result.append(strategy_info)
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting symbol strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/parameters")
async def get_strategy_parameters(
    request: ParameterSetRequest,
    regime_manager: MarketRegimeManager = Depends(get_regime_manager)
) -> Dict[str, Any]:
    """
    Get optimized parameters for a strategy.
    
    Args:
        request: Parameter set request
        
    Returns:
        Dict of optimized parameters
    """
    try:
        # Get parameters
        params = regime_manager.get_parameter_set(
            request.strategy_id, request.symbol, request.timeframe
        )
        
        return params
        
    except Exception as e:
        logger.error(f"Error getting strategy parameters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/timeframe-mapping")
async def set_timeframe_mapping(
    request: TimeframeMapRequest,
    regime_manager: MarketRegimeManager = Depends(get_regime_manager)
) -> Dict[str, Any]:
    """
    Set preferred timeframe for a symbol in a specific regime.
    
    Args:
        request: Timeframe mapping request
        
    Returns:
        Success status
    """
    try:
        # Convert regime string to enum
        try:
            regime_type = MarketRegimeType(request.regime)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid regime type: {request.regime}")
        
        # Set mapping
        regime_manager.strategy_selector.set_timeframe_mapping(
            request.symbol, regime_type, request.timeframe
        )
        
        return {"success": True, "message": f"Set preferred timeframe for {request.symbol} in {request.regime} to {request.timeframe}"}
        
    except Exception as e:
        logger.error(f"Error setting timeframe mapping: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary")
async def get_regime_summary(
    regime_manager: MarketRegimeManager = Depends(get_regime_manager)
) -> Dict[str, Any]:
    """
    Get comprehensive regime performance summary.
    
    Returns:
        Dict with summary information
    """
    try:
        # Get summary
        summary = regime_manager.get_regime_performance_summary()
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting regime summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/scores/{regime}")
async def get_strategy_scores(
    regime: str,
    min_score: float = Query(0.0, description="Minimum score to include"),
    regime_manager: MarketRegimeManager = Depends(get_regime_manager)
) -> List[StrategyScoreInfo]:
    """
    Get strategy scores for a regime.
    
    Args:
        regime: Regime type
        min_score: Minimum score to include
        
    Returns:
        List of strategy scores
    """
    try:
        # Convert regime string to enum
        try:
            regime_type = MarketRegimeType(regime)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid regime type: {regime}")
        
        # Get scores
        scores = regime_manager.strategy_selector.get_strategy_scores(regime_type)
        
        # Convert to list
        score_list = [
            {"strategy_id": sid, "score": score}
            for sid, score in scores.items()
            if score >= min_score
        ]
        
        # Sort by score (descending)
        score_list.sort(key=lambda x: x["score"], reverse=True)
        
        # Add rank
        result = []
        for i, item in enumerate(score_list):
            result.append(StrategyScoreInfo(
                strategy_id=item["strategy_id"],
                score=item["score"],
                rank=i+1
            ))
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting strategy scores: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Function to add routes to the main app
def setup_regime_routes(app):
    """
    Add regime routes to the main app.
    
    Args:
        app: FastAPI app
    """
    app.include_router(router)
    
    logger.info("Registered regime dashboard routes")

# Frontend Components - These would be imported in the frontend code

def get_regime_dashboard_components():
    """
    Get frontend components for the regime dashboard.
    
    Returns:
        Dict of component configurations
    """
    return {
        "regime_summary": {
            "type": "card",
            "title": "Market Regime Summary",
            "endpoint": "/regimes/summary",
            "refresh_interval": 60000,  # 1 minute
            "visualization": "summary_table"
        },
        "regime_monitor": {
            "type": "card",
            "title": "Current Regimes",
            "endpoint": "/regimes/current",
            "refresh_interval": 30000,  # 30 seconds
            "visualization": "regime_grid"
        },
        "regime_history": {
            "type": "card",
            "title": "Regime History",
            "endpoint": "/regimes/history/{symbol}/{timeframe}",
            "parameters": ["symbol", "timeframe"],
            "refresh_interval": 60000,  # 1 minute
            "visualization": "timeline"
        },
        "regime_transitions": {
            "type": "card",
            "title": "Regime Transitions",
            "endpoint": "/regimes/transitions/{symbol}",
            "parameters": ["symbol", "days"],
            "refresh_interval": 300000,  # 5 minutes
            "visualization": "transition_chart"
        },
        "strategy_scores": {
            "type": "card",
            "title": "Strategy Performance by Regime",
            "endpoint": "/regimes/scores/{regime}",
            "parameters": ["regime"],
            "refresh_interval": 300000,  # 5 minutes
            "visualization": "score_chart"
        },
        "active_strategies": {
            "type": "card",
            "title": "Active Strategies",
            "endpoint": "/regimes/strategies/{symbol}",
            "parameters": ["symbol"],
            "refresh_interval": 60000,  # 1 minute
            "visualization": "strategy_table"
        }
    }
