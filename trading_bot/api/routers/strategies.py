"""
Strategies API routes.
Provides endpoints for trading strategies data and management.
"""
import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import random
from datetime import datetime, timedelta

# Import WebSocket manager
from trading_bot.api.websocket_manager import enabled_manager

logger = logging.getLogger("TradingBotAPI.Strategies")

router = APIRouter()

class StrategyData(BaseModel):
    id: str
    name: str
    description: str
    status: str
    allocation: float
    daily: float
    weekly: float
    monthly: float
    yearly: float
    activeTrades: int
    signalStrength: float
    lastUpdated: str
    marketFit: Optional[float] = Field(None, description="Score representing how well the strategy fits current market conditions")
    riskScore: Optional[float] = Field(None, description="Risk assessment score for the strategy")
    performanceScore: Optional[float] = Field(None, description="Overall performance score")
    trendScore: Optional[float] = Field(None, description="Score for how well the strategy performs in trending markets")
    volatilityScore: Optional[float] = Field(None, description="Score for how well the strategy performs in volatile markets")
    rankings: Optional[Dict[str, int]] = Field(None, description="Rankings for different metrics")
    
@router.get("/strategies", response_model=List[StrategyData])
async def get_strategies(status: Optional[str] = None):
    """Get trading strategies filtered by status."""
    try:
        # Define strategy templates based on memory of available strategies
        stock_strategies = [
            {"name": "StocksTrendFollowingStrategy", "description": "Follows medium-term market trends for stocks"},
            {"name": "GapTradingStrategy", "description": "Trades gaps in stock prices at market open"},
            {"name": "EarningsAnnouncementStrategy", "description": "Trades around earnings announcements"},
            {"name": "NewsSentimentStrategy", "description": "Trades based on news sentiment analysis"},
            {"name": "SectorRotationStrategy", "description": "Rotates between sectors based on market cycles"},
            {"name": "ShortSellingStrategy", "description": "Short sells overvalued or declining stocks"},
            {"name": "VolumeSurgeStrategy", "description": "Trades unusually high volume patterns"}
        ]
        
        options_strategies = [
            {"name": "BullCallSpreadStrategy", "description": "Bull call spread for moderately bullish outlook"},
            {"name": "BearPutSpreadStrategy", "description": "Bear put spread for moderately bearish outlook"},
            {"name": "IronCondorStrategy", "description": "Iron condor for range-bound markets"},
            {"name": "StraddleStrategy", "description": "Straddle for high volatility events"}
        ]
        
        forex_strategies = [
            {"name": "ForexTrendFollowingStrategy", "description": "Follows trends in currency pairs"},
            {"name": "ForexBreakoutStrategy", "description": "Trades breakouts in currency pairs"}
        ]
        
        crypto_strategies = [
            {"name": "CryptoSwingStrategy", "description": "Swing trading cryptocurrencies"},
            {"name": "CryptoMACDStrategy", "description": "MACD-based cryptocurrency trading"}
        ]
        
        # Combine all strategies
        all_strategies = []
        all_strategies.extend(stock_strategies)
        all_strategies.extend(options_strategies)
        all_strategies.extend(forex_strategies)
        all_strategies.extend(crypto_strategies)
        
        # Create full strategy objects with performance metrics
        strategies = []
        for strategy_template in all_strategies:
            # Generate random performance data
            daily = round(random.uniform(-2, 2), 2)
            weekly = round(random.uniform(-5, 5), 2)
            monthly = round(random.uniform(-10, 10), 2)
            yearly = round(random.uniform(-20, 40), 2)
            
            # Determine status
            strategy_status = random.choice(["active", "inactive", "testing"]) if not status else status
            
            # Create strategy object
            strategy = {
                "id": f"{strategy_template['name']}-{random.randint(1000, 9999)}",
                "name": strategy_template["name"],
                "description": strategy_template["description"],
                "status": strategy_status,
                "allocation": round(random.uniform(2, 15), 1),
                "daily": daily,
                "weekly": weekly,
                "monthly": monthly,
                "yearly": yearly,
                "activeTrades": random.randint(0, 5),
                "signalStrength": round(random.uniform(0, 1), 2),
                "lastUpdated": (datetime.now() - timedelta(minutes=random.randint(5, 60))).isoformat(),
                "marketFit": round(random.uniform(0, 1), 2),
                "riskScore": round(random.uniform(0.1, 0.9), 2),
                "performanceScore": round(random.uniform(0.1, 0.9), 2),
                "trendScore": round(random.uniform(0.1, 0.9), 2),
                "volatilityScore": round(random.uniform(0.1, 0.9), 2),
                "rankings": {
                    "overall": random.randint(1, len(all_strategies)),
                    "risk": random.randint(1, len(all_strategies)),
                    "performance": random.randint(1, len(all_strategies)),
                    "trend": random.randint(1, len(all_strategies)),
                    "volatility": random.randint(1, len(all_strategies))
                }
            }
            
            strategies.append(strategy)
        
        # Filter by status if provided
        if status:
            strategies = [s for s in strategies if s["status"] == status]
            
        return strategies
    except Exception as e:
        logger.error(f"Error fetching strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching strategies: {str(e)}")

@router.get("/strategies/{strategy_id}", response_model=StrategyData)
async def get_strategy(strategy_id: str):
    """Get information about a specific strategy."""
    try:
        # Get all strategies
        strategies = await get_strategies()
        
        # Find the specific strategy
        for strategy in strategies:
            if strategy["id"] == strategy_id:
                return strategy
        
        # Strategy not found
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error fetching strategy {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching strategy: {str(e)}")

@router.get("/strategies/active", response_model=List[StrategyData])
async def get_active_strategies():
    """Get all active trading strategies."""
    return await get_strategies(status="active")

@router.get("/strategies/ranked", response_model=List[StrategyData])
async def get_ranked_strategies(metric: str = "overall", limit: int = 10):
    """Get strategies ranked by a specific metric.
    
    Available metrics:
    - overall: Overall strategy ranking
    - risk: Risk-adjusted performance
    - performance: Raw performance
    - trend: Performance in trending markets
    - volatility: Performance in volatile markets
    - marketFit: Fit to current market conditions
    """
    try:
        # Get all strategies
        strategies = await get_strategies()
        
        # Sort based on the requested metric
        if metric in ["overall", "risk", "performance", "trend", "volatility"]:
            # Sort by the ranking (lower is better)
            sorted_strategies = sorted(strategies, key=lambda x: x["rankings"][metric])
        elif metric == "marketFit":
            # Sort by marketFit score (higher is better)
            sorted_strategies = sorted(strategies, key=lambda x: x["marketFit"], reverse=True)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid ranking metric: {metric}")
        
        # Return the top N strategies
        return sorted_strategies[:limit]
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error fetching ranked strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching ranked strategies: {str(e)}")

@router.post("/strategies/{strategy_id}/enable")
async def enable_strategy(strategy_id: str):
    """Enable a specific strategy."""
    try:
        # Get all strategies
        strategies = await get_strategies()
        
        # Find and update the specific strategy
        for strategy in strategies:
            if strategy["id"] == strategy_id:
                strategy["status"] = "active"
                # Broadcast strategy update to subscribed clients
                await enabled_manager.broadcast_to_channel("strategies", "strategy_update", strategy)
                return strategy
        
        # Strategy not found
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error enabling strategy {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error enabling strategy: {str(e)}")

@router.post("/strategies/{strategy_id}/disable")
async def disable_strategy(strategy_id: str):
    """Disable a specific strategy."""
    try:
        # Get all strategies
        strategies = await get_strategies()
        
        # Find and update the specific strategy
        for strategy in strategies:
            if strategy["id"] == strategy_id:
                strategy["status"] = "inactive"
                # Broadcast strategy update to subscribed clients
                await enabled_manager.broadcast_to_channel("strategies", "strategy_update", strategy)
                return strategy
        
        # Strategy not found
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error disabling strategy {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error disabling strategy: {str(e)}")
