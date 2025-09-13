"""
Strategy Endpoints for BenBot Trading Dashboard

This module provides endpoints for strategy-related data including:
- Active strategies
- Strategy rankings
- Strategy performance metrics
- Strategy signals
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import List, Dict, Any, Optional, Literal, Union
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import random
import asyncio
import json
from trading_bot.api.websocket_manager import enabled_manager
from trading_bot.api.websocket_channels import MessageType

router = APIRouter(prefix="/strategies", tags=["TradingStrategies"])

# ---- Model Definitions ----

class StrategyPerformance(BaseModel):
    total_return: float = Field(..., alias="totalReturn")
    win_rate: float = Field(..., alias="winRate")
    sharpe_ratio: float = Field(..., alias="sharpeRatio")
    max_drawdown: float = Field(..., alias="maxDrawdown")
    avg_return_per_trade: float = Field(..., alias="avgReturnPerTrade")
    profit_factor: float = Field(..., alias="profitFactor")
    
    class Config:
        allow_population_by_field_name = True

class StrategyRisk(BaseModel):
    volatility: float
    value_at_risk: float = Field(..., alias="valueAtRisk")
    expected_shortfall: float = Field(..., alias="expectedShortfall")
    beta: Optional[float] = None
    
    class Config:
        allow_population_by_field_name = True

class StrategyParameters(BaseModel):
    params: Dict[str, Any]
    optimized: bool = False
    last_optimized: Optional[str] = Field(None, alias="lastOptimized")
    
    class Config:
        allow_population_by_field_name = True

class Strategy(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    type: str
    status: Literal["active", "inactive", "monitoring"]
    rank: int
    confidence: float
    performance: StrategyPerformance
    risk: StrategyRisk
    parameters: StrategyParameters
    symbols: List[str]
    compatible_regimes: List[str] = Field([], alias="compatibleRegimes")
    created_at: str = Field(..., alias="createdAt")
    last_updated: str = Field(..., alias="lastUpdated")
    last_execution: Optional[str] = Field(None, alias="lastExecution")
    
    class Config:
        allow_population_by_field_name = True

class TradeCandidate(BaseModel):
    id: str
    strategy_id: str = Field(..., alias="strategyId")
    strategy_name: str = Field(..., alias="strategyName")
    symbol: str
    direction: Literal["long", "short"]
    confidence: float
    entry_price: float = Field(..., alias="entryPrice")
    stop_loss: float = Field(..., alias="stopLoss")
    take_profit: float = Field(..., alias="takeProfit")
    risk_reward_ratio: float = Field(..., alias="riskRewardRatio")
    timeframe: str
    reasons: List[str]
    signals: Dict[str, Any]
    conditions_met: List[str] = Field(..., alias="conditionsMet")
    generated_at: str = Field(..., alias="generatedAt")
    expires_at: str = Field(..., alias="expiresAt")
    status: Literal["pending", "executed", "expired", "rejected"]
    
    class Config:
        allow_population_by_field_name = True

# ---- Sample Data Generators ----

def generate_sample_strategies() -> List[Strategy]:
    now = datetime.utcnow()
    
    strategies = [
        Strategy(
            id="strat-001",
            name="Trend Following Momentum",
            description="Captures market momentum across multiple timeframes using technical indicators",
            type="momentum",
            status="active",
            rank=1,
            confidence=0.85,
            performance=StrategyPerformance(
                total_return=24.5,
                win_rate=0.68,
                sharpe_ratio=1.8,
                max_drawdown=12.3,
                avg_return_per_trade=1.2,
                profit_factor=2.1
            ),
            risk=StrategyRisk(
                volatility=14.2,
                value_at_risk=8.5,
                expected_shortfall=10.8,
                beta=0.95
            ),
            parameters=StrategyParameters(
                params={
                    "fast_ma": 12,
                    "slow_ma": 26,
                    "signal": 9,
                    "rsi_period": 14,
                    "rsi_overbought": 70,
                    "rsi_oversold": 30
                },
                optimized=True,
                last_optimized=(now - timedelta(days=7)).isoformat()
            ),
            symbols=["SPY", "QQQ", "DIA", "IWM"],
            compatible_regimes=["bullish", "neutral"],
            created_at=(now - timedelta(days=120)).isoformat(),
            last_updated=(now - timedelta(hours=6)).isoformat(),
            last_execution=(now - timedelta(hours=2)).isoformat()
        ),
        Strategy(
            id="strat-002",
            name="Mean Reversion Volatility",
            description="Exploits short-term price deviations using statistical methods",
            type="mean_reversion",
            status="active",
            rank=2,
            confidence=0.78,
            performance=StrategyPerformance(
                total_return=18.2,
                win_rate=0.72,
                sharpe_ratio=1.6,
                max_drawdown=9.8,
                avg_return_per_trade=0.9,
                profit_factor=1.8
            ),
            risk=StrategyRisk(
                volatility=11.5,
                value_at_risk=7.2,
                expected_shortfall=9.3,
                beta=0.85
            ),
            parameters=StrategyParameters(
                params={
                    "lookback_period": 20,
                    "entry_z_score": 2.0,
                    "exit_z_score": 0.5,
                    "max_holding_period": 5
                },
                optimized=True,
                last_optimized=(now - timedelta(days=12)).isoformat()
            ),
            symbols=["SPY", "QQQ", "TLT", "GLD", "USO"],
            compatible_regimes=["volatile", "neutral"],
            created_at=(now - timedelta(days=90)).isoformat(),
            last_updated=(now - timedelta(hours=12)).isoformat(),
            last_execution=(now - timedelta(hours=4)).isoformat()
        ),
        Strategy(
            id="strat-003",
            name="Adaptive Regime Allocation",
            description="Dynamically allocates assets based on detected market regimes",
            type="regime_based",
            status="active",
            rank=3,
            confidence=0.82,
            performance=StrategyPerformance(
                total_return=21.8,
                win_rate=0.65,
                sharpe_ratio=1.7,
                max_drawdown=14.5,
                avg_return_per_trade=1.1,
                profit_factor=1.9
            ),
            risk=StrategyRisk(
                volatility=13.8,
                value_at_risk=8.1,
                expected_shortfall=10.2,
                beta=0.92
            ),
            parameters=StrategyParameters(
                params={
                    "regime_detection_window": 60,
                    "allocation_weights": {
                        "bullish": {"SPY": 0.6, "QQQ": 0.4},
                        "bearish": {"TLT": 0.5, "GLD": 0.3, "SH": 0.2},
                        "volatile": {"VXX": 0.2, "GLD": 0.4, "TLT": 0.4}
                    }
                },
                optimized=True,
                last_optimized=(now - timedelta(days=14)).isoformat()
            ),
            symbols=["SPY", "QQQ", "TLT", "GLD", "SH", "VXX"],
            compatible_regimes=["bullish", "bearish", "volatile", "neutral"],
            created_at=(now - timedelta(days=150)).isoformat(),
            last_updated=(now - timedelta(hours=8)).isoformat(),
            last_execution=(now - timedelta(hours=6)).isoformat()
        ),
        Strategy(
            id="strat-004",
            name="Sentiment-Driven Momentum",
            description="Leverages market sentiment analysis with momentum indicators",
            type="sentiment",
            status="active",
            rank=4,
            confidence=0.75,
            performance=StrategyPerformance(
                total_return=16.4,
                win_rate=0.62,
                sharpe_ratio=1.5,
                max_drawdown=13.2,
                avg_return_per_trade=1.0,
                profit_factor=1.7
            ),
            risk=StrategyRisk(
                volatility=15.5,
                value_at_risk=9.8,
                expected_shortfall=12.1,
                beta=1.05
            ),
            parameters=StrategyParameters(
                params={
                    "sentiment_threshold": 0.65,
                    "momentum_lookback": 10,
                    "news_impact_weight": 0.4,
                    "technical_weight": 0.6
                },
                optimized=False,
                last_optimized=None
            ),
            symbols=["SPY", "XLK", "XLF", "XLC"],
            compatible_regimes=["bullish", "neutral"],
            created_at=(now - timedelta(days=60)).isoformat(),
            last_updated=(now - timedelta(hours=10)).isoformat(),
            last_execution=(now - timedelta(hours=8)).isoformat()
        ),
        Strategy(
            id="strat-005",
            name="Volatility Breakout",
            description="Captures significant price movements after periods of contraction",
            type="breakout",
            status="monitoring",
            rank=5,
            confidence=0.70,
            performance=StrategyPerformance(
                total_return=14.8,
                win_rate=0.55,
                sharpe_ratio=1.3,
                max_drawdown=16.5,
                avg_return_per_trade=1.4,
                profit_factor=1.6
            ),
            risk=StrategyRisk(
                volatility=18.2,
                value_at_risk=11.5,
                expected_shortfall=14.2,
                beta=1.15
            ),
            parameters=StrategyParameters(
                params={
                    "volatility_lookback": 20,
                    "breakout_threshold": 2.0,
                    "atr_multiple": 1.5,
                    "max_position_size": 0.1
                },
                optimized=True,
                last_optimized=(now - timedelta(days=21)).isoformat()
            ),
            symbols=["SPY", "QQQ", "IWM", "EEM", "EFA"],
            compatible_regimes=["volatile"],
            created_at=(now - timedelta(days=85)).isoformat(),
            last_updated=(now - timedelta(hours=24)).isoformat(),
            last_execution=(now - timedelta(hours=10)).isoformat()
        ),
        
        # Straddle/Strangle Volatility Strategy
        Strategy(
            id="strat-008",
            name="Straddle/Strangle Strategy",
            description="Options strategy that profits from significant price movements in either direction by simultaneously buying calls and puts",
            type="volatility",
            status="active",
            rank=3,
            confidence=0.78,
            performance=StrategyPerformance(
                total_return=28.4,
                win_rate=0.63,
                sharpe_ratio=1.9,
                max_drawdown=14.2,
                avg_return_per_trade=1.6,
                profit_factor=2.3
            ),
            risk=StrategyRisk(
                volatility=22.5,
                value_at_risk=12.8,
                expected_shortfall=15.6,
                beta=0.65
            ),
            parameters=StrategyParameters(
                params={
                    "days_to_expiration": 45,
                    "iv_percentile_threshold": 30,
                    "max_loss_pct": 50,
                    "profit_target_pct": 35,
                    "vix_change_exit": 10,
                    "use_strangle": True,
                    "strike_width_pct": 5,
                    "close_days_before_expiry": 7,
                    "entry_time": "market_open",
                    "min_option_liquidity": 0.6,
                    "use_earnings_events": True
                },
                optimized=True,
                last_optimized=(now - timedelta(days=14)).isoformat()
            ),
            symbols=["SPY", "QQQ", "IWM", "AAPL", "MSFT", "TSLA", "NVDA", "AMZN"],
            compatible_regimes=["volatile", "event_driven", "unknown"],
            created_at=(now - timedelta(days=95)).isoformat(),
            last_updated=(now - timedelta(days=5)).isoformat(),
            last_execution=(now - timedelta(hours=18)).isoformat()
        )
    ]
    
    return strategies

def generate_trade_candidates() -> List[TradeCandidate]:
    now = datetime.utcnow()
    
    candidates = [
        TradeCandidate(
            id="trade-001",
            strategy_id="strat-001",
            strategy_name="Trend Following Momentum",
            symbol="AAPL",
            direction="long",
            confidence=0.88,
            entry_price=187.50,
            stop_loss=180.25,
            take_profit=202.00,
            risk_reward_ratio=2.5,
            timeframe="daily",
            reasons=[
                "Strong momentum on daily and weekly timeframes",
                "Recent earnings beat",
                "Above all major moving averages",
                "Sector leadership"
            ],
            signals={
                "macd": "positive_crossover",
                "rsi": 65.2,
                "volume_trend": "increasing"
            },
            conditions_met=[
                "price_above_200ma",
                "positive_macd_crossover",
                "bullish_sector_trend",
                "volume_confirmation"
            ],
            generated_at=now.isoformat(),
            expires_at=(now + timedelta(days=2)).isoformat(),
            status="pending"
        ),
        TradeCandidate(
            id="trade-002",
            strategy_id="strat-002",
            strategy_name="Mean Reversion Volatility",
            symbol="QQQ",
            direction="long",
            confidence=0.76,
            entry_price=400.50,
            stop_loss=390.00,
            take_profit=425.00,
            risk_reward_ratio=2.3,
            timeframe="daily",
            reasons=[
                "Oversold on multiple indicators",
                "Positive divergence on RSI",
                "Near-term support level",
                "Alignment with broader market trend"
            ],
            signals={
                "rsi": 32.5,
                "stochastic": "oversold",
                "bollinger_bands": "lower_touch"
            },
            conditions_met=[
                "rsi_oversold",
                "stochastic_oversold",
                "price_near_support",
                "positive_divergence"
            ],
            generated_at=now.isoformat(),
            expires_at=(now + timedelta(days=1)).isoformat(),
            status="pending"
        ),
        TradeCandidate(
            id="trade-003",
            strategy_id="strat-003",
            strategy_name="Adaptive Regime Allocation",
            symbol="TLT",
            direction="long",
            confidence=0.82,
            entry_price=95.75,
            stop_loss=93.50,
            take_profit=101.25,
            risk_reward_ratio=2.4,
            timeframe="daily",
            reasons=[
                "Regime shift detected favoring bonds",
                "Yield curve dynamics supportive",
                "Risk-off sentiment increasing",
                "Technical support level"
            ],
            signals={
                "regime_score": 0.78,
                "yield_curve_trend": "flattening",
                "relative_strength": "improving"
            },
            conditions_met=[
                "bullish_regime_for_bonds",
                "above_50_day_ma",
                "yield_curve_support",
                "risk_metrics_favorable"
            ],
            generated_at=now.isoformat(),
            expires_at=(now + timedelta(days=3)).isoformat(),
            status="pending"
        ),
        TradeCandidate(
            id="trade-004",
            strategy_id="strat-004",
            strategy_name="Sentiment-Driven Momentum",
            symbol="NVDA",
            direction="long",
            confidence=0.85,
            entry_price=820.25,
            stop_loss=780.00,
            take_profit=895.00,
            risk_reward_ratio=2.1,
            timeframe="daily",
            reasons=[
                "Extremely positive sentiment metrics",
                "Strong momentum across timeframes",
                "Sector leadership",
                "Favorable news sentiment"
            ],
            signals={
                "sentiment_score": 0.88,
                "social_media_buzz": "very_high",
                "news_sentiment": "positive"
            },
            conditions_met=[
                "high_sentiment_score",
                "positive_price_action",
                "above_key_moving_averages",
                "high_relative_volume"
            ],
            generated_at=now.isoformat(),
            expires_at=(now + timedelta(days=2)).isoformat(),
            status="pending"
        ),
        TradeCandidate(
            id="trade-005",
            strategy_id="strat-005",
            strategy_name="Volatility Breakout",
            symbol="TSLA",
            direction="long",
            confidence=0.72,
            entry_price=245.75,
            stop_loss=230.00,
            take_profit=278.00,
            risk_reward_ratio=2.7,
            timeframe="daily",
            reasons=[
                "Breakout from volatility contraction",
                "Volume confirmation",
                "Key resistance level broken",
                "Sector rotation beneficial"
            ],
            signals={
                "volatility_contraction": "significant",
                "breakout_strength": "high",
                "volume_surge": 2.5
            },
            conditions_met=[
                "volatility_contraction_period",
                "price_breakout",
                "volume_confirmation",
                "bullish_sector_alignment"
            ],
            generated_at=now.isoformat(),
            expires_at=(now + timedelta(days=1)).isoformat(),
            status="pending"
        )
    ]
    
    return candidates

# ---- Endpoints ----

@router.get("", response_model=List[Strategy])
async def get_all_strategies(status: Optional[str] = None, type: Optional[str] = None):
    """Get all available strategies with optional filtering"""
    strategies = generate_sample_strategies()
    
    # Apply filters if specified
    if status:
        strategies = [s for s in strategies if s.status == status]
    if type:
        strategies = [s for s in strategies if s.type == type]
        
    return strategies

@router.get("/active", response_model=List[Strategy])
async def get_active_strategies():
    """Get currently active strategies"""
    strategies = generate_sample_strategies()
    return [s for s in strategies if s.status == "active"]

@router.get("/rankings", response_model=List[Strategy])
async def get_strategy_rankings():
    """Get strategies ranked by performance and confidence"""
    strategies = generate_sample_strategies()
    # Sort by rank (which is already set in our sample data)
    return sorted(strategies, key=lambda s: s.rank)

@router.get("/{strategy_id}", response_model=Strategy)
async def get_strategy_by_id(strategy_id: str):
    """Get detailed information about a specific strategy"""
    strategies = generate_sample_strategies()
    for strategy in strategies:
        if strategy.id == strategy_id:
            return strategy
    raise HTTPException(status_code=404, detail=f"Strategy with ID {strategy_id} not found")

@router.get("/trades/candidates", response_model=List[TradeCandidate])
async def get_trade_candidates(
    strategy_id: Optional[str] = None,
    symbol: Optional[str] = None,
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    status: Optional[str] = None
):
    """Get current trade candidates with optional filtering"""
    candidates = generate_trade_candidates()
    
    # Apply filters
    filtered_candidates = candidates
    
    if strategy_id:
        filtered_candidates = [c for c in filtered_candidates if c.strategy_id == strategy_id]
    
    if symbol:
        filtered_candidates = [c for c in filtered_candidates if c.symbol.upper() == symbol.upper()]
    
    if min_confidence > 0:
        filtered_candidates = [c for c in filtered_candidates if c.confidence >= min_confidence]
    
    if status:
        filtered_candidates = [c for c in filtered_candidates if c.status == status]
    
    return filtered_candidates

# ---- WebSocket Broadcast Functions ----

async def broadcast_strategy_update(strategy: Strategy):
    """Broadcast a strategy update to all connected clients"""
    await enabled_manager.broadcast_to_channel(
        "strategy", 
        MessageType.STRATEGY_UPDATE.value, 
        strategy.dict()
    )

async def broadcast_strategy_rankings(strategies: List[Strategy]):
    """Broadcast updated strategy rankings to all connected clients"""
    await enabled_manager.broadcast_to_channel(
        "strategy", 
        MessageType.STRATEGY_RANKING_UPDATE.value, 
        {"strategies": [s.dict() for s in strategies]}
    )

async def broadcast_new_trade_candidate(candidate: TradeCandidate):
    """Broadcast a new trade candidate to all connected clients"""
    await enabled_manager.broadcast_to_channel(
        "strategy", 
        MessageType.STRATEGY_SIGNAL.value, 
        candidate.dict()
    )

# This function could be called from a scheduled task
async def update_strategies_periodically(background_tasks: BackgroundTasks):
    """Background task to periodically update strategies and broadcast changes"""
    while True:
        try:
            # In a real implementation, this would fetch actual strategy updates
            strategies = generate_sample_strategies()
            
            # Broadcast updated rankings
            await broadcast_strategy_rankings(strategies)
            
            # Broadcast individual strategy updates (you'd normally only broadcast what changed)
            for strategy in strategies[:2]:  # Just update a couple as an example
                await broadcast_strategy_update(strategy)
                
            # Generate a new trade candidate occasionally
            candidates = generate_trade_candidates()
            if random.random() > 0.7:  # 30% chance of a new candidate
                await broadcast_new_trade_candidate(random.choice(candidates))
                
        except Exception as e:
            print(f"Error in strategy update task: {str(e)}")
            
        await asyncio.sleep(60)  # Update every minute
