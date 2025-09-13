"""
Strategy Library

Contains definitions and metadata for all trading strategies used in the system.
Each strategy includes information about optimal market regimes, risk profiles,
asset classes, and performance characteristics.
"""

from typing import Dict, List, Any

# Strategy metadata definitions
STRATEGY_METADATA = [
    {
        "name": "breakout_swing",
        "description": "Captures price movements when asset breaks through resistance or support levels",
        "preferred_regimes": ["trending", "volatile"],
        "avoid_regimes": ["sideways"],
        "risk_level": 7,  # 1-10 scale
        "drawdown_potential": 15,  # percentage
        "recommended_hold_time": "1-5 days",
        "asset_classes": ["stocks", "futures", "forex"],
        "strategy_type": "momentum",
        "indicators": ["price channels", "volume", "ATR", "support/resistance"],
        "performance": {
            "bull_market": "good",
            "bear_market": "moderate",
            "high_volatility": "very good",
            "low_volatility": "poor"
        }
    },
    {
        "name": "momentum",
        "description": "Follows established price trends using momentum indicators",
        "preferred_regimes": ["trending", "bullish", "bearish"],
        "avoid_regimes": ["choppy", "sideways"],
        "risk_level": 6,
        "drawdown_potential": 12,
        "recommended_hold_time": "3-10 days",
        "asset_classes": ["stocks", "ETFs", "futures"],
        "strategy_type": "trend",
        "indicators": ["RSI", "MACD", "moving averages", "ADX"],
        "performance": {
            "bull_market": "excellent",
            "bear_market": "good",
            "high_volatility": "moderate",
            "low_volatility": "poor"
        }
    },
    {
        "name": "mean_reversion",
        "description": "Capitalizes on price returning to average after deviation",
        "preferred_regimes": ["sideways", "range-bound", "oversold", "overbought"],
        "avoid_regimes": ["strongly trending", "news-driven"],
        "risk_level": 5,
        "drawdown_potential": 8,
        "recommended_hold_time": "1-3 days",
        "asset_classes": ["stocks", "ETFs", "indices"],
        "strategy_type": "reversal",
        "indicators": ["Bollinger Bands", "RSI", "stochastic oscillator", "ATR"],
        "performance": {
            "bull_market": "moderate",
            "bear_market": "moderate",
            "high_volatility": "poor",
            "low_volatility": "excellent"
        }
    },
    {
        "name": "trend_following",
        "description": "Ride established trends for longer-term positions",
        "preferred_regimes": ["strong trending", "bullish", "bearish"],
        "avoid_regimes": ["choppy", "sideways", "high volatility"],
        "risk_level": 5,
        "drawdown_potential": 15,
        "recommended_hold_time": "10-30 days",
        "asset_classes": ["stocks", "ETFs", "futures", "forex"],
        "strategy_type": "trend",
        "indicators": ["moving averages", "ADX", "MACD", "Ichimoku cloud"],
        "performance": {
            "bull_market": "excellent",
            "bear_market": "good",
            "high_volatility": "poor",
            "low_volatility": "good"
        }
    },
    {
        "name": "volatility_breakout",
        "description": "Captures explosive price movements during volatility expansion",
        "preferred_regimes": ["volatile", "news-driven", "regime shifts"],
        "avoid_regimes": ["sideways", "low volatility"],
        "risk_level": 8,
        "drawdown_potential": 20,
        "recommended_hold_time": "hours to 2 days",
        "asset_classes": ["stocks", "options", "futures", "forex"],
        "strategy_type": "momentum",
        "indicators": ["ATR", "Bollinger Band width", "historical volatility", "volume"],
        "performance": {
            "bull_market": "moderate",
            "bear_market": "good",
            "high_volatility": "excellent",
            "low_volatility": "very poor"
        }
    },
    {
        "name": "option_spreads",
        "description": "Uses multi-leg option positions to capitalize on volatility or directional views",
        "preferred_regimes": ["all", "high IV", "pre-earnings"],
        "avoid_regimes": ["extreme gaps"],
        "risk_level": 4,
        "drawdown_potential": 10,
        "recommended_hold_time": "5-30 days",
        "asset_classes": ["options"],
        "strategy_type": "income",
        "indicators": ["implied volatility", "IV rank", "IV percentile", "probability calculator"],
        "performance": {
            "bull_market": "good",
            "bear_market": "good",
            "high_volatility": "excellent",
            "low_volatility": "moderate"
        }
    },
    {
        "name": "swing_pullback",
        "description": "Enters during temporary pullbacks in established trends",
        "preferred_regimes": ["trending", "bullish", "bearish"],
        "avoid_regimes": ["choppy", "range-bound"],
        "risk_level": 6,
        "drawdown_potential": 10,
        "recommended_hold_time": "2-7 days",
        "asset_classes": ["stocks", "ETFs", "futures"],
        "strategy_type": "trend",
        "indicators": ["fibonacci retracements", "moving averages", "RSI", "volume"],
        "performance": {
            "bull_market": "excellent",
            "bear_market": "good",
            "high_volatility": "moderate",
            "low_volatility": "moderate"
        }
    },
    {
        "name": "gap_trading",
        "description": "Exploits price gaps at market open",
        "preferred_regimes": ["volatile", "news-driven"],
        "avoid_regimes": ["flat", "low volume"],
        "risk_level": 8,
        "drawdown_potential": 15,
        "recommended_hold_time": "intraday to 2 days",
        "asset_classes": ["stocks", "ETFs", "futures"],
        "strategy_type": "momentum",
        "indicators": ["pre-market volume", "gap percentage", "prior day's range", "news catalyst"],
        "performance": {
            "bull_market": "good",
            "bear_market": "good",
            "high_volatility": "excellent",
            "low_volatility": "poor"
        }
    }
]

# Dictionary mapping strategy names to their metadata
STRATEGY_MAP: Dict[str, Dict[str, Any]] = {s["name"]: s for s in STRATEGY_METADATA}


def get_strategy_for_regime(regime: str) -> List[Dict[str, Any]]:
    """
    Get strategies suitable for a specific market regime.
    
    Args:
        regime: Market regime (e.g., 'trending', 'volatile', 'sideways')
        
    Returns:
        List of strategy metadata objects suitable for the specified regime
    """
    return [
        s for s in STRATEGY_METADATA 
        if regime in s["preferred_regimes"] and regime not in s["avoid_regimes"]
    ]


def get_strategy_by_risk(max_risk_level: int) -> List[Dict[str, Any]]:
    """
    Get strategies below a specified risk level.
    
    Args:
        max_risk_level: Maximum risk level (1-10)
        
    Returns:
        List of strategy metadata objects with risk level <= max_risk_level
    """
    return [s for s in STRATEGY_METADATA if s["risk_level"] <= max_risk_level]


def get_strategy_by_asset_class(asset_class: str) -> List[Dict[str, Any]]:
    """
    Get strategies suitable for a specific asset class.
    
    Args:
        asset_class: Asset class (e.g., 'stocks', 'options', 'futures')
        
    Returns:
        List of strategy metadata objects suitable for the specified asset class
    """
    return [s for s in STRATEGY_METADATA if asset_class in s["asset_classes"]]


def get_strategy_names() -> List[str]:
    """Get a list of all available strategy names."""
    return [s["name"] for s in STRATEGY_METADATA]


def get_strategy_by_name(name: str) -> Dict[str, Any]:
    """
    Get strategy metadata by name.
    
    Args:
        name: Strategy name
        
    Returns:
        Strategy metadata object or None if not found
    """
    return STRATEGY_MAP.get(name)


# Optional: Add historical performance data that could be used for strategy rotation
HISTORICAL_PERFORMANCE = {
    "breakout_swing": {
        "2023-Q1": {"win_rate": 0.62, "avg_gain": 2.1, "max_drawdown": 5.2},
        "2023-Q2": {"win_rate": 0.58, "avg_gain": 1.8, "max_drawdown": 6.0},
        "2023-Q3": {"win_rate": 0.65, "avg_gain": 2.4, "max_drawdown": 4.8},
        "2023-Q4": {"win_rate": 0.60, "avg_gain": 1.9, "max_drawdown": 5.5}
    },
    "momentum": {
        "2023-Q1": {"win_rate": 0.55, "avg_gain": 2.5, "max_drawdown": 7.0},
        "2023-Q2": {"win_rate": 0.68, "avg_gain": 3.2, "max_drawdown": 5.8},
        "2023-Q3": {"win_rate": 0.70, "avg_gain": 3.5, "max_drawdown": 5.5},
        "2023-Q4": {"win_rate": 0.63, "avg_gain": 2.8, "max_drawdown": 6.2}
    },
    "mean_reversion": {
        "2023-Q1": {"win_rate": 0.72, "avg_gain": 1.2, "max_drawdown": 3.5},
        "2023-Q2": {"win_rate": 0.68, "avg_gain": 1.1, "max_drawdown": 3.8},
        "2023-Q3": {"win_rate": 0.60, "avg_gain": 0.9, "max_drawdown": 4.2},
        "2023-Q4": {"win_rate": 0.66, "avg_gain": 1.0, "max_drawdown": 3.9}
    }
    # Add more strategies as needed
} 