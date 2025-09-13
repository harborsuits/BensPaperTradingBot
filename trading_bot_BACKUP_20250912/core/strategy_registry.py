# trading_bot/core/strategy_registry.py
"""
StrategyRegistry: Central registry for all trading strategies (including ML/AI).
Allows registration, retrieval, and extension of strategies by name.
Robust, extensible, and future-proof for all strategy logic.
"""

import threading
import logging
from typing import Dict, Type, Any
from enum import Enum

logger = logging.getLogger("StrategyRegistry")

class StrategyType(Enum):
    """Enumeration of strategy types."""
    STOCKS = "stocks"
    OPTIONS = "options"
    FOREX = "forex"
    CRYPTO = "crypto"
    MULTI_ASSET = "multi_asset"
    ARBITRAGE = "arbitrage"
    EVENT_DRIVEN = "event_driven"
    MACRO_TREND = "macro_trend"
    PATTERN_DETECTION = "pattern_detection"
    MACHINE_LEARNING = "machine_learning"
    REGIME_AWARE = "regime_aware"
    NEWS_SENTIMENT = "news_sentiment"
    EXTERNAL_SIGNAL = "external_signal"
    # Added for options strategies
    VOLATILITY = "volatility"
    INCOME = "income"
    # Additional strategy types
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    RANGE = "range"
    MOMENTUM = "momentum"
    CARRY = "carry"
    SCALPING = "scalping"
    SWING = "swing"
    DAY_TRADING = "day_trading"
    POSITION = "position"
    ML_BASED = "ml_based"
    ENSEMBLE = "ensemble"

class AssetClass(Enum):
    """Enumeration of asset classes."""
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    CRYPTO = "crypto"
    DERIVATIVE = "derivative"
    MULTI_ASSET = "multi_asset"

class MarketRegime(Enum):
    """Enumeration of market regime types."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"

class TimeFrame(Enum):
    """Enumeration of timeframes."""
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1M"

class StrategyRegistry:
    _registry: Dict[str, Any] = {}
    _lock = threading.Lock()

    @classmethod
    def register(cls, name: str, strategy_cls: Any):
        with cls._lock:
            if name in cls._registry:
                logger.warning(f"Strategy '{name}' is already registered. Overwriting.")
            cls._registry[name] = strategy_cls
            logger.info(f"Registered strategy: {name}")

    @classmethod
    def get(cls, name: str):
        with cls._lock:
            if name not in cls._registry:
                raise ValueError(f"Strategy '{name}' not registered.")
            return cls._registry[name]

    @classmethod
    def list_strategies(cls):
        with cls._lock:
            return list(cls._registry.keys())
            
    @classmethod
    def list_strategies_by_asset(cls, asset_type: str):
        """Return strategies for a specific asset type (stock, crypto, forex, options)."""
        with cls._lock:
            if asset_type.lower() == "stock":
                return [s for s in cls._registry.keys() if s.startswith("stock_")]
            elif asset_type.lower() == "crypto":
                return [s for s in cls._registry.keys() if s.startswith("crypto_")]
            elif asset_type.lower() == "forex":
                return [s for s in cls._registry.keys() if s.startswith("forex_")]
            elif asset_type.lower() == "options":
                return [s for s in cls._registry.keys() if s.startswith("options_")]
            else:
                return cls.list_strategies()

    @classmethod
    def create(cls, name: str, *args, **kwargs):
        strategy_cls = cls.get(name)
        return strategy_cls(*args, **kwargs)

# Strategy initialization is now lazy - strategies are registered on demand
# This prevents import-time hangs and TensorFlow mutex issues

def _initialize_strategies():
    """Lazy initialization of strategies to avoid import-time hangs"""
    # This function will be called when strategies are first needed
    pass

# Asset type mapping for asset detection - initialize empty
AssetTypeStrategies = {
    "stock": [],
    "crypto": [],
    "forex": [],
    "options": [],
}
