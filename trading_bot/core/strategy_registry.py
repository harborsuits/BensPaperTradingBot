# trading_bot/core/strategy_registry.py
"""
StrategyRegistry: Central registry for all trading strategies (including ML/AI).
Allows registration, retrieval, and extension of strategies by name.
Robust, extensible, and future-proof for all strategy logic.
"""

import threading
import logging
from typing import Dict, Type, Any

logger = logging.getLogger("StrategyRegistry")

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

# Enhanced Strategy Registry with comprehensive multi-asset strategy support
# Import base strategies
from trading_bot.strategy.implementations.standard_strategies import (
    MomentumStrategy, TrendFollowingStrategy, MeanReversionStrategy
)
from trading_bot.strategy.rl_strategy import MetaLearningStrategy
from trading_bot.strategy.ml_strategy import MLStrategy

# Import stock-specific strategies
from trading_bot.strategies.stocks.momentum import (
    StockMomentumStrategy, RelativeStrengthStrategy, PriceVolumeStrategy
)
from trading_bot.strategies.stocks.mean_reversion import (
    StockMeanReversionStrategy, RSIReversionStrategy, BollingerBandStrategy
)
from trading_bot.strategies.stocks.trend import (
    MovingAverageCrossStrategy, MAACDStrategy, TrendChannelStrategy
)
from trading_bot.strategies.stocks.breakout import (
    VolatilityBreakoutStrategy, PriceBreakoutStrategy, VolumeBreakoutStrategy
)

# Import crypto-specific strategies
# Import crypto strategies only if available
try:
    from trading_bot.strategies.crypto.momentum import (
        CryptoMomentumStrategy, CryptoRSIStrategy 
    )
    CRYPTO_MOMENTUM_AVAILABLE = True
except ImportError:
    # Define empty placeholders if imports fail
    CryptoMomentumStrategy = None
    CryptoRSIStrategy = None
    CRYPTO_MOMENTUM_AVAILABLE = False
    logging.warning("Crypto momentum strategies not available - this is fine if you're not using them")
# Import crypto mean reversion strategies only if available
try:
    from trading_bot.strategies.crypto.mean_reversion import (
        CryptoMeanReversionStrategy, CryptoRangeTradingStrategy
    )
    CRYPTO_MEAN_REVERSION_AVAILABLE = True
except ImportError:
    # Define empty placeholders if imports fail
    CryptoMeanReversionStrategy = None
    CryptoRangeTradingStrategy = None
    CRYPTO_MEAN_REVERSION_AVAILABLE = False
    logging.warning("Crypto mean reversion strategies not available - this is fine if you're not using them")

# Import crypto onchain strategies only if available
try:
    from trading_bot.strategies.crypto.onchain import (
        OnChainAnalysisStrategy, TokenFlowStrategy
    )
    CRYPTO_ONCHAIN_AVAILABLE = True
except ImportError:
    # Define empty placeholders if imports fail
    OnChainAnalysisStrategy = None
    TokenFlowStrategy = None
    CRYPTO_ONCHAIN_AVAILABLE = False
    logging.warning("Crypto onchain strategies not available - this is fine if you're not using them")

# Import forex-specific strategies
try:
    from trading_bot.strategies.forex.trend import (
        ForexTrendStrategy, ForexMAStrategy
    )
    FOREX_TREND_AVAILABLE = True
except ImportError as e:
    logging.error(f"Error importing forex trend strategies: {e}")
    FOREX_TREND_AVAILABLE = False
    # Create dummy objects for the missing classes
    class ForexTrendStrategy:
        """Stub implementation for ForexTrendStrategy"""
        pass
    
    class ForexMAStrategy:
        """Stub implementation for ForexMAStrategy"""
        pass

try:
    from trading_bot.strategies.forex.range import (
        ForexRangeStrategy, ForexSupportResistanceStrategy
    )
    FOREX_RANGE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Error importing forex range strategies: {e}")
    FOREX_RANGE_AVAILABLE = False
    # Create dummy objects for the missing classes
    class ForexRangeStrategy:
        """Stub implementation for ForexRangeStrategy"""
        pass
    
    class ForexSupportResistanceStrategy:
        """Stub implementation for ForexSupportResistanceStrategy"""
        pass

try:
    from trading_bot.strategies.forex.carry import (
        ForexCarryTradeStrategy, InterestRateDifferentialStrategy
    )
    FOREX_CARRY_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Error importing forex carry strategies: {e}")
    FOREX_CARRY_AVAILABLE = False
    # Create dummy objects for the missing classes
    class ForexCarryTradeStrategy:
        """Stub implementation for ForexCarryTradeStrategy"""
        pass
    
    class InterestRateDifferentialStrategy:
        """Stub implementation for InterestRateDifferentialStrategy"""
        pass

# Import options-specific strategies - use try-except to handle missing modules
try:
    from trading_bot.strategies.options.income_strategies import (
        CoveredCallStrategy, CashSecuredPutStrategy
    )
    INCOME_STRATEGIES_AVAILABLE = True
except ImportError as e:
    logging.error(f"Error importing options income strategies: {e}")
    INCOME_STRATEGIES_AVAILABLE = False
    # Create dummy classes
    class CoveredCallStrategy:
        """Stub implementation for CoveredCallStrategy"""
        pass
    
    class CashSecuredPutStrategy:
        """Stub implementation for CashSecuredPutStrategy"""
        pass
    
    class WheelStrategy:
        """Stub implementation for WheelStrategy"""
        pass

try:
    from trading_bot.strategies.options.volatility_spreads import (
        StrangleStrategy, StraddleStrategy, StraddleStrangleStrategy
    )
    # Try to import strangle variants
    try:
        from trading_bot.strategies.options.strangles import (
            LongStrangleStrategy, ShortStrangleStrategy
        )
    except ImportError:
        # Create stub implementations if they don't exist
        class LongStrangleStrategy:
            """Stub implementation for LongStrangleStrategy"""
            pass
            
        class ShortStrangleStrategy:
            """Stub implementation for ShortStrangleStrategy"""
            pass
    
    # Try to import the volatility arbitrage strategy
    try:
        from trading_bot.strategies.options.volatility import VolatilityArbitrageStrategy, VIXStrategy, ImpliedVolatilityStrategy
    except ImportError:
        # Create stub implementations if they don't exist
        class VolatilityArbitrageStrategy:
            """Stub implementation for VolatilityArbitrageStrategy"""
            pass
            
        class VIXStrategy:
            """Stub implementation for VIXStrategy"""
            pass
            
        class ImpliedVolatilityStrategy:
            """Stub implementation for ImpliedVolatilityStrategy"""
            pass
            
    VOLATILITY_STRATEGIES_AVAILABLE = True
except ImportError as e:
    logging.error(f"Error importing options volatility strategies: {e}")
    VOLATILITY_STRATEGIES_AVAILABLE = False
    # Create dummy classes
    class StrangleStrategy:
        """Stub implementation for StrangleStrategy"""
        pass
    
    class StraddleStrategy:
        """Stub implementation for StraddleStrategy"""
        pass
    
    class StraddleStrangleStrategy:
        """Stub implementation for StraddleStrangleStrategy"""
        pass
        
    class LongStrangleStrategy:
        """Stub implementation for LongStrangleStrategy"""
        pass
        
    class ShortStrangleStrategy:
        """Stub implementation for ShortStrangleStrategy"""
        pass
        
    class VolatilityArbitrageStrategy:
        """Stub implementation for VolatilityArbitrageStrategy"""
        pass
        
    class VIXStrategy:
        """Stub implementation for VIXStrategy"""
        pass
        
    class ImpliedVolatilityStrategy:
        """Stub implementation for ImpliedVolatilityStrategy"""
        pass

try:
    from trading_bot.strategies.options.complex_spreads import (
        IronCondorStrategy
    )
    # Try to import the bull/bear spread strategies
    try:
        from trading_bot.strategies.options.spreads import (
            BullCallSpreadStrategy, BearPutSpreadStrategy
        )
    except ImportError:
        # Create stub implementations if they don't exist
        class BullCallSpreadStrategy:
            """Stub implementation for BullCallSpreadStrategy"""
            pass
            
        class BearPutSpreadStrategy:
            """Stub implementation for BearPutSpreadStrategy"""
            pass
    
    COMPLEX_SPREADS_AVAILABLE = True
except ImportError as e:
    logging.error(f"Error importing options complex spread strategies: {e}")
    COMPLEX_SPREADS_AVAILABLE = False
    # Create dummy classes
    class IronCondorStrategy:
        """Stub implementation for IronCondorStrategy"""
        pass
        
    class BullCallSpreadStrategy:
        """Stub implementation for BullCallSpreadStrategy"""
        pass
        
    class BearPutSpreadStrategy:
        """Stub implementation for BearPutSpreadStrategy"""
        pass

# Register base strategies
StrategyRegistry.register("momentum", MomentumStrategy)
StrategyRegistry.register("trend_following", TrendFollowingStrategy)
StrategyRegistry.register("mean_reversion", MeanReversionStrategy)
StrategyRegistry.register("meta_learning", MetaLearningStrategy)
StrategyRegistry.register("ml_strategy", MLStrategy)

# Register stock-specific strategies
StrategyRegistry.register("stock_momentum", StockMomentumStrategy)
StrategyRegistry.register("stock_relative_strength", RelativeStrengthStrategy)
StrategyRegistry.register("stock_price_volume", PriceVolumeStrategy)
StrategyRegistry.register("stock_mean_reversion", StockMeanReversionStrategy)
StrategyRegistry.register("stock_rsi_reversion", RSIReversionStrategy)
StrategyRegistry.register("stock_bollinger", BollingerBandStrategy)
StrategyRegistry.register("stock_ma_cross", MovingAverageCrossStrategy)
StrategyRegistry.register("stock_macd", MAACDStrategy)
StrategyRegistry.register("stock_trend_channel", TrendChannelStrategy)
StrategyRegistry.register("stock_volatility_breakout", VolatilityBreakoutStrategy)
StrategyRegistry.register("stock_price_breakout", PriceBreakoutStrategy)
StrategyRegistry.register("stock_volume_breakout", VolumeBreakoutStrategy)

# Register crypto-specific strategies only if they're available
if CRYPTO_MOMENTUM_AVAILABLE:
    StrategyRegistry.register("crypto_momentum", CryptoMomentumStrategy)
    StrategyRegistry.register("crypto_rsi", CryptoRSIStrategy)

if CRYPTO_MEAN_REVERSION_AVAILABLE:
    StrategyRegistry.register("crypto_mean_reversion", CryptoMeanReversionStrategy)
    StrategyRegistry.register("crypto_range", CryptoRangeTradingStrategy)

if CRYPTO_ONCHAIN_AVAILABLE:
    StrategyRegistry.register("crypto_onchain", OnChainAnalysisStrategy)
    StrategyRegistry.register("crypto_token_flow", TokenFlowStrategy)

# Register forex-specific strategies
StrategyRegistry.register("forex_trend", ForexTrendStrategy)
StrategyRegistry.register("forex_ma", ForexMAStrategy)
StrategyRegistry.register("forex_range", ForexRangeStrategy)
StrategyRegistry.register("forex_support_resistance", ForexSupportResistanceStrategy)
StrategyRegistry.register("forex_carry", ForexCarryTradeStrategy)
StrategyRegistry.register("forex_interest_rate", InterestRateDifferentialStrategy)

# Register options-specific strategies
StrategyRegistry.register("options_covered_call", CoveredCallStrategy)
StrategyRegistry.register("options_cash_secured_put", CashSecuredPutStrategy)
StrategyRegistry.register("options_wheel", WheelStrategy)
StrategyRegistry.register("options_volatility_arbitrage", VolatilityArbitrageStrategy)
StrategyRegistry.register("options_vix", VIXStrategy)
StrategyRegistry.register("options_implied_volatility", ImpliedVolatilityStrategy)
StrategyRegistry.register("options_bull_call_spread", BullCallSpreadStrategy)
StrategyRegistry.register("options_bear_put_spread", BearPutSpreadStrategy)
StrategyRegistry.register("options_iron_condor", IronCondorStrategy)
StrategyRegistry.register("options_long_strangle", LongStrangleStrategy)
StrategyRegistry.register("options_short_strangle", ShortStrangleStrategy)

# Asset type mapping for asset detection
AssetTypeStrategies = {
    "stock": [strategy for strategy in StrategyRegistry._registry.keys() if strategy.startswith("stock_")],
    "crypto": [strategy for strategy in StrategyRegistry._registry.keys() if strategy.startswith("crypto_")],
    "forex": [strategy for strategy in StrategyRegistry._registry.keys() if strategy.startswith("forex_")],
    "options": [strategy for strategy in StrategyRegistry._registry.keys() if strategy.startswith("options_")],
}
