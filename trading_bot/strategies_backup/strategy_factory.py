"""
Strategy Factory

This module provides a factory for creating trading strategies
based on strategy type and configuration.
"""

import logging
import os
from typing import Dict, Any, Optional, Type, Union

logger = logging.getLogger(__name__)

# Import typed settings if available
try:
    from trading_bot.config.typed_settings import (
        load_config, StrategySettings, NotificationSettings, TradingBotSettings
    )
    from trading_bot.config.migration_utils import get_config_from_legacy_path
    TYPED_SETTINGS_AVAILABLE = True
except ImportError:
    # Fallback to our local settings implementation
    from trading_bot.strategies.strategy_settings import StrategySettings, TradingBotSettings
    TYPED_SETTINGS_AVAILABLE = False

# Import our notification wrapper
try:
    from trading_bot.strategies.strategy_notification_wrapper import wrap_strategy_with_notifications
    NOTIFICATION_WRAPPER_AVAILABLE = True
except ImportError:
    logger.warning("Strategy notification wrapper not available")
    NOTIFICATION_WRAPPER_AVAILABLE = False

# Import available strategies
try:
    # Stock strategies
    from trading_bot.strategies.stocks.momentum import MomentumStrategy
    from trading_bot.strategies.stocks.mean_reversion import MeanReversionStrategy
    from trading_bot.strategies.stocks.trend import MultiTimeframeCorrelationStrategy as TrendFollowingStrategy
    from trading_bot.strategies.stocks.breakout import VolatilityBreakoutStrategy
    from trading_bot.strategies.hybrid_strategy_adapter import HybridStrategyAdapter
    
    # Forex strategies
    from trading_bot.strategies.forex.trend_following_strategy import ForexTrendFollowingStrategy
    from trading_bot.strategies.forex.range_trading_strategy import ForexRangeTradingStrategy
    from trading_bot.strategies.forex.breakout_strategy import ForexBreakoutStrategy
    from trading_bot.strategies.forex.momentum_strategy import ForexMomentumStrategy
    from trading_bot.strategies.forex.scalping_strategy import ForexScalpingStrategy
    from trading_bot.strategies.forex.swing_trading_strategy import ForexSwingTradingStrategy
    from trading_bot.strategies.forex.news_trading_strategy import ForexNewsTrading
    from trading_bot.strategies.forex.carry_trade_strategy import ForexCarryTradeStrategy
    from trading_bot.strategies.forex.grid_trading_strategy import ForexGridTradingStrategy
    from trading_bot.strategies.forex.counter_trend_strategy import ForexCounterTrendStrategy
    from trading_bot.strategies.forex.day_trading_strategy import ForexDayTradingStrategy
    from trading_bot.strategies.forex.position_trading_strategy import ForexPositionTradingStrategy
    from trading_bot.strategies.forex.price_action_strategy import PriceActionStrategy
    from trading_bot.strategies.forex.retracement_strategy import ForexRetracementStrategy
    from trading_bot.strategies.forex.algorithmic_meta_strategy import AlgorithmicMetaStrategy
    from trading_bot.strategies.forex.one_hour_strategy import OneHourForexStrategy
    from trading_bot.strategies.forex.pips_a_day_strategy import PipsADayStrategy
    from trading_bot.strategies.forex.arbitrage_strategy import ArbitrageStrategy

    STRATEGIES_AVAILABLE = True
except ImportError:
    logger.warning("Some strategy modules could not be imported. Using mock strategies.")
    STRATEGIES_AVAILABLE = False

    # Define mock strategy classes for when real ones aren't available
    class BaseStrategy:
        """Base class for all trading strategies"""
        
        def __init__(self, config=None):
            self.config = config or {}
            logger.info(f"Initialized {self.__class__.__name__}")
            
        def generate_signals(self, data, **kwargs):
            """Generate trading signals"""
            return {"action": "hold", "confidence": 0.5}
        
        @classmethod
        def is_available(cls):
            return True
    
    class MomentumStrategy(BaseStrategy):
        """Mock momentum strategy"""
        pass
    
    class MeanReversionStrategy(BaseStrategy):
        """Mock mean reversion strategy"""
        pass
    
    class TrendFollowingStrategy(BaseStrategy):
        """Mock trend following strategy"""
        pass
    
    class VolatilityBreakoutStrategy(BaseStrategy):
        """Mock volatility breakout strategy"""
        pass


class StrategyFactory:
    """Factory class for creating strategy objects"""
    
    @staticmethod
    def create_strategy(strategy_type: str, 
                     config: Optional[Dict[str, Any]] = None, 
                     enable_notifications: bool = True,
                     settings: Optional[Union[StrategySettings, TradingBotSettings]] = None,
                     config_path: Optional[str] = None):
        """
        Create a strategy object by type
        
        Args:
            strategy_type: Type of strategy to create
            config: Optional configuration dictionary (legacy approach)
            enable_notifications: Whether to enable notifications for this strategy
            settings: Optional typed settings object (new approach)
            config_path: Optional path to config file to load settings from
            
        Returns:
            Strategy object with notification capabilities if enabled
        """
        # Prioritize typed settings if available, fall back to legacy config
        if TYPED_SETTINGS_AVAILABLE:
            # If full settings provided, extract strategy settings
            if settings is not None:
                if hasattr(settings, 'strategy') and hasattr(settings, 'notification'):
                    # Full TradingBotSettings provided
                    strategy_settings = settings.strategy
                    notification_settings = settings.notification
                elif hasattr(settings, 'parameters'):  
                    # Just StrategySettings provided
                    strategy_settings = settings
                    # Try to load notification settings from config path
                    notification_settings = None
                    if config_path:
                        try:
                            full_config = load_config(config_path)
                            notification_settings = full_config.notification
                        except Exception as e:
                            logger.warning(f"Could not load notification settings from config: {e}")
                else:
                    strategy_settings = None
                    notification_settings = None
            # If no settings provided but config path is, try to load from there
            elif config_path:
                try:
                    full_config = load_config(config_path)
                    strategy_settings = full_config.strategy
                    notification_settings = full_config.notification
                except Exception as e:
                    logger.warning(f"Could not load typed settings from path: {e}")
                    strategy_settings = None
                    notification_settings = None
            else:
                strategy_settings = None
                notification_settings = None
                
            # Convert typed settings to config dict if available
            if strategy_settings:
                # Use typed settings parameters for the specific strategy
                strategy_params = getattr(strategy_settings.parameters, strategy_type, {})
                # Start with strategy-specific parameters
                combined_config = {k: v for k, v in strategy_params.__dict__.items() if not k.startswith('_')}
                # Add general strategy settings
                combined_config.update({
                    'risk_per_trade': strategy_settings.risk_per_trade,
                    'max_positions': strategy_settings.max_positions
                })
                # Merge with provided legacy config (legacy takes precedence)
                if config:
                    combined_config.update(config)
                config = combined_config
            elif config is None:
                config = {}
        else:
            # Typed settings not available, just use the provided config
            config = config or {}
        
        # Create base strategy based on type
        strategy = None
        
        # Stock strategies
        if strategy_type.lower() == "momentum":
            strategy = MomentumStrategy("Momentum Strategy", parameters=config)
        elif strategy_type.lower() == "mean_reversion":
            strategy = MeanReversionStrategy("Mean Reversion Strategy", config)
        elif strategy_type.lower() == "trend_following":
            strategy = TrendFollowingStrategy("Trend Following Strategy", config)
        elif strategy_type.lower() == "volatility_breakout":
            strategy = VolatilityBreakoutStrategy("Volatility Breakout Strategy", config)
        elif strategy_type.lower() == "hybrid":
            strategy = HybridStrategyAdapter("Hybrid Strategy", parameters=config)
            logger.info("Created hybrid strategy combining technical, ML, and WeightedAvgPeak signals")
        
        # Forex strategies
        elif strategy_type.lower() == "forex_trend_following":
            strategy = ForexTrendFollowingStrategy("Forex Trend-Following Strategy", parameters=config)
            logger.info("Created forex trend-following strategy for currency pairs")
        elif strategy_type.lower() == "forex_range_trading":
            strategy = ForexRangeTradingStrategy("Forex Range Trading Strategy", parameters=config)
            logger.info("Created forex range trading strategy for currency pairs")
        elif strategy_type.lower() == "forex_breakout":
            strategy = ForexBreakoutStrategy("Forex Breakout Strategy", parameters=config)
            logger.info("Created forex breakout strategy for currency pairs")
        elif strategy_type.lower() == "forex_momentum":
            strategy = ForexMomentumStrategy("Forex Momentum Strategy", parameters=config)
            logger.info("Created forex momentum strategy for currency pairs")
        elif strategy_type.lower() == "forex_scalping":
            strategy = ForexScalpingStrategy("Forex Scalping Strategy", parameters=config)
            logger.info("Created forex scalping strategy for currency pairs")
        elif strategy_type.lower() == "forex_swing":
            strategy = ForexSwingTradingStrategy("Forex Swing Trading Strategy", parameters=config)
            logger.info("Created forex swing trading strategy for currency pairs")
        elif strategy_type.lower() == "forex_news":
            strategy = ForexNewsTrading("Forex News Trading Strategy", parameters=config)
            logger.info("Created forex news trading strategy for currency pairs")
        elif strategy_type.lower() == "forex_carry":
            strategy = ForexCarryTradeStrategy("Forex Carry Trade Strategy", parameters=config)
            logger.info("Created forex carry trade strategy for currency pairs")
        elif strategy_type.lower() == "forex_grid":
            strategy = ForexGridTradingStrategy("Forex Grid Trading Strategy", parameters=config)
            logger.info("Created forex grid trading strategy for currency pairs")
        elif strategy_type.lower() == "forex_counter_trend":
            strategy = ForexCounterTrendStrategy("Forex Counter-Trend Strategy", parameters=config)
            logger.info("Created forex counter-trend strategy for currency pairs")
        elif strategy_type.lower() == "forex_day_trading":
            strategy = ForexDayTradingStrategy("Forex Day Trading Strategy", parameters=config)
            logger.info("Created forex day trading strategy for currency pairs")
        elif strategy_type.lower() == "forex_position":
            strategy = ForexPositionTradingStrategy("Forex Position Trading Strategy", parameters=config)
            logger.info("Created forex position trading strategy for currency pairs")
        elif strategy_type.lower() == "forex_price_action":
            strategy = PriceActionStrategy("Forex Price Action Strategy", parameters=config)
            logger.info("Created forex price action strategy for currency pairs")
        elif strategy_type.lower() == "forex_retracement":
            strategy = ForexRetracementStrategy("Forex Retracement Strategy", parameters=config)
            logger.info("Created forex retracement strategy for currency pairs")
        elif strategy_type.lower() == "algorithmic_meta":
            strategy = AlgorithmicMetaStrategy("Algorithmic Meta Strategy", parameters=config)
            logger.info("Created algorithmic meta strategy for currency pairs")
        elif strategy_type.lower() == "one_hour_forex":
            strategy = OneHourForexStrategy("One-Hour Forex Strategy", parameters=config)
            logger.info("Created one-hour forex strategy for currency pairs")
        elif strategy_type.lower() == "pips_a_day":
            strategy = PipsADayStrategy("Pips-a-Day Forex Strategy", parameters=config)
            logger.info("Created pips-a-day forex strategy for currency pairs")
        elif strategy_type.lower() == "arbitrage" or strategy_type.lower() == "forex_arbitrage":
            strategy = ArbitrageStrategy("Forex Arbitrage Strategy", parameters=config)
            logger.info("Created arbitrage forex strategy for currency pairs")
        
        # Default fallback
        else:
            logger.warning(f"Unknown strategy type: {strategy_type}. Using momentum strategy.")
            strategy = MomentumStrategy("Momentum Strategy", parameters=config)
        
        # Wrap with notification capabilities if enabled
        if enable_notifications and NOTIFICATION_WRAPPER_AVAILABLE:
            # Check for notification settings from typed settings
            telegram_token = None
            telegram_chat_id = None
            
            if TYPED_SETTINGS_AVAILABLE and 'notification_settings' in locals() and notification_settings:
                # Use notification settings from typed config
                telegram_token = notification_settings.telegram_token
                telegram_chat_id = notification_settings.telegram_chat_id
                logger.info("Using telegram notification settings from typed config")
            
            # Fall back to environment variables or config dict
            telegram_token = telegram_token or os.environ.get("TELEGRAM_BOT_TOKEN") or config.get("telegram_token")
            telegram_chat_id = telegram_chat_id or os.environ.get("TELEGRAM_CHAT_ID") or config.get("telegram_chat_id")
            
            if telegram_token and telegram_chat_id:
                # Wrap the strategy
                wrapped_strategy = wrap_strategy_with_notifications(strategy, telegram_token, telegram_chat_id)
                logger.info(f"Strategy {strategy_type} wrapped with notification capabilities")
                return wrapped_strategy
            else:
                logger.warning("Notification wrapper enabled but telegram credentials missing")
                return strategy
        
        return strategy
    
    @staticmethod
    def available_strategies():
        """Get names of available strategies"""
        return [
            # Stock strategies
            "momentum",
            "mean_reversion",
            "trend_following",
            "volatility_breakout",
            "hybrid",
            
            # Forex strategies
            "forex_trend_following",
            "forex_range_trading",
            "forex_breakout",
            "forex_momentum",
            "forex_scalping",
            "forex_swing",
            "forex_news",
            "forex_carry",
            "forex_grid",
            "forex_counter_trend",
            'forex_day_trading',
            'forex_position_trading',
            'forex_price_action',
            'forex_retracement',
            'algorithmic_meta',
            'one_hour_forex',
            'forex_pips_a_day',
            'forex_arbitrage'
        ]