#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Options Trading Strategies Package

This package contains implementations of various options trading strategies:

1. Vertical Spreads:
   - Bull Call Spread: Buying a call option and selling another call at higher strike
   - Bear Put Spread: Buying a put option and selling another put at lower strike
   - Bull Put Spread: Selling a put option and buying another put at lower strike
   
2. Complex Spreads:
   - Iron Condor: Combination of bull put spread and bear call spread
   - Butterfly Spread: Buying and selling calls or puts at specific strike prices
   - Iron Butterfly: Combining an at-the-money straddle with OTM strangle
   - Broken Wing Butterfly: Asymmetric butterfly with unequal wing distances
   
3. Time Spreads:
   - Calendar Spread: Selling short-term options and buying longer-term options
   - Diagonal Spread: Calendar spread with different strike prices
   - Double Calendar: Two calendar spreads at different strike prices
   
4. Volatility Spreads:
   - Straddle: Buying a call and put at the same strike price
   - Strangle: Buying a call and put at different strike prices
   
5. Advanced Spreads:
   - Jade Lizard: Combination of a short strangle and a long call spread
   - Ratio Spread: Buying and selling different quantities of options
"""

# Import base strategy
from trading_bot.strategies.options.base.options_base_strategy import OptionsBaseStrategy

# Import vertical spread strategies
try:
    # Use new implementations where available
    try:
        from trading_bot.strategies.options.vertical_spreads.bull_call_spread_strategy_new import BullCallSpreadStrategy
    except ImportError:
        from trading_bot.strategies.options.vertical_spreads.bull_call_spread_strategy import BullCallSpreadStrategy
    
    from trading_bot.strategies.options.vertical_spreads.bear_put_spread_strategy import BearPutSpreadStrategy
    from trading_bot.strategies.options.vertical_spreads.bull_put_spread_strategy import BullPutSpreadStrategy
except ImportError as e:
    import logging
    logging.warning(f"Some vertical spread strategies could not be imported: {e}")

# Import complex spread strategies
try:
    # Use new implementations where available
    try:
        from trading_bot.strategies.options.complex_spreads.iron_condor_strategy_new import IronCondorStrategy
    except ImportError:
        from trading_bot.strategies.options.complex_spreads.iron_condor_strategy import IronCondorStrategy
    
    from trading_bot.strategies.options.complex_spreads.butterfly_spread_strategy import ButterflySpreadStrategy
    from trading_bot.strategies.options.complex_spreads.collar_strategy import CollarStrategy
    
    # These may not exist yet, so handle them separately
    try:
        from trading_bot.strategies.options.complex_spreads.iron_butterfly_strategy import IronButterflyStrategy
    except ImportError:
        IronButterflyStrategy = None
        
    try:
        from trading_bot.strategies.options.complex_spreads.broken_wing_butterfly_strategy import BrokenWingButterflyStrategy
    except ImportError:
        BrokenWingButterflyStrategy = None
except ImportError as e:
    import logging
    logging.warning(f"Some complex spread strategies could not be imported: {e}")

# Import time spread strategies
try:
    from trading_bot.strategies.options.time_spreads.calendar_spread_strategy import CalendarSpreadStrategy
    from trading_bot.strategies.options.time_spreads.diagonal_spread_strategy import DiagonalSpreadStrategy
    from trading_bot.strategies.options.time_spreads.double_calendar_strategy import DoubleCalendarStrategy
except ImportError as e:
    import logging
    logging.warning(f"Some time spread strategies could not be imported: {e}")

# Import volatility spread strategies
try:
    from trading_bot.strategies.options.volatility_spreads.straddle_strangle_strategy import StraddleStrangleStrategy
    from trading_bot.strategies.options.volatility_spreads.strangle_strategy import StrangleStrategy
    from trading_bot.strategies.options.volatility_spreads.straddle_strategy import StraddleStrategy
except ImportError as e:
    import logging
    logging.warning(f"Some volatility spread strategies could not be imported: {e}")

# Import advanced spread strategies
try:
    from trading_bot.strategies.options.advanced_spreads.jade_lizard_strategy import JadeLizardStrategy
    from trading_bot.strategies.options.advanced_spreads.ratio_spread_strategy import RatioSpreadStrategy
except ImportError as e:
    import logging
    logging.warning(f"Some advanced spread strategies could not be imported: {e}")

# Register all strategies with the strategy registry
from trading_bot.core.strategy_registry import StrategyRegistry, StrategyType, AssetClass, MarketRegime, TimeFrame

try:
    # Register the base strategy
    StrategyRegistry.register("options_base", OptionsBaseStrategy)
    
    # FOCUS ON THE KEY STRATEGIES AS SPECIFIED BY THE USER
    # 1. Register Iron Condor - INCOME strategy
    try:
        # Prioritize the new implementation
        from trading_bot.strategies.options.complex_spreads.iron_condor_strategy_new import IronCondorStrategyNew
        StrategyRegistry.register(
            "iron_condor", 
            IronCondorStrategyNew,
            metadata={
                "type": StrategyType.INCOME,
                "asset_class": AssetClass.OPTIONS,
                "market_regime": MarketRegime.RANGE_BOUND,
                "timeframe": TimeFrame.SWING
            }
        )
        logging.info("Registered Iron Condor Strategy (New Implementation)")
    except ImportError:
        # Fall back to the original implementation if needed
        StrategyRegistry.register("iron_condor", IronCondorStrategy)
        logging.info("Registered Iron Condor Strategy (Original Implementation)")
    
    # 2. Register Strangle - VOLATILITY strategy
    StrategyRegistry.register(
        "strangle", 
        StrangleStrategy,
        metadata={
            "type": StrategyType.VOLATILITY,
            "asset_class": AssetClass.OPTIONS,
            "market_regime": MarketRegime.VOLATILE,
            "timeframe": TimeFrame.SWING
        }
    )
    logging.info("Registered Strangle Strategy")
    
    # 3. Register Straddle - VOLATILITY strategy
    StrategyRegistry.register(
        "straddle", 
        StraddleStrategy,
        metadata={
            "type": StrategyType.VOLATILITY,
            "asset_class": AssetClass.OPTIONS,
            "market_regime": MarketRegime.VOLATILE,
            "timeframe": TimeFrame.SWING
        }
    )
    logging.info("Registered Straddle Strategy")
    
    # 4. Register Combined Straddle/Strangle strategy - VOLATILITY strategy
    StrategyRegistry.register(
        "straddle_strangle", 
        StraddleStrangleStrategy,
        metadata={
            "type": StrategyType.VOLATILITY,
            "asset_class": AssetClass.OPTIONS,
            "market_regime": MarketRegime.VOLATILE,
            "timeframe": TimeFrame.SWING
        }
    )
    logging.info("Registered Straddle/Strangle Combined Strategy")
    
except (NameError, ImportError) as e:
    import logging
    logging.warning(f"Not all options strategies could be registered: {e}")
