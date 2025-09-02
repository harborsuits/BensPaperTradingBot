"""
Options Strategies Module

This module provides options trading strategies, including income, directional,
and volatility-based approaches.
"""

# Import options strategies
try:
    from trading_bot.strategies.options.spreads.butterfly_spread_strategy import ButterflySpreadStrategy
except ImportError:
    try:
        from trading_bot.strategies.options.spreads.butterfly_spread import ButterflySpreadStrategy
    except ImportError:
        print("Warning: Unable to import ButterflySpreadStrategy")
        # Create a placeholder class to avoid import errors
        class ButterflySpreadStrategy:
            pass

# Import other strategy modules with try/except to handle potential errors
try:
    from trading_bot.strategies.options.covered_call import CoveredCallStrategy
except ImportError:
    print("Warning: Unable to import CoveredCallStrategy")
    class CoveredCallStrategy:
        pass

try:
    from trading_bot.strategies.options.protective_put import ProtectivePutStrategy
except ImportError:
    print("Warning: Unable to import ProtectivePutStrategy")
    class ProtectivePutStrategy:
        pass

try:
    from trading_bot.strategies.options.iron_condor import IronCondorStrategy
except ImportError:
    print("Warning: Unable to import IronCondorStrategy")
    class IronCondorStrategy:
        pass

try:
    from trading_bot.strategies.options.iron_butterfly import IronButterflyStrategy
except ImportError:
    print("Warning: Unable to import IronButterflyStrategy")
    class IronButterflyStrategy:
        pass

try:
    from trading_bot.strategies.options.bull_put_spread import BullPutSpreadStrategy
except ImportError:
    print("Warning: Unable to import BullPutSpreadStrategy")
    class BullPutSpreadStrategy:
        pass

try:
    from trading_bot.strategies.options.bear_call_spread import BearCallSpreadStrategy
except ImportError:
    print("Warning: Unable to import BearCallSpreadStrategy")
    class BearCallSpreadStrategy:
        pass

try:
    from trading_bot.strategies.options.long_straddle import LongStraddleStrategy
except ImportError:
    print("Warning: Unable to import LongStraddleStrategy")
    class LongStraddleStrategy:
        pass

try:
    from trading_bot.strategies.options.short_straddle import ShortStraddleStrategy
except ImportError:
    print("Warning: Unable to import ShortStraddleStrategy")
    class ShortStraddleStrategy:
        pass

try:
    from trading_bot.strategies.options.long_strangle import LongStrangleStrategy
except ImportError:
    print("Warning: Unable to import LongStrangleStrategy")
    class LongStrangleStrategy:
        pass

try:
    from trading_bot.strategies.options.short_strangle import ShortStrangleStrategy
except ImportError:
    print("Warning: Unable to import ShortStrangleStrategy")
    class ShortStrangleStrategy:
        pass

# Define public API
__all__ = [
    'ButterflySpreadStrategy',
    'CoveredCallStrategy',
    'ProtectivePutStrategy',
    'IronCondorStrategy',
    'IronButterflyStrategy',
    'BullPutSpreadStrategy',
    'BearCallSpreadStrategy',
    'LongStraddleStrategy',
    'ShortStraddleStrategy',
    'LongStrangleStrategy',
    'ShortStrangleStrategy',
] 