#!/usr/bin/env python3
import sys
import logging
import signal

# Set up timeout
def timeout_handler(signum, frame):
    print("TIMEOUT: Script took too long to execute")
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(3)  # 3 second timeout

logging.basicConfig(level=logging.WARNING)

print("Step 1: Testing basic Python import...")
try:
    import trading_bot
    print("✓ trading_bot package imported")
except Exception as e:
    print(f"❌ Failed to import trading_bot: {e}")
    sys.exit(1)

print("Step 2: Testing core module import...")
try:
    import trading_bot.core
    print("✓ trading_bot.core imported")
except Exception as e:
    print(f"❌ Failed to import trading_bot.core: {e}")
    sys.exit(1)

print("Step 3: Testing strategy_registry module...")
try:
    import trading_bot.core.strategy_registry
    print("✓ trading_bot.core.strategy_registry imported")
except Exception as e:
    print(f"❌ Failed to import strategy_registry: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Step 4: Testing StrategyRegistry class...")
try:
    from trading_bot.core.strategy_registry import StrategyRegistry
    print("✓ StrategyRegistry imported successfully")
except Exception as e:
    print(f"❌ Failed to import StrategyRegistry: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Step 5: Testing StrategyType enum...")
try:
    from trading_bot.core.strategy_registry import StrategyType
    print(f"✓ StrategyType imported: {StrategyType}")
    print(f"✓ VOLATILITY exists: {hasattr(StrategyType, 'VOLATILITY')}")
    if hasattr(StrategyType, 'VOLATILITY'):
        print(f"✓ StrategyType.VOLATILITY = {StrategyType.VOLATILITY}")
    print(f"✓ INCOME exists: {hasattr(StrategyType, 'INCOME')}")
    if hasattr(StrategyType, 'INCOME'):
        print(f"✓ StrategyType.INCOME = {StrategyType.INCOME}")
except Exception as e:
    print(f"❌ Failed to access StrategyType: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("🎉 All import tests passed!")
signal.alarm(0)  # Cancel timeout
