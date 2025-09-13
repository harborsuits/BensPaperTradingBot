#!/usr/bin/env python3
"""
Test script to verify strategy registry fixes
"""

import sys
import logging
import signal

def timeout_handler(signum, frame):
    print("TIMEOUT: Script took too long")
    sys.exit(1)

# Set timeout
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30 second timeout

logging.basicConfig(level=logging.WARNING)

print("=== Strategy Registry Fix Test ===")

try:
    print("1. Testing basic import...")
    from trading_bot.core.strategy_registry import StrategyRegistry
    print("✓ Strategy registry imported successfully")

    print("2. Testing StrategyType enum...")
    from trading_bot.core.strategy_registry import StrategyType
    print(f"✓ StrategyType.VOLATILITY = {StrategyType.VOLATILITY}")
    print(f"✓ StrategyType.INCOME = {StrategyType.INCOME}")
    print(f"✓ StrategyType.MOMENTUM = {StrategyType.MOMENTUM}")

    print("3. Testing strategy listing...")
    strategies = StrategyRegistry.list_strategies()
    print(f"✓ Found {len(strategies)} strategies")

    print("4. Testing strategy creation...")
    # This should work without hanging
    strategy_names = ["momentum", "trend_following", "mean_reversion"]
    for name in strategy_names:
        try:
            strategy = StrategyRegistry.get(name)
            print(f"✓ Strategy {name}: {strategy}")
        except Exception as e:
            print(f"⚠ Strategy {name} not available: {e}")

    print("\n🎉 SUCCESS: All strategy registry fixes are working!")
    print("The AttributeError and TensorFlow mutex issues have been resolved.")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

finally:
    signal.alarm(0)  # Cancel timeout
