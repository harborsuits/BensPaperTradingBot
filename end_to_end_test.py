#!/usr/bin/env python3
"""
End-to-End Trading System Test
Tests the complete pipeline: market data → strategy → backtest → results
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def test_market_data():
    """Test market data loading"""
    print("🧪 Testing Market Data...")

    csv_path = "data/SPY.csv"
    if not Path(csv_path).exists():
        print("❌ SPY.csv not found")
        return False

    try:
        df = pd.read_csv(csv_path, parse_dates=[0])
        df = df.rename(columns={df.columns[0]: "Date"})
        df.set_index("Date", inplace=True)
        px = df["Close"].astype(float)

        print(f"✅ Loaded {len(px)} SPY price bars")
        print(".2f")
        print(".2f")
        return True
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def test_strategy_signal_generation():
    """Test strategy signal generation"""
    print("\n🧪 Testing Strategy Signals...")

    try:
        # Load data
        df = pd.read_csv("data/SPY.csv", parse_dates=[0])
        df = df.rename(columns={df.columns[0]: "Date"})
        df.set_index("Date", inplace=True)
        px = df["Close"].astype(float)

        # Generate MA signals
        fast_ma = px.rolling(5).mean()
        slow_ma = px.rolling(15).mean()
        signals = (fast_ma > slow_ma).astype(int)

        # Count signals
        signal_changes = signals.diff().abs()
        total_signals = signal_changes.sum()

        print(f"✅ Generated {int(total_signals)} trading signals")
        print(f"   Buy signals: {int(signals.sum())}")
        print(f"   Sell signals: {len(signals) - int(signals.sum())}")
        return True
    except Exception as e:
        print(f"❌ Error generating signals: {e}")
        return False

def test_backtest_execution():
    """Test backtest execution"""
    print("\n🧪 Testing Backtest Execution...")

    try:
        # Load data
        df = pd.read_csv("data/SPY.csv", parse_dates=[0])
        df = df.rename(columns={df.columns[0]: "Date"})
        df.set_index("Date", inplace=True)
        px = df["Close"].astype(float)

        # Calculate signals
        fast_ma = px.rolling(5).mean()
        slow_ma = px.rolling(15).mean()
        signal = (fast_ma > slow_ma).astype(int)

        # Calculate returns
        ret = px.pct_change().fillna(0.0)
        strat = signal.shift(1).fillna(0) * ret

        # Apply costs
        turns = signal.diff().abs().fillna(0)
        cost_per_turn = (1.0 + 2.0) / 10000.0  # 1bps fee + 2bps slip
        costs = turns * cost_per_turn
        after_cost = strat - costs

        # Calculate metrics
        ann_factor = 252.0
        sharpe = (after_cost.mean() / (after_cost.std() + 1e-12)) * np.sqrt(ann_factor)
        total_return = (1.0 + after_cost).prod() - 1.0

        print("✅ Backtest executed successfully")
        print(".2f")
        print(".2%")
        print(".2f")
        return True
    except Exception as e:
        print(f"❌ Error in backtest: {e}")
        return False

def test_paper_trading_integration():
    """Test paper trading integration"""
    print("\n🧪 Testing Paper Trading Integration...")

    try:
        # Test server connectivity
        import requests
        response = requests.get("http://localhost:4000/api/live/status", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running")
        else:
            print("❌ API server not responding")
            return False

        # Test paper account
        account_response = requests.get("http://localhost:4000/api/paper/account", timeout=5)
        if account_response.status_code == 200:
            account_data = account_response.json()
            balance = account_data["balances"]["total_cash"]
            print(".2f")
        else:
            print("❌ Paper account endpoint not working")
            return False

        return True
    except Exception as e:
        print(f"❌ Error testing paper trading: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 END-TO-END TRADING SYSTEM TEST")
    print("=" * 50)

    tests = [
        ("Market Data Loading", test_market_data),
        ("Strategy Signal Generation", test_strategy_signal_generation),
        ("Backtest Execution", test_backtest_execution),
        ("Paper Trading Integration", test_paper_trading_integration)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")

    print("\n" + "=" * 50)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED - System is ready for trading!")
        print("\n💡 Next Steps:")
        print("   1. Start the AI orchestrator: ./start_ai_orchestrator.sh")
        print("   2. Monitor performance: curl http://localhost:4000/api/live/ai/status")
        print("   3. View trades: curl http://localhost:4000/api/paper/orders")
    else:
        print(f"⚠️  {total - passed} tests failed - Fix issues before trading")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
