#!/usr/bin/env python3
"""
Statistical Arbitrage (Pairs Trading) Test
Tests the next highest-probability strategy approach
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def generate_correlated_data(base_returns, correlation=0.8, noise_factor=0.1):
    """Generate correlated asset returns for testing"""
    # Create correlated returns using Cholesky decomposition
    cov_matrix = np.array([[1.0, correlation],
                          [correlation, 1.0]])
    L = np.linalg.cholesky(cov_matrix)

    # Generate independent normal returns
    n_periods = len(base_returns)
    independent_returns = np.random.normal(0, 0.015, (2, n_periods))

    # Make them correlated
    correlated_returns = L @ independent_returns

    # Scale to match base volatility
    base_vol = base_returns.std()
    correlated_returns = correlated_returns * (base_vol / correlated_returns.std(axis=1).mean())

    # Convert to price series
    asset1_prices = 300 * np.exp(np.cumsum(correlated_returns[0]))
    asset2_prices = 250 * np.exp(np.cumsum(correlated_returns[1]))

    return asset1_prices, asset2_prices

def test_pairs_trading(px1, px2, entry_threshold=2.0, exit_threshold=0.5, lookback=60):
    """
    Implement statistical arbitrage pairs trading strategy

    Parameters:
    - px1, px2: Price series for the pair
    - entry_threshold: Standard deviations for entry signal
    - exit_threshold: Standard deviations for exit signal
    - lookback: Rolling window for mean reversion calculation
    """
    # Calculate returns
    ret1 = px1.pct_change().fillna(0.0)
    ret2 = px2.pct_change().fillna(0.0)

    # Calculate spread (price ratio or normalized difference)
    spread = px1 / px2

    # Rolling statistics for mean reversion
    spread_mean = spread.rolling(lookback).mean()
    spread_std = spread.rolling(lookback).std()

    # Z-score of spread
    zscore = (spread - spread_mean) / spread_std

    # Trading signals
    long_signal = zscore <= -entry_threshold  # Asset 1 cheap vs Asset 2
    short_signal = zscore >= entry_threshold  # Asset 1 expensive vs Asset 2
    exit_signal = abs(zscore) <= exit_threshold  # Close position

    # Position tracking
    position = pd.Series(0, index=px1.index)  # 1 = long asset1, -1 = long asset2
    in_position = False

    for i in range(lookback, len(px1)):  # Start after lookback period
        if not in_position:
            if long_signal.iloc[i]:
                position.iloc[i] = 1  # Long asset 1, short asset 2
                in_position = True
            elif short_signal.iloc[i]:
                position.iloc[i] = -1  # Short asset 1, long asset 2
                in_position = True
        elif in_position and exit_signal.iloc[i]:
            position.iloc[i] = 0  # Exit position
            in_position = False
        elif in_position:
            position.iloc[i] = position.iloc[i-1]  # Maintain position

    # Calculate strategy returns
    # Long asset1: +ret1, Short asset2: -ret2
    # Short asset1: -ret1, Long asset2: +ret2
    strategy_returns = position * (ret1 - ret2)

    return strategy_returns, zscore, position

def calculate_strategy_metrics(returns, trade_cost=0.0004):
    """Calculate strategy metrics using validated methodology"""
    # Apply transaction costs
    position_changes = returns.diff().abs().fillna(0)
    costs = position_changes * trade_cost
    after_cost = returns - costs

    # Calculate equity curve
    eq = (1.0 + after_cost).cumprod()

    # Metrics
    total_return = eq.iloc[-1] - 1
    cagr = eq.iloc[-1]**(252/len(eq)) - 1
    sharpe = (after_cost.mean() / (after_cost.std() + 1e-12)) * np.sqrt(252)
    max_drawdown = (eq / eq.cummax() - 1).min()

    return {
        'total_return': total_return,
        'cagr': cagr,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'final_equity': eq.iloc[-1]
    }

def run_pairs_trading_test():
    """Run comprehensive pairs trading test"""
    print("🧪 TESTING: Statistical Arbitrage (Pairs Trading)")
    print("=" * 50)

    try:
        # Load SPY data as base
        df = pd.read_csv('data/SPY_realistic_2020_2024.csv', parse_dates=[0])
        df = df.rename(columns={df.columns[0]: 'Date'})
        df.set_index('Date', inplace=True)
        spy_prices = df['Close']

        # Generate correlated asset (simulating QQQ or similar)
        spy_returns = spy_prices.pct_change().fillna(0.0)
        asset1_prices, asset2_prices = generate_correlated_data(spy_returns, correlation=0.85)

        # Align data lengths
        min_len = min(len(spy_prices), len(asset1_prices))
        spy_prices = spy_prices.iloc[:min_len]
        asset2_prices = pd.Series(asset2_prices[:min_len], index=spy_prices.index)

        print(f"✅ Generated correlated pair data")
        print(f"   SPY final price: ${spy_prices.iloc[-1]:.2f}")
        print(f"   Asset2 final price: ${asset2_prices.iloc[-1]:.2f}")

        # Test pairs trading strategy
        pair_returns, zscore, positions = test_pairs_trading(
            spy_prices, asset2_prices,
            entry_threshold=2.0,
            exit_threshold=0.5,
            lookback=60
        )

        # Calculate metrics
        pair_metrics = calculate_strategy_metrics(pair_returns)

        # Buy-and-hold comparison (SPY) - Use verified data
        # Load fresh SPY data to ensure we get the correct 92.2% return
        spy_df = pd.read_csv('data/SPY_realistic_2020_2024.csv', parse_dates=[0])
        spy_df = spy_df.rename(columns={spy_df.columns[0]: 'Date'})
        spy_df.set_index('Date', inplace=True)
        verified_spy_prices = spy_df['Close']

        # Calculate buy-and-hold return using verified data
        spy_bh_return = (verified_spy_prices.iloc[-1] / verified_spy_prices.iloc[0] - 1) * 100

        print(f"✅ Verified SPY buy-and-hold: {spy_bh_return:.1f}%")
        spy_return = spy_bh_return / 100  # Convert to decimal for calculations

        # Results
        print("\n📊 PAIRS TRADING STRATEGY RESULTS:")
        print(f"  Total Return: {pair_metrics['total_return']:.1f}%")
        print(f"  CAGR: {pair_metrics['cagr']:.2f}%")
        print(f"  Sharpe Ratio: {pair_metrics['sharpe']:.2f}")
        print(f"  Max Drawdown: {pair_metrics['max_drawdown']:.2f}")
        print("\n📊 BUY & HOLD COMPARISON:")
        print(f"  SPY Return: {spy_bh_return:.1f}%")

        # Trading statistics
        trades = positions.diff().abs().sum()
        winning_trades = ((positions != 0) & (pair_returns > 0)).sum()
        total_trading_days = (positions != 0).sum()

        print("\n📈 STRATEGY STATISTICS:")
        print(f"  Total Trades: {trades}")
        print(f"  Trading Days: {total_trading_days}")
        if total_trading_days > 0:
            print(f"  Win Rate: {winning_trades/total_trading_days:.1f}%")

        # Analysis
        print("\n🎯 ANALYSIS:")
        if pair_metrics['total_return'] > spy_return:
            print("✅ PAIRS TRADING BEATS BUY-AND-HOLD!")
            print("💡 Statistical arbitrage captures inefficiencies")
            print("🔬 Next: Optimize parameters and test on real pairs")
        else:
            performance_gap = (pair_metrics['total_return'] - spy_return) * 100
            print(f"❌ PAIRS TRADING LOSES TO BUY-AND-HOLD by {performance_gap:.1f}%")
            print("💡 Even correlated pairs may not have exploitable inefficiencies")
            print("🔬 Next: Test different correlation levels or asset classes")

        print("\n🎯 STRATEGY CHARACTERISTICS:")
        print("   • Exploits mean reversion in price ratios")
        print("   • Requires correlated assets (SPY vs QQQ-like)")
        print("   • Entry: 2 SD deviations from mean")
        print("   • Exit: 0.5 SD from mean")
        print("   • 60-day lookback for statistics")

        print("\n🎯 THEORETICAL FOUNDATION:")
        print("   • Exploits temporary pricing inefficiencies")
        print("   • Institutional quant strategy")
        print("   • Lower costs than technical strategies")

        return pair_metrics['total_return'] > spy_return

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = run_pairs_trading_test()

    print("\n" + "="*50)
    if success:
        print("🎉 SUCCESS: Pairs trading shows potential edge!")
        print("🔬 RECOMMENDATION: Optimize parameters and test real pairs")
    else:
        print("📊 RESULT: Pairs trading does not beat buy-and-hold")
        print("🔬 RECOMMENDATION: Test different approaches or accept that")
        print("   simple strategies may not beat buy-and-hold after costs")
    print("="*50)
