#!/usr/bin/env python3
"""
Covered Call Strategy Test
Tests the highest-probability strategy: collecting option premium while holding stock
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_spy_data():
    """Load SPY data for testing"""
    df = pd.read_csv('data/SPY_realistic_2020_2024.csv', parse_dates=[0])
    df = df.rename(columns={df.columns[0]: 'Date'})
    df.set_index('Date', inplace=True)
    return df['Close']

def simulate_covered_call(px, strike_pct_above=0.05, option_cost_pct=0.02):
    """
    Simulate covered call strategy
    - Buy SPY at market
    - Sell monthly calls 5% out-of-the-money
    - Account for option premium and assignment risk
    """
    returns = px.pct_change().fillna(0.0)

    # Strategy returns (initialize as float array)
    strategy_returns = pd.Series(0.0, index=px.index)

    # Monthly option selling (assume ~21 trading days per month)
    monthly_premium_rate = option_cost_pct / 12  # Monthly premium as % of stock value
    days_in_month = 21

    # Simulate covered call strategy
    for i in range(len(px)):
        daily_return = returns.iloc[i]

        # Check if we need to roll options (every ~21 trading days)
        if i % days_in_month == 0:
            # Sell new monthly call (5% OTM)
            current_price = px.iloc[i]
            strike_price = current_price * (1 + strike_pct_above)

            # Collect premium (80% of theoretical due to bid-ask spread)
            premium_collected = monthly_premium_rate * 0.8

            # Add premium to today's return
            daily_return += premium_collected

        # Check for assignment risk
        # Look back up to 21 days to find the strike price we sold
        assignment_risk = False
        for back_days in range(min(21, i+1)):
            lookback_idx = i - back_days
            if lookback_idx % days_in_month == 0:  # This was an option selling day
                strike_price = px.iloc[lookback_idx] * (1 + strike_pct_above)
                if px.iloc[i] > strike_price:
                    # Assignment! Stock called away at strike price
                    assignment_return = (strike_price - px.iloc[i-1]) / px.iloc[i-1] if i > 0 else 0
                    daily_return = assignment_return
                    assignment_risk = True
                    break

        strategy_returns.iloc[i] = daily_return

    return strategy_returns

def calculate_strategy_metrics(returns, trade_cost=0.0004):
    """Calculate strategy metrics using validated methodology"""
    # Apply transaction costs (minimal for monthly strategy)
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
        'equity_curve': eq
    }

def run_covered_call_test():
    """Run the covered call strategy test"""
    print("🧪 TESTING: Covered Call Strategy")
    print("=" * 50)

    try:
        # Load data
        px = load_spy_data()
        print(f"✅ Loaded {len(px)} data points")
        print(f"Buy & Hold baseline: {((px.iloc[-1]/px.iloc[0]-1)*100):.1f}%")
        # Test covered call strategy
        cc_returns = simulate_covered_call(px)
        cc_metrics = calculate_strategy_metrics(cc_returns)

        # Test buy-and-hold for comparison (zero transaction costs)
        bh_returns = px.pct_change().fillna(0.0)
        bh_eq = (1.0 + bh_returns).cumprod()
        bh_metrics = {
            'total_return': bh_eq.iloc[-1] - 1,
            'cagr': bh_eq.iloc[-1]**(252/len(bh_eq)) - 1,
            'sharpe': (bh_returns.mean() / (bh_returns.std() + 1e-12)) * np.sqrt(252),
            'max_drawdown': (bh_eq / bh_eq.cummax() - 1).min(),
            'final_equity': bh_eq.iloc[-1]
        }

        # Results
        print("\n📊 COVERED CALL STRATEGY RESULTS:")
        print(f"  Total Return: {cc_metrics['total_return']:.1f}%")
        print(f"  CAGR: {cc_metrics['cagr']:.2f}%")
        print(f"  Sharpe Ratio: {cc_metrics['sharpe']:.2f}")
        print(f"  Max Drawdown: {cc_metrics['max_drawdown']:.2f}")
        print("\n📊 BUY & HOLD COMPARISON:")
        print(f"  Total Return: {bh_metrics['total_return']:.1f}%")
        print(f"  CAGR: {bh_metrics['cagr']:.2f}%")
        print(f"  Sharpe Ratio: {bh_metrics['sharpe']:.2f}")
        print(f"  Max Drawdown: {bh_metrics['max_drawdown']:.2f}")
        # Analysis
        print("\n🎯 ANALYSIS:")
        if cc_metrics['total_return'] > bh_metrics['total_return']:
            print("✅ COVERED CALL BEATS BUY-AND-HOLD!")
            print("💡 This suggests options income strategy has edge")
            print("🔬 Next: Test with different strike prices and expirations")
        else:
            print("❌ COVERED CALL LOSES TO BUY-AND-HOLD")
            print("💡 Even theoretically sound strategies can fail")
            print("🔬 Next: Test statistical arbitrage (pairs trading)")

        print("\n📈 STRATEGY CHARACTERISTICS:")
        print("   • Monthly option selling (5% OTM calls)")
        print("   • 80% of theoretical premium collected")
        print("   • Assignment risk if stock rises above strike")
        print("   • Low transaction costs (monthly trades)")

        print("\n🎯 THEORETICAL FOUNDATION:")
        print("   • Exploits volatility risk premium")
        print("   • Institutional options desks use similar approaches")
        print("   • Lower costs than daily technical strategies")

        return cc_metrics['total_return'] > bh_metrics['total_return']

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = run_covered_call_test()

    print("\n" + "="*50)
    if success:
        print("🎉 SUCCESS: Covered call strategy shows potential edge!")
        print("🔬 RECOMMENDATION: Optimize parameters and test further")
    else:
        print("📊 RESULT: Covered call strategy does not beat buy-and-hold")
        print("🔬 RECOMMENDATION: Test statistical arbitrage next")
    print("="*50)
