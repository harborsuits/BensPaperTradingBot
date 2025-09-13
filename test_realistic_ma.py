#!/usr/bin/env python3
"""
Test MA strategy on realistic market data
"""

import pandas as pd
import numpy as np

def create_realistic_spy():
    """Create realistic SPY data with different market conditions"""
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')
    price = 322.0
    prices = [price]

    for i in range(1, len(dates)):
        date = dates[i]

        if date.year == 2020:
            if date.month <= 3:
                daily_return = np.random.normal(-0.025, 0.06)  # Bear
            else:
                daily_return = np.random.normal(0.015, 0.03)   # Bull
        elif date.year == 2021:
            daily_return = np.random.normal(0.008, 0.025)      # Bull
        elif date.year == 2022:
            daily_return = np.random.normal(-0.01, 0.035)      # Bear
        elif date.year == 2023:
            daily_return = np.random.normal(0.0002, 0.015)     # Sideways
        else:
            daily_return = np.random.normal(0.006, 0.02)       # Mild bull

        price *= (1 + daily_return)
        price = max(price, 200)
        price = min(price, 550)
        prices.append(price)

    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.015))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.015))) for p in prices],
        'Close': prices,
        'Volume': [np.random.randint(40000000, 120000000) for _ in prices]
    })
    df.set_index('Date', inplace=True)
    return df

def test_ma_strategy():
    """Test MA crossover strategy with realistic metrics"""

    print("🎯 Testing MA Strategy on Realistic Market Data")
    print("=" * 50)

    # Create data
    np.random.seed(42)
    df = create_realistic_spy()
    df.to_csv('data/SPY_realistic_test.csv')

    # Strategy
    fast_ma = 10
    slow_ma = 30
    df['Fast_MA'] = df['Close'].rolling(fast_ma).mean()
    df['Slow_MA'] = df['Close'].rolling(slow_ma).mean()
    df['Signal'] = (df['Fast_MA'] > df['Slow_MA']).astype(int)

    # Returns
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']

    # Costs (0.1% per trade)
    df['Trades'] = df['Signal'].diff().abs()
    df['Costs'] = df['Trades'] * 0.001
    df['Strategy_After_Cost'] = df['Strategy_Returns'] - df['Costs']

    # Performance metrics
    strategy_returns = df['Strategy_After_Cost'].dropna()
    buy_hold_returns = df['Returns'].dropna()

    total_strategy = (1 + strategy_returns).prod() - 1
    total_buy_hold = (1 + buy_hold_returns).prod() - 1

    strategy_sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    buy_hold_sharpe = buy_hold_returns.mean() / buy_hold_returns.std() * np.sqrt(252)

    position_days = (df['Signal'] == 1).sum()
    winning_days = ((df['Signal'] == 1) & (df['Strategy_After_Cost'] > 0)).sum()
    win_rate = winning_days / position_days if position_days > 0 else 0

    cumulative = (1 + strategy_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    print("📊 Strategy Performance (10/30 MA Crossover):"    print(f"  Total Return: {total_strategy:.1%}")
    print(f"  Buy & Hold SPY: {total_buy_hold:.1%}")
    print(f"  Outperformance: {(total_strategy - total_buy_hold):.1%}")
    print(f"  Sharpe Ratio: {strategy_sharpe:.2f}")
    print(f"  Win Rate: {win_rate:.1%} ({winning_days}/{position_days} days)")
    print(f"  Max Drawdown: {max_drawdown:.1%}")

    print("
📈 Annual Performance:"    df['Year'] = df.index.year
    annual_strategy = df.groupby('Year')['Strategy_After_Cost'].apply(lambda x: (1 + x.dropna()).prod() - 1)
    annual_buy_hold = df.groupby('Year')['Returns'].apply(lambda x: (1 + x.dropna()).prod() - 1)

    for year in sorted(df['Year'].unique()):
        if year in annual_strategy.index and year in annual_buy_hold.index:
            s_ret = annual_strategy[year]
            b_ret = annual_buy_hold[year]
            print("4d")

    print("
✅ Realistic backtest completed!"    print("💡 This demonstrates proper risk-adjusted performance metrics"
    return True

if __name__ == "__main__":
    test_ma_strategy()
