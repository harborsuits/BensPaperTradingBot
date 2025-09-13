#!/usr/bin/env python3
"""
Simple realistic MA strategy test
"""

import pandas as pd
import numpy as np

def test_ma_strategy():
    """Test MA strategy with realistic metrics"""

    print("Testing MA Strategy on Realistic Data")
    print("=" * 40)

    # Create realistic data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')

    price = 322.0
    prices = [price]

    for i in range(1, len(dates)):
        date = dates[i]

        if date.year == 2020:
            if date.month <= 3:
                daily_return = np.random.normal(-0.025, 0.06)
            else:
                daily_return = np.random.normal(0.015, 0.03)
        elif date.year == 2021:
            daily_return = np.random.normal(0.008, 0.025)
        elif date.year == 2022:
            daily_return = np.random.normal(-0.01, 0.035)
        elif date.year == 2023:
            daily_return = np.random.normal(0.0002, 0.015)
        else:
            daily_return = np.random.normal(0.006, 0.02)

        price *= (1 + daily_return)
        price = max(price, 200)
        price = min(price, 550)
        prices.append(price)

    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Close': prices
    })
    df.set_index('Date', inplace=True)

    # Strategy
    df['Fast_MA'] = df['Close'].rolling(10).mean()
    df['Slow_MA'] = df['Close'].rolling(30).mean()
    df['Signal'] = (df['Fast_MA'] > df['Slow_MA']).astype(int)
    df['Returns'] = df['Close'].pct_change()
    df['Strategy'] = df['Signal'].shift(1) * df['Returns']

    # Performance
    strategy_returns = df['Strategy'].dropna()
    buy_hold_returns = df['Returns'].dropna()

    total_strategy = (1 + strategy_returns).prod() - 1
    total_buy_hold = (1 + buy_hold_returns).prod() - 1

    strategy_sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    buy_hold_sharpe = buy_hold_returns.mean() / buy_hold_returns.std() * np.sqrt(252)

    position_days = (df['Signal'] == 1).sum()
    winning_days = ((df['Signal'] == 1) & (df['Strategy'] > 0)).sum()
    win_rate = winning_days / position_days if position_days > 0 else 0

    print("Strategy Performance (10/30 MA):")
    print(".1f")
    print(".1f")
    print(".1f")
    print(".2f")
    print(".1f")
    print("({}/{})".format(winning_days, position_days))

    # Annual performance
    df['Year'] = df.index.year
    annual_strategy = df.groupby('Year')['Strategy'].apply(lambda x: (1 + x.dropna()).prod() - 1)
    annual_buy_hold = df.groupby('Year')['Returns'].apply(lambda x: (1 + x.dropna()).prod() - 1)

    print("\nAnnual Performance:")
    for year in [2020, 2021, 2022, 2023, 2024]:
        if year in annual_strategy.index:
            s_ret = annual_strategy[year]
            b_ret = annual_buy_hold[year] if year in annual_buy_hold.index else 0
            print("4d")

    print("\nRealistic assessment:")
    if total_strategy > total_buy_hold:
        print("Strategy outperformed buy-and-hold!")
    else:
        print("Strategy underperformed buy-and-hold")

    return True

if __name__ == "__main__":
    test_ma_strategy()
