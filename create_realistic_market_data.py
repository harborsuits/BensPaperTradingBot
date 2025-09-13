#!/usr/bin/env python3
"""
Create realistic SPY market data with bull, bear, and sideways periods
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_realistic_spy_data():
    """Create SPY data that mimics real market conditions"""

    # Create date range from 2020-2024
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='B')  # Business days only

    # Base SPY price (roughly matches real SPY levels)
    base_price = 300.0

    # Create price series with different market regimes
    prices = []
    current_price = base_price

    for i, date in enumerate(dates):
        # Define market regimes by year
        if date.year == 2020:
            # 2020: High volatility, COVID crash and recovery
            if date.month <= 3:
                # Bear market (COVID crash)
                daily_return = np.random.normal(-0.02, 0.04)  # Mean -2%, vol 4%
            else:
                # Bull market recovery
                daily_return = np.random.normal(0.015, 0.025)  # Mean 1.5%, vol 2.5%
        elif date.year == 2021:
            # 2021: Strong bull market
            daily_return = np.random.normal(0.01, 0.015)  # Mean 1%, vol 1.5%
        elif date.year == 2022:
            # 2022: Bear market
            daily_return = np.random.normal(-0.005, 0.02)  # Mean -0.5%, vol 2%
        elif date.year == 2023:
            # 2023: Sideways/choppy
            daily_return = np.random.normal(0.0005, 0.012)  # Mean 0.05%, vol 1.2%
        else:  # 2024
            # 2024: Mild bull
            daily_return = np.random.normal(0.008, 0.018)  # Mean 0.8%, vol 1.8%

        # Apply return and ensure reasonable bounds
        current_price *= (1 + daily_return)
        current_price = max(current_price, 200)  # Floor at ~$200
        current_price = min(current_price, 600)  # Cap at ~$600

        prices.append(current_price)

    # Create OHLC data
    np.random.seed(42)  # For reproducible results
    data = []

    for i, (date, close) in enumerate(zip(dates, prices)):
        # Create realistic OHLC from close price
        volatility = 0.02  # 2% daily volatility
        high = close * (1 + abs(np.random.normal(0, volatility)))
        low = close * (1 - abs(np.random.normal(0, volatility)))
        open_price = data[-1]['Close'] if data else close * (1 + np.random.normal(0, 0.005))
        volume = np.random.randint(50000000, 150000000)  # Realistic volume

        data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': volume
        })

    # Create DataFrame
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    return df

def analyze_market_conditions(df):
    """Analyze the created market data"""
    print("📊 Market Conditions Analysis:")
    print(f"Total trading days: {len(df)}")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")

    # Annual returns
    df['Year'] = df.index.year
    annual_returns = df.groupby('Year')['Close'].agg(['first', 'last'])
    annual_returns['Return'] = (annual_returns['last'] / annual_returns['first'] - 1) * 100

    print("\n📈 Annual Performance:")
    for year, row in annual_returns.iterrows():
        print(f"  {year}: {row['Return']:.2f}%")

    # Overall performance
    total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
    print("\n🌍 Overall Performance:")
    print(f"  Total Return: {total_return:.1f}%")
    print(f"  Start Price: ${df['Close'].iloc[0]:.2f}")
    print(f"  End Price: ${df['Close'].iloc[-1]:.2f}")

    # Buy and hold return
    buy_hold_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
    print(f"  Buy & Hold Return: {buy_hold_return:.1f}%")

def main():
    print("🎯 Creating realistic SPY market data (2020-2024)...")

    df = create_realistic_spy_data()

    # Save to CSV
    df.to_csv('data/SPY_realistic_2020_2024.csv')
    print(f"✅ Saved {len(df)} trading days to data/SPY_realistic_2020_2024.csv")

    # Analyze the data
    analyze_market_conditions(df)

    print("\n🚀 Ready for realistic strategy testing!")
    print("💡 This data includes:")
    print("   • 2020: High volatility crash and recovery")
    print("   • 2021: Strong bull market")
    print("   • 2022: Bear market")
    print("   • 2023: Sideways/choppy market")
    print("   • 2024: Mild bull market")

if __name__ == "__main__":
    main()