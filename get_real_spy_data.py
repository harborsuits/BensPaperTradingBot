#!/usr/bin/env python3
"""
Get real SPY data from 2020-2024 for proper backtesting
"""

import yfinance as yf
import pandas as pd

def get_real_spy_data():
    print("📈 Downloading real SPY data (2020-2024)...")

    try:
        # Download SPY data
        spy = yf.download('SPY', start='2020-01-01', end='2024-12-31', progress=False)

        if len(spy) == 0:
            print("❌ No data downloaded")
            return False

        # Clean up the data (yfinance returns MultiIndex columns)
        spy = spy.reset_index()
        spy.columns = ['Date', 'Adj_Close', 'Close', 'High', 'Low', 'Open', 'Volume']

        # Keep only the columns we need
        spy = spy[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        spy['Date'] = pd.to_datetime(spy['Date'])
        spy.set_index('Date', inplace=True)

        # Save to CSV
        spy.to_csv('data/SPY_2020_2024.csv')
        print(f"✅ Downloaded {len(spy)} trading days")

        # Show summary
        print(f"Date range: {spy.index[0].date()} to {spy.index[-1].date()}")
        print(".2f")
        print(".2f")

        # Calculate annual performance
        spy['Year'] = spy.index.year
        annual = spy.groupby('Year')['Close'].agg(['first', 'last'])
        annual['Return'] = (annual['last'] / annual['first'] - 1) * 100

        print("\n📈 SPY Annual Returns:")
        for year, row in annual.iterrows():
            print("4d")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = get_real_spy_data()
    if success:
        print("\n🎯 Real SPY data ready for backtesting!")
    else:
        print("\n⚠️  Failed to get SPY data")
