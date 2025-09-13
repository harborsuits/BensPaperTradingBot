#!/usr/bin/env python3
"""
Download SPY data using Alpaca adapter
"""

import pandas as pd
from trading_bot.data_sources.alpaca_adapter import get_alpaca_data_sync

def download_spy_alpaca():
    print("📈 Downloading SPY data from Alpaca (2020-2024)...")

    # Download data in chunks
    years = [
        ("2020-01-01T00:00:00Z", "2021-01-01T00:00:00Z"),
        ("2021-01-01T00:00:00Z", "2022-01-01T00:00:00Z"),
        ("2022-01-01T00:00:00Z", "2023-01-01T00:00:00Z"),
        ("2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z"),
        ("2024-01-01T00:00:00Z", "2024-12-31T00:00:00Z")
    ]

    all_data = []

    for start, end in years:
        print(f"Downloading {start[:10]} to {end[:10]}...")
        try:
            data = get_alpaca_data_sync("SPY", start, end, "1Day")
            if data:
                all_data.extend(data)
                print(f"  ✅ Got {len(data)} bars")
            else:
                print("  ❌ No data received")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    if all_data:
        # Create DataFrame and save
        df = pd.DataFrame(all_data)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.sort_index()

        # Save to CSV
        df.to_csv('data/SPY_alpaca_2020_2024.csv')
        print(f"\n✅ Saved {len(df)} total trading days to data/SPY_alpaca_2020_2024.csv")

        # Show summary
        print("
📊 Data Summary:"        print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")

        # Calculate annual returns
        df['Year'] = df.index.year
        annual_returns = df.groupby('Year')['Close'].agg(['first', 'last'])
        annual_returns['Return'] = (annual_returns['last'] / annual_returns['first'] - 1) * 100

        print("
📈 Annual Performance:"        for year, row in annual_returns.iterrows():
            print("2d")

        return True
    else:
        print("❌ Failed to download any data")
        return False

if __name__ == "__main__":
    success = download_spy_alpaca()
    if success:
        print("\n🎯 Ready for realistic Alpaca backtesting!")
    else:
        print("\n⚠️  Alpaca data download failed")