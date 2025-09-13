#!/usr/bin/env python3
"""
Get SPY data in smaller chunks to avoid rate limits
"""

import yfinance as yf
import pandas as pd
import time

def get_spy_chunked():
    print("📈 Downloading SPY data in chunks...")

    chunks = [
        ('2020-01-01', '2021-01-01'),
        ('2021-01-01', '2022-01-01'),
        ('2022-01-01', '2023-01-01'),
        ('2023-01-01', '2024-01-01'),
    ]

    all_data = []

    for start, end in chunks:
        print(f"Getting {start} to {end}...")
        try:
            data = yf.download('SPY', start=start, end=end, progress=False)
            if len(data) > 0:
                # Convert to simple format
                data = data.reset_index()
                data.columns = ['Date', 'Adj_Close', 'Close', 'High', 'Low', 'Open', 'Volume']
                data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                all_data.append(data)
                print(f"  ✅ {len(data)} days")
            else:
                print("  ❌ No data")
        except Exception as e:
            print(f"  ❌ Error: {e}")

        time.sleep(2)  # Wait between requests

    if all_data:
        # Combine all chunks
        combined = pd.concat(all_data, ignore_index=True)
        combined['Date'] = pd.to_datetime(combined['Date'])
        combined = combined.drop_duplicates(subset='Date').sort_values('Date')
        combined.set_index('Date', inplace=True)

        combined.to_csv('data/SPY_real_2020_2024.csv')
        print(f"\n✅ Saved {len(combined)} total days to data/SPY_real_2020_2024.csv")

        # Show key stats
        print(f"Date range: {combined.index[0].date()} to {combined.index[-1].date()}")
        print(".2f")
        print(".2f")

        # SPY performance by year
        combined['Year'] = combined.index.year
        annual = combined.groupby('Year')['Close'].agg(['first', 'last'])
        annual['Return'] = (annual['last'] / annual['first'] - 1) * 100

        print("\n📈 SPY Annual Returns:")
        for year, row in annual.iterrows():
            print("4d")

        return True

    return False

if __name__ == "__main__":
    success = get_spy_chunked()
    if success:
        print("\n🎯 Ready to test strategies on real data!")
    else:
        print("\n⚠️  Could not get data - will use synthetic data for now")
