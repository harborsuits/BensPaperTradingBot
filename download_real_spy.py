#!/usr/bin/env python3
"""
Download real SPY data for proper backtesting
"""

import yfinance as yf
import pandas as pd
import time

def download_spy_data():
    print("📈 Downloading real SPY data (2020-2024)...")

    # Download in smaller chunks to avoid rate limits
    chunks = [
        ('2020-01-01', '2021-01-01'),
        ('2021-01-01', '2022-01-01'),
        ('2022-01-01', '2023-01-01'),
        ('2023-01-01', '2024-01-01'),
        ('2024-01-01', '2024-12-31')
    ]

    all_data = []
    for start, end in chunks:
        print(f"Downloading {start} to {end}...")
        try:
            data = yf.download('SPY', start=start, end=end, progress=False)
            if len(data) > 0:
                all_data.append(data)
                print(f"  ✅ Got {len(data)} bars")
            else:
                print(f"  ❌ No data for {start}-{end}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

        # Small delay between requests
        time.sleep(1)

    if all_data:
        # Combine all chunks
        combined = pd.concat(all_data)
        combined = combined[~combined.index.duplicated(keep='first')]  # Remove duplicates
        combined.to_csv('data/SPY_real_2020_2024.csv')

        print(f"\n📊 Combined data: {len(combined)} total bars")
        print(f"Date range: {combined.index[0].date()} to {combined.index[-1].date()}")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(f"Saved to: data/SPY_real_2020_2024.csv")

        return True
    else:
        print("❌ Failed to download any data")
        return False

if __name__ == "__main__":
    success = download_spy_data()
    if success:
        print("\n🎯 Ready for realistic backtesting!")
    else:
        print("\n⚠️  Consider manual data download or different source")
