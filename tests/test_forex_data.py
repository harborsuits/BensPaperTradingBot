#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test forex data download using yfinance with proper formatting
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
import os

# Configure data directory
data_dir = "data/market_data"
os.makedirs(data_dir, exist_ok=True)

# Forex pairs to download
forex_pairs = {
    "EURUSD": "EURUSD=X", 
    "GBPUSD": "GBPUSD=X", 
    "USDJPY": "USDJPY=X"
}

# Period to download
start_date = "2020-01-01"
end_date = "2022-12-31"
interval = "1d"

print(f"Downloading forex data from {start_date} to {end_date}")

# Download and save data for each pair
for pair_name, ticker in forex_pairs.items():
    print(f"Downloading {pair_name} ({ticker})...")
    
    try:
        # Download data
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True,
            progress=False
        )
        
        if data.empty or len(data) < 10:
            print(f"  Error: Insufficient data received for {pair_name}")
            continue
            
        # Save to CSV
        file_path = os.path.join(data_dir, f"{pair_name}_{interval}.csv")
        data.to_csv(file_path)
        print(f"  Successfully downloaded {len(data)} bars for {pair_name}")
        print(f"  Saved to {file_path}")
        
        # Display sample
        print(f"  Data sample:")
        print(data.head(3))
        print()
        
    except Exception as e:
        print(f"  Error downloading {pair_name}: {str(e)}")

print("Done!")
