#!/usr/bin/env python3
"""
Download SPY data for backtesting
"""

import yfinance as yf
import pandas as pd

print('Downloading SPY data...')
try:
    spy = yf.download('SPY', start='2023-01-01', end='2024-01-01', progress=False)
    if len(spy) > 0:
        spy.to_csv('data/SPY_real.csv')
        print(f'Successfully downloaded {len(spy)} SPY bars')
        print(f'Date range: {spy.index[0].date()} to {spy.index[-1].date()}')
        print(f'Latest price: ${spy.iloc[-1].Close:.2f}')
        print(f'Average volume: {spy.Volume.mean():.0f}')
    else:
        print('No data downloaded')
except Exception as e:
    print(f'Error: {e}')
