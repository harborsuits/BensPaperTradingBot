#!/usr/bin/env python3
"""
Synthetic Market Generator Test (Standalone)

This standalone script demonstrates the capabilities of our Synthetic Market Generator
without relying on module imports. It directly imports the core classes and functions
from the source files.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from enum import Enum

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import directly from files
from trading_bot.autonomous.synthetic_market_generator import (
    MarketRegimeType, 
    PriceSeriesGenerator,
    SyntheticMarketGenerator
)

from trading_bot.autonomous.synthetic_market_generator_correlations import (
    CorrelationStructure,
    CorrelatedMarketGenerator
)

# Create output directory
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), ".trading_bot", "synthetic_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def test_single_asset_regimes():
    """Test generating different market regimes for a single asset."""
    print("Testing single asset market regimes...")
    
    # Create generator
    generator = SyntheticMarketGenerator(seed=42)
    
    # Generate different regime scenarios
    regimes = [
        MarketRegimeType.BULLISH,
        MarketRegimeType.BEARISH,
        MarketRegimeType.VOLATILE,
        MarketRegimeType.MEAN_REVERTING,
        MarketRegimeType.CRASH,
        MarketRegimeType.RECOVERY
    ]
    
    # Store results
    results = {}
    
    for regime in regimes:
        # Generate the data
        df = generator.generate_regime_scenario(
            regime=regime,
            days=252,
            base_price=100.0,
            volatility=0.015
        )
        
        # Save data
        filename = os.path.join(OUTPUT_DIR, f"regime_{regime.value}.csv")
        df.to_csv(filename)
        
        # Store for plotting
        results[regime.value] = df
        
        # Print summary
        returns = df['close'].pct_change().dropna()
        print(f"  {regime.value}: Volatility: {returns.std()*100:.2f}%, "
              f"Return: {(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:.2f}%")
    
    # Generate plot (if matplotlib is available)
    try:
        plt.figure(figsize=(15, 10))
        
        for i, (regime_name, df) in enumerate(results.items()):
            plt.subplot(3, 2, i+1)
            plt.plot(df.index, df['close'])
            plt.title(f"{regime_name.title()} Market")
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "market_regimes.png"))
        print(f"  Saved regime plots to {os.path.join(OUTPUT_DIR, 'market_regimes.png')}")
    except Exception as e:
        print(f"  Warning: Could not generate plots - {str(e)}")


def test_correlated_assets():
    """Test generating correlated asset price series."""
    print("Testing correlated assets...")
    
    # Create generator
    generator = CorrelatedMarketGenerator(seed=42)
    
    # Define asset list
    assets = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']
    
    # Create correlation structure
    correlation = CorrelationStructure(assets, seed=42)
    
    # Set up sector structure
    sectors = {
        'Equity': ['SPY', 'QQQ', 'IWM'],
        'Safe Haven': ['GLD', 'TLT']
    }
    
    correlation.set_sector_structure(
        sectors,
        intra_sector_correlation=0.8,
        inter_sector_correlation=-0.3  # Negative correlation between equities and safe havens
    )
    
    # Print correlation matrix
    print("\nCorrelation Matrix:")
    print(correlation.get_correlation_dataframe().round(2))
    
    # Generate correlated price data
    price_data = generator.generate_correlated_series(
        correlation,
        days=252,
        volatilities={'SPY': 0.01, 'QQQ': 0.015, 'IWM': 0.012, 'GLD': 0.008, 'TLT': 0.007}
    )
    
    # Print summary for each asset
    print("\nGenerated price series:")
    for asset, df in price_data.items():
        returns = df['close'].pct_change().dropna()
        print(f"  {asset}: {len(df)} days, Volatility: {returns.std()*100:.2f}%, "
              f"Return: {(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:.2f}%")
    
    # Save data
    generator.save_to_csv(price_data, os.path.join(OUTPUT_DIR, "correlated"))
    print(f"  Saved correlated price data to {os.path.join(OUTPUT_DIR, 'correlated')}")
    
    # Calculate actual correlation of returns
    returns = pd.DataFrame({
        asset: df['close'].pct_change().dropna()
        for asset, df in price_data.items()
    })
    
    return_corr = returns.corr()
    print("\nActual Return Correlation:")
    print(return_corr.round(2))


if __name__ == "__main__":
    print("Synthetic Market Generator Test")
    print("===============================")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Run tests
    print("\n1. Testing Single Asset Market Regimes")
    print("-------------------------------------")
    test_single_asset_regimes()
    
    print("\n2. Testing Correlated Assets")
    print("---------------------------")
    test_correlated_assets()
    
    print("\nTests completed! Generated data saved to:")
    print(OUTPUT_DIR)
