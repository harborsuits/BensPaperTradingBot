#!/usr/bin/env python3
"""
Simplified Synthetic Market Generator Test

This script demonstrates the core functionality of our Synthetic Market Generator
without external visualization dependencies. It shows how we can generate different
market regimes and correlated asset price movements for testing our trading strategies.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enum import Enum

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import directly from files
try:
    from trading_bot.autonomous.synthetic_market_generator import (
        MarketRegimeType, 
        PriceSeriesGenerator,
        SyntheticMarketGenerator
    )

    from trading_bot.autonomous.synthetic_market_generator_correlations import (
        CorrelationStructure,
        CorrelatedMarketGenerator
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Attempting direct imports...")
    
    # If the module approach fails, define the classes directly
    # This is just for demonstration and would need to be updated
    # with the actual implementations
    
    class MarketRegimeType(str, Enum):
        """Types of market regimes for synthetic data generation."""
        BULLISH = "bullish"
        BEARISH = "bearish"
        VOLATILE = "volatile"
        MEAN_REVERTING = "mean_reverting"
        CRASH = "crash"
        RECOVERY = "recovery"
    
    class SyntheticMarketGenerator:
        def __init__(self, seed=None):
            print("Using fallback SyntheticMarketGenerator")
            self.seed = seed
    
    class CorrelationStructure:
        def __init__(self, assets, seed=None):
            print("Using fallback CorrelationStructure")
            self.assets = assets
            self.seed = seed
    
    class CorrelatedMarketGenerator:
        def __init__(self, seed=None):
            print("Using fallback CorrelatedMarketGenerator")
            self.seed = seed

# Create output directory
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), ".trading_bot", "synthetic_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def test_single_asset_generation():
    """Test generating a simple price series."""
    print("Testing basic price series generation...")
    
    try:
        # Create generator
        generator = PriceSeriesGenerator(base_price=100.0, volatility=0.01, seed=42)
        
        # Generate a random walk
        df = generator.generate_random_walk(days=252)
        
        # Calculate some statistics
        returns = df['close'].pct_change().dropna()
        annualized_volatility = returns.std() * np.sqrt(252)
        total_return = df['close'].iloc[-1] / df['close'].iloc[0] - 1
        
        print(f"  Generated {len(df)} days of price data")
        print(f"  Starting price: ${df['close'].iloc[0]:.2f}")
        print(f"  Ending price: ${df['close'].iloc[-1]:.2f}")
        print(f"  Total return: {total_return*100:.2f}%")
        print(f"  Annualized volatility: {annualized_volatility*100:.2f}%")
        
        # Save data
        filename = os.path.join(OUTPUT_DIR, "basic_price_series.csv")
        df.to_csv(filename)
        print(f"  Saved to {filename}")
        
        return True
    except Exception as e:
        print(f"  Error in test_single_asset_generation: {e}")
        return False


def test_market_regimes():
    """Test generating different market regimes."""
    print("Testing market regime generation...")
    
    try:
        # Create generator
        generator = SyntheticMarketGenerator(seed=42)
        
        # Generate different regimes
        regimes = [
            MarketRegimeType.BULLISH,
            MarketRegimeType.BEARISH,
            MarketRegimeType.VOLATILE,
            MarketRegimeType.MEAN_REVERTING,
            MarketRegimeType.CRASH,
            MarketRegimeType.RECOVERY
        ]
        
        for regime in regimes:
            # Generate the regime
            df = generator.generate_regime_scenario(
                regime=regime,
                days=252,
                base_price=100.0,
                volatility=0.015
            )
            
            # Calculate statistics
            returns = df['close'].pct_change().dropna()
            total_return = df['close'].iloc[-1] / df['close'].iloc[0] - 1
            
            # Print summary
            print(f"  {regime.value}: Days: {len(df)}, "
                  f"Return: {total_return*100:.2f}%, "
                  f"Volatility: {returns.std()*100:.2f}%")
            
            # Save data
            filename = os.path.join(OUTPUT_DIR, f"regime_{regime.value}.csv")
            df.to_csv(filename)
        
        print(f"  Saved regime data to {OUTPUT_DIR}")
        return True
    except Exception as e:
        print(f"  Error in test_market_regimes: {e}")
        return False


def test_correlation_structure():
    """Test creating correlation structures."""
    print("Testing correlation structure creation...")
    
    try:
        # Define assets
        assets = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']
        
        # Create correlation structure
        correlation = CorrelationStructure(assets, seed=42)
        
        # Define sectors
        sectors = {
            'Equity': ['SPY', 'QQQ', 'IWM'],
            'Safe Haven': ['GLD', 'TLT']
        }
        
        # Set up sector structure
        correlation.set_sector_structure(
            sectors,
            intra_sector_correlation=0.8,
            inter_sector_correlation=-0.3
        )
        
        # Get correlation matrix
        corr_matrix = correlation.get_correlation_dataframe()
        
        # Print correlation matrix
        print("\nCorrelation Matrix:")
        print(corr_matrix.round(2))
        
        return True
    except Exception as e:
        print(f"  Error in test_correlation_structure: {e}")
        return False


def test_correlated_assets():
    """Test generating correlated asset prices."""
    print("Testing correlated asset generation...")
    
    try:
        # Create generator
        generator = CorrelatedMarketGenerator(seed=42)
        
        # Define assets
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
            inter_sector_correlation=-0.3
        )
        
        # Generate correlated price data
        price_data = generator.generate_correlated_series(
            correlation,
            days=252,
            volatilities={'SPY': 0.01, 'QQQ': 0.015, 'IWM': 0.012, 'GLD': 0.008, 'TLT': 0.007}
        )
        
        # Print summary for each asset
        print("\nGenerated correlated price series:")
        for asset, df in price_data.items():
            returns = df['close'].pct_change().dropna()
            total_return = df['close'].iloc[-1] / df['close'].iloc[0] - 1
            
            print(f"  {asset}: Days: {len(df)}, "
                  f"Return: {total_return*100:.2f}%, "
                  f"Volatility: {returns.std()*100:.2f}%")
        
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
        
        return True
    except Exception as e:
        print(f"  Error in test_correlated_assets: {e}")
        return False


if __name__ == "__main__":
    print("Synthetic Market Generator Test (Simple Version)")
    print("===============================================")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Run tests
    tests = [
        ("Basic Price Series Generation", test_single_asset_generation),
        ("Market Regime Generation", test_market_regimes),
        ("Correlation Structure Creation", test_correlation_structure),
        ("Correlated Asset Generation", test_correlated_assets)
    ]
    
    results = []
    
    for name, test_func in tests:
        print(f"\n{name}")
        print("-" * len(name))
        
        success = test_func()
        results.append((name, success))
    
    # Print summary
    print("\nTest Results Summary")
    print("===================")
    
    all_passed = True
    for name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{name}: {status}")
        all_passed = all_passed and success
    
    print("\nAll tests passed!" if all_passed else "\nSome tests failed!")
    
    print(f"\nGenerated data saved to: {OUTPUT_DIR}")
