#!/usr/bin/env python3
"""
Synthetic Market Generator Test

This script demonstrates the capabilities of the Synthetic Market Generator
and Correlated Market Generator components, showing how they can be used
to create realistic test scenarios for our trading strategies.

The generated data can be used with our A/B Testing Framework to evaluate
strategy performance under different market conditions.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import our synthetic market generators
from trading_bot.autonomous.synthetic_market_generator import (
    get_synthetic_market_generator, MarketRegimeType
)
from trading_bot.autonomous.synthetic_market_generator_correlations import (
    get_correlated_market_generator, CorrelationStructure
)

# Create output directory
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), ".trading_bot", "synthetic_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def test_single_asset_regimes():
    """Test generating different market regimes for a single asset."""
    # Get generator
    generator = get_synthetic_market_generator()
    
    # Generate different regime scenarios
    regimes = [
        MarketRegimeType.BULLISH,
        MarketRegimeType.BEARISH,
        MarketRegimeType.VOLATILE,
        MarketRegimeType.MEAN_REVERTING,
        MarketRegimeType.CRASH,
        MarketRegimeType.RECOVERY
    ]
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    for i, regime in enumerate(regimes):
        # Generate the data
        df = generator.generate_regime_scenario(
            regime=regime,
            days=252,
            base_price=100.0,
            volatility=0.015
        )
        
        # Plot
        plt.subplot(3, 2, i+1)
        plt.plot(df.index, df['close'])
        plt.title(f"{regime.value.title()} Market")
        plt.grid(True)
        
        # Save data
        filename = os.path.join(OUTPUT_DIR, f"regime_{regime.value}.csv")
        df.to_csv(filename)
        
        # Print summary
        returns = df['close'].pct_change().dropna()
        print(f"{regime.value}: Volatility: {returns.std()*100:.2f}%, "
              f"Return: {(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:.2f}%")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "market_regimes.png"))
    print(f"Saved regime plots to {os.path.join(OUTPUT_DIR, 'market_regimes.png')}")


def test_correlated_assets():
    """Test generating correlated asset price series."""
    # Get generator
    generator = get_correlated_market_generator()
    
    # Define asset list
    assets = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']
    
    # Create correlation structure
    correlation = CorrelationStructure(assets)
    
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
    
    # Apply some random variation
    correlation.apply_random_variation(0.1)
    
    # Print correlation matrix
    print("\nCorrelation Matrix:")
    print(correlation.get_correlation_dataframe().round(2))
    
    # Generate correlated price data
    price_data = generator.generate_correlated_series(
        correlation,
        days=252,
        volatilities={'SPY': 0.01, 'QQQ': 0.015, 'IWM': 0.012, 'GLD': 0.008, 'TLT': 0.007}
    )
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Normalize prices for better comparison
    normalized_data = {}
    for asset, df in price_data.items():
        normalized_data[asset] = df['close'] / df['close'].iloc[0] * 100
    
    # Convert to DataFrame for easy plotting
    normalized_df = pd.DataFrame(normalized_data)
    
    # Plot normalized prices
    plt.subplot(2, 1, 1)
    for asset in assets:
        plt.plot(normalized_df.index, normalized_df[asset], label=asset)
    plt.title("Correlated Asset Prices (Normalized)")
    plt.legend()
    plt.grid(True)
    
    # Calculate correlation of returns
    returns = pd.DataFrame({
        asset: df['close'].pct_change().dropna()
        for asset, df in price_data.items()
    })
    
    return_corr = returns.corr()
    
    # Plot return correlation heatmap
    plt.subplot(2, 1, 2)
    plt.imshow(return_corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Return Correlation Matrix")
    
    # Add correlation values
    for i in range(len(return_corr)):
        for j in range(len(return_corr)):
            plt.text(j, i, f"{return_corr.iloc[i, j]:.2f}",
                     ha="center", va="center", color="black" if abs(return_corr.iloc[i, j]) < 0.7 else "white")
    
    plt.xticks(range(len(return_corr)), return_corr.columns, rotation=45)
    plt.yticks(range(len(return_corr)), return_corr.index)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlated_assets.png"))
    print(f"Saved correlation plots to {os.path.join(OUTPUT_DIR, 'correlated_assets.png')}")
    
    # Save data
    generator.save_to_csv(price_data, os.path.join(OUTPUT_DIR, "correlated"))
    print(f"Saved correlated price data to {os.path.join(OUTPUT_DIR, 'correlated')}")


def test_correlation_regime_change():
    """Test generating price series with changing correlation regimes."""
    # Get generator
    generator = get_correlated_market_generator()
    
    # Define asset list
    assets = ['SPY', 'QQQ', 'GLD', 'TLT']
    
    # Create initial correlation structure (normal market)
    initial_correlation = CorrelationStructure(assets)
    
    # Set initial correlations
    initial_correlation.set_pairwise_correlation('SPY', 'QQQ', 0.8)  # Equity correlated
    initial_correlation.set_pairwise_correlation('SPY', 'GLD', 0.1)  # Low correlation with gold
    initial_correlation.set_pairwise_correlation('SPY', 'TLT', -0.2)  # Slight negative with bonds
    initial_correlation.set_pairwise_correlation('QQQ', 'GLD', 0.1)
    initial_correlation.set_pairwise_correlation('QQQ', 'TLT', -0.2)
    initial_correlation.set_pairwise_correlation('GLD', 'TLT', 0.3)   # Moderate safe haven correlation
    
    # Create target correlation structure (crisis regime)
    target_correlation = CorrelationStructure(assets)
    
    # In crisis, correlations often approach 1 (everything moves together)
    target_correlation.set_pairwise_correlation('SPY', 'QQQ', 0.95)  # Even stronger equity correlation
    target_correlation.set_pairwise_correlation('SPY', 'GLD', -0.7)  # Strong negative (flight to safety)
    target_correlation.set_pairwise_correlation('SPY', 'TLT', -0.8)  # Strong negative (flight to safety)
    target_correlation.set_pairwise_correlation('QQQ', 'GLD', -0.7)
    target_correlation.set_pairwise_correlation('QQQ', 'TLT', -0.8)
    target_correlation.set_pairwise_correlation('GLD', 'TLT', 0.8)   # Strong safe haven correlation
    
    # Generate price data with changing correlation regime
    price_data = generator.generate_changing_correlation_regime(
        initial_correlation,
        target_correlation,
        days=252,
        regime_change_start=126,  # Start change halfway through
        regime_change_duration=20,  # Transition over 20 days
        volatilities={'SPY': 0.01, 'QQQ': 0.015, 'GLD': 0.008, 'TLT': 0.007}
    )
    
    # Plot results
    plt.figure(figsize=(15, 12))
    
    # Normalize prices for better comparison
    normalized_data = {}
    for asset, df in price_data.items():
        normalized_data[asset] = df['close'] / df['close'].iloc[0] * 100
    
    # Convert to DataFrame for easy plotting
    normalized_df = pd.DataFrame(normalized_data)
    
    # Plot normalized prices
    plt.subplot(3, 1, 1)
    for asset in assets:
        plt.plot(normalized_df.index, normalized_df[asset], label=asset)
    
    # Add vertical lines showing regime change period
    change_start = normalized_df.index[126]
    change_end = normalized_df.index[126 + 20]
    plt.axvline(change_start, color='r', linestyle='--', alpha=0.5, label='Regime Change Start')
    plt.axvline(change_end, color='g', linestyle='--', alpha=0.5, label='Regime Change End')
    
    plt.title("Asset Prices During Correlation Regime Change")
    plt.legend()
    plt.grid(True)
    
    # Calculate rolling 30-day correlations
    window = 30
    rolling_corr = pd.DataFrame(index=normalized_df.index[window-1:])
    
    pairs = [
        ('SPY', 'GLD', 'Equity-Gold Correlation'),
        ('SPY', 'TLT', 'Equity-Bond Correlation')
    ]
    
    for asset1, asset2, label in pairs:
        corr_values = []
        for i in range(window, len(normalized_df)+1):
            corr = normalized_df[asset1][i-window:i].corr(normalized_df[asset2][i-window:i])
            corr_values.append(corr)
        rolling_corr[label] = corr_values
    
    # Plot rolling correlations
    plt.subplot(3, 1, 2)
    for column in rolling_corr.columns:
        plt.plot(rolling_corr.index, rolling_corr[column], label=column)
    
    # Add the same vertical lines
    plt.axvline(change_start, color='r', linestyle='--', alpha=0.5)
    plt.axvline(change_end, color='g', linestyle='--', alpha=0.5)
    
    plt.title(f"{window}-Day Rolling Correlations")
    plt.legend()
    plt.grid(True)
    plt.ylim(-1, 1)
    
    # Plot rolling volatility (30-day)
    rolling_vol = pd.DataFrame(index=normalized_df.index[window-1:])
    
    for asset in assets:
        vol_values = []
        for i in range(window, len(normalized_df)+1):
            vol = normalized_df[asset][i-window:i].pct_change().std() * np.sqrt(252)  # Annualized
            vol_values.append(vol)
        rolling_vol[f"{asset} Volatility"] = vol_values
    
    # Plot rolling volatility
    plt.subplot(3, 1, 3)
    for column in rolling_vol.columns:
        plt.plot(rolling_vol.index, rolling_vol[column], label=column)
    
    # Add the same vertical lines
    plt.axvline(change_start, color='r', linestyle='--', alpha=0.5)
    plt.axvline(change_end, color='g', linestyle='--', alpha=0.5)
    
    plt.title(f"{window}-Day Rolling Volatility (Annualized)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_regime_change.png"))
    print(f"Saved regime change plots to {os.path.join(OUTPUT_DIR, 'correlation_regime_change.png')}")
    
    # Save data
    generator.save_to_csv(price_data, os.path.join(OUTPUT_DIR, "regime_change"))
    print(f"Saved regime change data to {os.path.join(OUTPUT_DIR, 'regime_change')}")


def test_market_regime_with_correlations():
    """Test applying market regimes to correlated assets."""
    # Get generator
    generator = get_correlated_market_generator()
    
    # Define asset list
    assets = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']
    
    # Create correlation structure
    correlation = CorrelationStructure(assets)
    
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
    
    # Generate baseline correlated price data
    price_data = generator.generate_correlated_series(
        correlation,
        days=252,
        volatilities={'SPY': 0.01, 'QQQ': 0.015, 'IWM': 0.012, 'GLD': 0.008, 'TLT': 0.007}
    )
    
    # Apply different market regimes
    regimes = [
        (MarketRegimeType.BULLISH, "Bullish"),
        (MarketRegimeType.BEARISH, "Bearish"),
        (MarketRegimeType.VOLATILE, "Volatile"),
        (MarketRegimeType.CRASH, "Crash")
    ]
    
    # Plot results
    plt.figure(figsize=(15, 15))
    
    for i, (regime, label) in enumerate(regimes):
        # Apply regime
        regime_data = generator.apply_market_regime(
            price_data, 
            regime,
            regime_params={'trend_strength': 0.002, 'volatility_factor': 2.5}
        )
        
        # Normalize for comparison
        normalized_data = {}
        for asset, df in regime_data.items():
            normalized_data[asset] = df['close'] / df['close'].iloc[0] * 100
        
        # Convert to DataFrame
        normalized_df = pd.DataFrame(normalized_data)
        
        # Plot
        plt.subplot(2, 2, i+1)
        for asset in assets:
            plt.plot(normalized_df.index, normalized_df[asset], label=asset)
        plt.title(f"{label} Market Regime")
        plt.legend()
        plt.grid(True)
        
        # Save data
        generator.save_to_csv(
            regime_data, 
            os.path.join(OUTPUT_DIR, f"regime_{regime.value}")
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "market_regimes_correlated.png"))
    print(f"Saved correlated regime plots to {os.path.join(OUTPUT_DIR, 'market_regimes_correlated.png')}")


if __name__ == "__main__":
    print("Synthetic Market Generator Test")
    print("-------------------------------")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Run tests
    print("\n1. Testing Single Asset Market Regimes")
    print("-------------------------------------")
    test_single_asset_regimes()
    
    print("\n2. Testing Correlated Assets")
    print("---------------------------")
    test_correlated_assets()
    
    print("\n3. Testing Correlation Regime Change")
    print("----------------------------------")
    test_correlation_regime_change()
    
    print("\n4. Testing Market Regimes with Correlations")
    print("-----------------------------------------")
    test_market_regime_with_correlations()
    
    print("\nAll tests completed successfully!")
