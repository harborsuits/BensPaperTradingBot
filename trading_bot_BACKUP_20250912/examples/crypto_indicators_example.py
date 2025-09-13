"""
Example script demonstrating how to use the CryptoIndicatorSuite.

This script shows how to:
1. Load cryptocurrency market data
2. Calculate technical indicators using the suite
3. Generate trading signals
4. Visualize the results
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the CryptoIndicatorSuite
from data.crypto_indicators import CryptoIndicatorSuite, IndicatorConfig

# Set up styling for plots
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12

def load_sample_data(file_path=None):
    """
    Load sample cryptocurrency data.
    
    Args:
        file_path: Path to CSV file with OHLCV data
        
    Returns:
        DataFrame with OHLCV data
    """
    if file_path and os.path.exists(file_path):
        # Load from CSV file
        df = pd.read_csv(file_path)
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        return df
    
    # If no file provided or file doesn't exist, generate synthetic data
    print("No valid data file provided. Generating synthetic data...")
    
    # Generate synthetic data
    np.random.seed(42)
    days = 200
    dates = pd.date_range('2023-01-01', periods=days, freq='D')
    
    # Generate random price series with realistic behavior
    close = np.random.normal(0, 1, days).cumsum() + 10000
    price_multiplier = 0.02  # Percentage price fluctuation
    
    # Generate OHLCV data
    df = pd.DataFrame({
        'open': close * (1 + np.random.normal(0, price_multiplier, days)),
        'high': close * (1 + np.abs(np.random.normal(0, price_multiplier, days))),
        'low': close * (1 - np.abs(np.random.normal(0, price_multiplier, days))),
        'close': close,
        'volume': np.random.lognormal(10, 1, days)
    }, index=dates)
    
    # Ensure high is the highest and low is the lowest
    for i in range(len(df)):
        values = [df.iloc[i]['open'], df.iloc[i]['close']]
        df.iloc[i, df.columns.get_loc('high')] = max(df.iloc[i]['high'], max(values))
        df.iloc[i, df.columns.get_loc('low')] = min(df.iloc[i]['low'], min(values))
    
    return df

def main():
    """Run the crypto indicators example."""
    print("Crypto Indicators Suite Example")
    print("-" * 50)
    
    # Load sample data
    df = load_sample_data()
    print(f"Loaded data with {len(df)} rows")
    
    # Custom indicator configuration
    config = {
        'default_length': 14,
        'fast_rsi_window': 8,  # More responsive for crypto
        'donchian_window': 20,
        'bollinger_dev': 2.5,  # Wider bands for crypto volatility
        'volume_profile_buckets': 10,
        'regime_lookback': 60  # Shorter lookback for crypto
    }
    
    # Initialize indicator suite
    indicator_suite = CryptoIndicatorSuite(config)
    
    # Calculate all indicators
    print("Calculating indicators...")
    df_with_indicators = indicator_suite.add_all_indicators(df)
    
    # Generate trading signals
    print("Generating signals...")
    df_with_signals = indicator_suite.generate_signals(df_with_indicators, strategy='combined')
    
    # Print some statistics
    print("\nIndicator Statistics:")
    print(f"RSI Mean: {df_with_indicators['rsi'].mean():.2f}")
    print(f"Bollinger Width Mean: {df_with_indicators['bollinger_width'].mean():.4f}")
    
    # Count signals
    buy_signals = (df_with_signals['combined_signal'] == 1).sum()
    sell_signals = (df_with_signals['combined_signal'] == -1).sum()
    print(f"\nBuy Signals: {buy_signals}")
    print(f"Sell Signals: {sell_signals}")
    
    # Plot the results
    plot_results(df_with_signals)
    
    print("\nExample completed successfully!")

def plot_results(df):
    """
    Plot the price chart with indicators and signals.
    
    Args:
        df: DataFrame with price data, indicators, and signals
    """
    # Create a figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(4, 1, height_ratios=[2, 1, 1, 1])
    
    # Price chart with signals
    ax1 = plt.subplot(gs[0])
    ax1.plot(df.index, df['close'], label='Close Price')
    ax1.plot(df.index, df['bollinger_upper'], 'r--', alpha=0.3)
    ax1.plot(df.index, df['bollinger_middle'], 'g--', alpha=0.3)
    ax1.plot(df.index, df['bollinger_lower'], 'r--', alpha=0.3)
    
    # Highlight regions with different volatility regimes
    if 'volatility_regime' in df.columns:
        for regime in ['high', 'normal', 'low']:
            mask = df['volatility_regime'] == regime
            if mask.any():
                color = {'high': 'red', 'normal': 'gray', 'low': 'green'}[regime]
                ax1.fill_between(df.index, df['low'].min(), df['high'].max(), 
                                where=mask, color=color, alpha=0.1)
    
    # Plot buy and sell signals
    buy_signals = df[df['combined_signal'] == 1].index
    sell_signals = df[df['combined_signal'] == -1].index
    
    ax1.scatter(buy_signals, df.loc[buy_signals, 'close'], marker='^', color='green', s=100, label='Buy')
    ax1.scatter(sell_signals, df.loc[sell_signals, 'close'], marker='v', color='red', s=100, label='Sell')
    
    ax1.set_title('Price Chart with Bollinger Bands and Signals')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # RSI chart
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.plot(df.index, df['rsi'], label='RSI')
    ax2.plot(df.index, df['fast_rsi'], label='Fast RSI', alpha=0.7)
    ax2.axhline(70, color='r', linestyle='--', alpha=0.3)
    ax2.axhline(30, color='g', linestyle='--', alpha=0.3)
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True)
    
    # MACD chart
    ax3 = plt.subplot(gs[2], sharex=ax1)
    ax3.plot(df.index, df['macd'], label='MACD')
    ax3.plot(df.index, df['macd_signal'], label='Signal Line')
    ax3.bar(df.index, df['macd_hist'], label='Histogram', alpha=0.3, color='gray')
    ax3.set_ylabel('MACD')
    ax3.legend()
    ax3.grid(True)
    
    # Volume chart
    ax4 = plt.subplot(gs[3], sharex=ax1)
    ax4.bar(df.index, df['volume'], label='Volume', alpha=0.3, color='blue')
    ax4.plot(df.index, df['volume'].rolling(20).mean(), label='20-Day Avg', color='orange')
    ax4.set_ylabel('Volume')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('crypto_indicators_example.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main() 