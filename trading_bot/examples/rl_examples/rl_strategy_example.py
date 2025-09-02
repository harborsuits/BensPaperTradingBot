#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RL Strategy Example - Demonstrates using reinforcement learning strategies
in the trading bot system.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import from our trading bot
from trading_bot.strategy.strategy_rotator import StrategyRotator
from trading_bot.common.market_types import MarketRegime
from trading_bot.strategy.rl_strategy import DQNStrategy, MetaLearningStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_sample_market_data(days: int = 365, volatility: float = 0.01):
    """
    Generate sample market data for testing.
    
    Args:
        days: Number of days of data to generate
        volatility: Daily volatility
        
    Returns:
        pd.DataFrame: DataFrame with generated market data
    """
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Starting price
    price = 100.0
    
    # Generate random returns with regime switches
    # First half: Bull market with low volatility
    # Second half: Bear market with high volatility
    half_point = len(dates) // 2
    
    # Create price series with regime changes
    prices = []
    regimes = []
    volumes = []
    
    # Bull market (first third)
    for i in range(len(dates) // 3):
        daily_return = np.random.normal(0.001, volatility)  # Positive drift
        price *= (1 + daily_return)
        prices.append(price)
        regimes.append(MarketRegime.BULL)
        volumes.append(np.random.normal(1000000, 200000))
    
    # Sideways market (second third)
    for i in range(len(dates) // 3, 2 * len(dates) // 3):
        daily_return = np.random.normal(0.0, volatility * 0.8)  # No drift, lower vol
        price *= (1 + daily_return)
        prices.append(price)
        regimes.append(MarketRegime.SIDEWAYS)
        volumes.append(np.random.normal(800000, 150000))
    
    # Bear market (final third)
    for i in range(2 * len(dates) // 3, len(dates)):
        daily_return = np.random.normal(-0.001, volatility * 1.5)  # Negative drift, higher vol
        price *= (1 + daily_return)
        prices.append(price)
        regimes.append(MarketRegime.BEAR)
        volumes.append(np.random.normal(1200000, 300000))
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates[:len(prices)],
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.005)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.005)) for p in prices],
        'close': [p * (1 + np.random.normal(0, 0.001)) for p in prices],
        'volume': volumes,
        'regime': regimes
    })
    
    df.set_index('date', inplace=True)
    return df

def plot_results(df, signals, combined_signal):
    """
    Plot market data and signals.
    
    Args:
        df: DataFrame with market data
        signals: Dict mapping strategy names to signals
        combined_signal: List of combined signals
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price data
    ax1.plot(df.index, df['close'], label='Close Price')
    ax1.set_title('Market Data and Trading Signals')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot signals
    for name, signal_values in signals.items():
        ax2.plot(df.index[-len(signal_values):], signal_values, label=f'{name} Signal', alpha=0.7)
    
    # Plot combined signal
    ax2.plot(df.index[-len(combined_signal):], combined_signal, label='Combined Signal', 
             color='black', linewidth=2)
    
    ax2.set_title('Strategy Signals')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Signal (-1 to 1)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """Run the RL strategy example."""
    # Generate sample market data
    logger.info("Generating sample market data...")
    df = generate_sample_market_data(days=180)
    
    # Create strategy rotator with default strategies (including RL)
    logger.info("Creating strategy rotator...")
    rotator = StrategyRotator()
    
    # Train RL strategies
    logger.info("Training RL strategies...")
    rewards = rotator.train_rl_strategies(df)
    
    # Generate signals for each day in the data
    logger.info("Generating trading signals...")
    signals_by_strategy = {s.name: [] for s in rotator.strategies}
    combined_signals = []
    
    # Use a rolling window to simulate "current" market data
    window_size = 30
    
    for i in range(window_size, len(df)):
        # Extract current window of data
        current_data = df.iloc[i-window_size:i]
        
        # Prepare market data dict for signal generation
        market_data = {
            "prices": current_data['close'].tolist(),
            "volumes": current_data['volume'].tolist(),
            "open": current_data['open'].tolist(),
            "high": current_data['high'].tolist(),
            "low": current_data['low'].tolist()
        }
        
        # Get regime from data
        current_regime = current_data.iloc[-1]['regime']
        rotator.update_market_regime(current_regime)
        
        # Generate signals
        all_signals = rotator.generate_signals(market_data)
        
        # Record signals
        for name, signal in all_signals.items():
            signals_by_strategy[name].append(signal)
        
        # Get combined signal
        combined_signal = rotator.get_combined_signal()
        combined_signals.append(combined_signal)
        
        # Every 20 days, update performance metrics (simplified for example)
        if i % 20 == 0:
            # Simulate performance based on signal quality
            # In a real system, this would be actual trading performance
            performance_data = {}
            
            # Simple performance metric: signal matches price movement
            next_return = (df.iloc[i+1]['close'] / df.iloc[i]['close']) - 1 if i < len(df) - 1 else 0
            
            for name, signal in all_signals.items():
                # Performance is positive if signal matches direction of price movement
                # Signal > 0 and price goes up, or Signal < 0 and price goes down
                if (signal > 0 and next_return > 0) or (signal < 0 and next_return < 0):
                    performance = abs(signal) * abs(next_return) * 10  # Scale up for visibility
                else:
                    performance = -abs(signal) * abs(next_return) * 5  # Penalty for wrong direction
                
                performance_data[name] = performance
            
            # Update strategy performance
            rotator.update_strategy_performance(performance_data)
    
    # Plot results
    logger.info("Plotting results...")
    plot_results(df, signals_by_strategy, combined_signals)
    
    # Print strategy weights
    logger.info("Final strategy weights:")
    weights = rotator.get_strategy_weights()
    for name, weight in weights.items():
        logger.info(f"  {name}: {weight:.2f}")

if __name__ == "__main__":
    main() 