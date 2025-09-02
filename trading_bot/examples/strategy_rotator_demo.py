#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script to demonstrate how the IntegratedStrategyRotator works with various strategies.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading_bot.strategies.integrated_strategy_rotator import IntegratedStrategyRotator, MarketRegime
from trading_bot.strategies.momentum_strategy import MomentumStrategy
from trading_bot.strategies.trend_following_strategy import TrendFollowingStrategy
from trading_bot.strategies.mean_reversion_strategy import MeanReversionStrategy
from trading_bot.backtesting.data_manager import DataManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("StrategyRotatorDemo")

def load_sample_data(symbols=None, start_date=None, end_date=None):
    """
    Load sample market data for demonstration.
    
    In a real-world scenario, this would fetch data from a data provider.
    Here we generate synthetic data for demonstration purposes.
    
    Args:
        symbols: List of symbols to include
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        DataFrame with market data
    """
    if symbols is None:
        symbols = ['SPY', 'QQQ', 'IWM', 'EEM', 'GLD', 'TLT']
    
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365*2)  # 2 years of data
    
    if end_date is None:
        end_date = datetime.now()
    
    # Create a date range
    date_rng = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Create a DataFrame with dates
    data = pd.DataFrame(index=date_rng)
    
    # Generate synthetic price data for each symbol
    for symbol in symbols:
        # Base parameters for the random walk
        drift = 0.0001  # Small positive drift
        volatility = 0.015  # Daily volatility
        
        # Adjust parameters based on symbol to create diversity
        if symbol == 'SPY':
            drift = 0.0002
            volatility = 0.01
        elif symbol == 'QQQ':
            drift = 0.00025
            volatility = 0.015
        elif symbol == 'IWM':
            drift = 0.00015
            volatility = 0.012
        elif symbol == 'EEM':
            drift = 0.0001
            volatility = 0.018
        elif symbol == 'GLD':
            drift = 0.0001
            volatility = 0.01
        elif symbol == 'TLT':
            drift = 0.00005
            volatility = 0.008
        
        # Generate returns with random walk
        returns = np.random.normal(drift, volatility, size=len(date_rng))
        
        # Add some autocorrelation (momentum and mean reversion effects)
        for i in range(1, len(returns)):
            # Momentum effect: 20% of previous return carries over
            momentum_factor = 0.2
            
            # Mean reversion effect: -10% of cumulative excess return
            mean_reversion_speed = 0.1
            excess_return = sum(returns[:i]) - i * drift
            mean_reversion = -mean_reversion_speed * excess_return
            
            returns[i] += momentum_factor * returns[i-1] + mean_reversion
        
        # Create price series from returns
        start_price = 100.0  # Arbitrary starting price
        prices = start_price * np.cumprod(1 + returns)
        
        # Add to DataFrame
        data[symbol] = prices
    
    # Add market regime shifts
    # We'll create different market regimes to demonstrate how the rotator adapts
    
    # Define regimes: bull (first 6 months), volatile (2 months), bear (4 months), recovery (rest)
    regime_periods = [
        (0, 120, 'bull'),  # First 6 months: bull market
        (120, 160, 'high_vol'),  # Next 2 months: high volatility
        (160, 240, 'bear'),  # Next 4 months: bear market
        (240, len(data), 'recovery')  # Rest: recovery
    ]
    
    # Apply regime effects to price data
    for start_idx, end_idx, regime in regime_periods:
        if regime == 'bull':
            # Bull market: higher drift, lower volatility
            for symbol in symbols:
                bull_drift = 0.0004  # Higher drift in bull market
                returns = np.random.normal(bull_drift, 0.01, size=end_idx-start_idx)
                prices = data[symbol].iloc[start_idx] * np.cumprod(1 + returns)
                data.loc[data.index[start_idx:end_idx], symbol] = prices
                
        elif regime == 'high_vol':
            # High volatility: neutral drift, high volatility
            for symbol in symbols:
                high_vol_drift = 0.0  # Neutral drift
                high_vol = 0.025  # Higher volatility
                returns = np.random.normal(high_vol_drift, high_vol, size=end_idx-start_idx)
                prices = data[symbol].iloc[start_idx] * np.cumprod(1 + returns)
                data.loc[data.index[start_idx:end_idx], symbol] = prices
                
        elif regime == 'bear':
            # Bear market: negative drift, high volatility
            for symbol in symbols:
                bear_drift = -0.0003  # Negative drift
                bear_vol = 0.02  # Higher volatility
                returns = np.random.normal(bear_drift, bear_vol, size=end_idx-start_idx)
                prices = data[symbol].iloc[start_idx] * np.cumprod(1 + returns)
                data.loc[data.index[start_idx:end_idx], symbol] = prices
                
        elif regime == 'recovery':
            # Recovery: positive drift, decreasing volatility
            for symbol in symbols:
                recovery_drift = 0.0003  # Positive drift
                recovery_vol = 0.015  # Moderate volatility
                returns = np.random.normal(recovery_drift, recovery_vol, size=end_idx-start_idx)
                prices = data[symbol].iloc[start_idx] * np.cumprod(1 + returns)
                data.loc[data.index[start_idx:end_idx], symbol] = prices
    
    return data

def initialize_strategies():
    """
    Initialize the strategy instances.
    
    Returns:
        Dict of strategy instances
    """
    strategies = {
        'momentum': MomentumStrategy(),
        'trend_following': TrendFollowingStrategy(),
        'mean_reversion': MeanReversionStrategy()
    }
    
    return strategies

def run_demo():
    """Run the strategy rotator demonstration."""
    logger.info("Starting Strategy Rotator Demo")
    
    # Initialize data manager for logging backtest data
    data_manager = DataManager(save_path="data/demo/backtest_history.json")
    
    # Load sample market data
    logger.info("Loading sample market data...")
    market_data = load_sample_data()
    logger.info(f"Loaded data for {len(market_data.columns)} symbols from {market_data.index[0]} to {market_data.index[-1]}")
    
    # Initialize strategies
    logger.info("Initializing strategies...")
    strategies = initialize_strategies()
    logger.info(f"Initialized {len(strategies)} strategies: {', '.join(strategies.keys())}")
    
    # Create the strategy rotator
    logger.info("Creating strategy rotator...")
    rotator = IntegratedStrategyRotator(
        strategies=list(strategies.keys()),
        initial_allocations=None,  # Default to equal weights
        data_dir='data/demo'  # Directory for storing state
    )
    
    # Perform initial rotation
    logger.info("Performing initial strategy rotation...")
    rotation_result = rotator.rotate_strategies(market_data, force_rotation=True)
    
    logger.info(f"Initial allocations: {rotation_result['new_allocations']}")
    logger.info(f"Market regime: {rotation_result['market_regime']}")
    
    # Log initial portfolio snapshot
    data_manager.log_portfolio_snapshot({
        'timestamp': market_data.index[0].isoformat(),
        'total_value': 100000.0,  # Initial capital
        'cash_value': 100000.0,
        'holdings': {},
        'daily_return': 0.0,
        'daily_volatility': 0.0,
        'drawdown': 0.0
    })
    
    # Simulate market evolution over time
    logger.info("Simulating market evolution...")
    
    # We'll run rotation for several periods to see how allocations change
    rotation_periods = 6  # Simulate 6 rotation periods
    allocations_history = []
    
    for period in range(rotation_periods):
        # Move forward in time (simulate passing time)
        current_date = market_data.index[-1] - timedelta(days=180) + timedelta(days=30*period)
        
        # Get data up to current date
        current_data = market_data[market_data.index <= current_date]
        
        if len(current_data) < 60:
            logger.warning("Insufficient data for rotation, skipping period")
            continue
        
        logger.info(f"Period {period+1}: Current date {current_date}")
        
        # Detect market regime
        regime = rotator.detect_market_regime(current_data)
        logger.info(f"Detected market regime: {regime.name}")
        
        # Update strategy performance with the latest data
        for name, strategy in strategies.items():
            strategy.update_performance(current_data)
            
            # Log strategy signals
            signals = strategy.generate_signals(current_data)
            for signal in signals:
                data_manager.log_signal({
                    'timestamp': current_date.isoformat(),
                    'strategy': name,
                    'symbol': signal.get('symbol'),
                    'signal_type': signal.get('type'),
                    'strength': signal.get('strength', 0.0),
                    'direction': signal.get('direction', 'neutral'),
                    'confidence': signal.get('confidence', 0.5),
                    'market_context': {'regime': regime.name}
                })
        
        # Perform strategy rotation
        rotation_result = rotator.rotate_strategies(current_data, force_rotation=True)
        
        # Record allocations
        allocations_history.append({
            'period': period + 1,
            'date': current_date,
            'regime': rotation_result['market_regime'],
            'allocations': rotation_result['new_allocations'].copy()
        })
        
        # Log allocation changes
        data_manager.log_custom_data('strategy_allocation', {
            'timestamp': current_date.isoformat(),
            'regime': rotation_result['market_regime'],
            'allocations': rotation_result['new_allocations']
        })
        
        # Log portfolio snapshot with updated values
        portfolio_value = 100000.0 * (1 + 0.01 * period)  # Simplified for demo
        data_manager.log_portfolio_snapshot({
            'timestamp': current_date.isoformat(),
            'total_value': portfolio_value,
            'cash_value': portfolio_value * 0.2,  # 20% cash
            'holdings': {sym: {'shares': 100, 'value': current_data[sym].iloc[-1] * 100} 
                        for sym in ['SPY', 'QQQ']},
            'daily_return': 0.005,
            'daily_volatility': 0.01,
            'drawdown': 0.0
        })
        
        logger.info(f"New allocations: {rotation_result['new_allocations']}")
    
    # Save all logged data
    data_manager.save()
    logger.info("Saved backtest data to disk")
    
    # Plot allocation changes over time
    plot_allocation_history(allocations_history)
    
    logger.info("Strategy Rotator Demo completed")

def plot_allocation_history(history):
    """
    Plot allocation changes over time.
    
    Args:
        history: List of allocation history entries
    """
    try:
        # Extract data for plotting
        periods = [entry['period'] for entry in history]
        regimes = [entry['regime'] for entry in history]
        
        # Get all strategy names
        strategies = list(history[0]['allocations'].keys())
        
        # Create a DataFrame for easier plotting
        data = pd.DataFrame(index=periods)
        data['regime'] = regimes
        
        for strategy in strategies:
            data[strategy] = [entry['allocations'].get(strategy, 0) for entry in history]
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Stacked bar chart for allocations
        bottom = np.zeros(len(periods))
        for strategy in strategies:
            plt.bar(periods, data[strategy], bottom=bottom, label=strategy)
            bottom += data[strategy]
        
        # Add regime markers
        for i, regime in enumerate(regimes):
            plt.text(periods[i], 103, regime, ha='center', fontsize=9, 
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
        
        plt.title('Strategy Allocations by Period and Market Regime')
        plt.xlabel('Period')
        plt.ylabel('Allocation (%)')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(strategies))
        plt.ylim(0, 110)  # Leave space for regime labels
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save plot
        os.makedirs('data/demo', exist_ok=True)
        plt.savefig('data/demo/allocation_history.png', bbox_inches='tight')
        plt.close()
        
        logger.info("Allocation history plot saved to data/demo/allocation_history.png")
    except Exception as e:
        logger.error(f"Error creating allocation history plot: {str(e)}")

if __name__ == "__main__":
    run_demo() 