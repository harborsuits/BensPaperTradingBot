#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithmic Meta-Strategy Test Script
Tests the ML-driven ensemble meta-strategy for forex trading
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Any

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import trading components
from trading_bot.strategies.strategy_factory import StrategyFactory
from trading_bot.strategies.forex.algorithmic_meta_strategy import AlgorithmicMetaStrategy
from trading_bot.strategies.base.forex_base import MarketRegime, ForexBaseStrategy
from trading_bot.utils.event_bus import EventBus

# Create directories for test outputs if needed
os.makedirs("test_results", exist_ok=True)

def generate_synthetic_data(symbols: List[str], days: int = 100, freq: str = '1H') -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic OHLCV data for testing
    
    Args:
        symbols: List of forex symbols
        days: Number of days of data
        freq: Data frequency
        
    Returns:
        Dictionary of symbol -> OHLCV DataFrame
    """
    data = {}
    
    # Generate timestamps
    end_time = pd.Timestamp.now(tz=pytz.UTC)
    start_time = end_time - pd.Timedelta(days=days)
    timestamps = pd.date_range(start=start_time, end=end_time, freq=freq)
    
    for symbol in symbols:
        # Generate base price and random walk
        base_price = 1.0 if 'JPY' in symbol else 100.0
        returns = np.random.normal(0, 0.0005, size=len(timestamps))
        
        # Add some trends, ranges, and volatility patterns
        # This will help simulate different market regimes
        trends = []
        
        # Add a strong uptrend
        trend_period = len(timestamps) // 5
        uptrend = np.linspace(0, 0.05, trend_period)
        trends.extend(uptrend)
        
        # Add a ranging period
        range_period = len(timestamps) // 4
        ranging = np.random.normal(0, 0.0002, size=range_period)
        trends.extend(ranging)
        
        # Add a downtrend
        downtrend_period = len(timestamps) // 5
        downtrend = np.linspace(0, -0.04, downtrend_period)
        trends.extend(downtrend)
        
        # Add volatile period
        volatile_period = len(timestamps) // 4
        volatile = np.random.normal(0, 0.001, size=volatile_period)
        trends.extend(volatile)
        
        # Fill remaining with random walk
        remaining = len(timestamps) - len(trends)
        if remaining > 0:
            random_walk = np.random.normal(0, 0.0003, size=remaining)
            trends.extend(random_walk)
        elif remaining < 0:
            # Trim if we have too many
            trends = trends[:len(timestamps)]
        
        # Combine random returns with trends
        combined_returns = returns + np.array(trends)
        
        # Generate prices with cumulative returns
        close_prices = base_price * (1 + np.cumsum(combined_returns))
        
        # Generate OHLCV data
        high_prices = close_prices * (1 + np.random.uniform(0, 0.002, size=len(timestamps)))
        low_prices = close_prices * (1 - np.random.uniform(0, 0.002, size=len(timestamps)))
        open_prices = close_prices * (1 + np.random.normal(0, 0.001, size=len(timestamps)))
        
        # Ensure low < open, close < high
        for i in range(len(timestamps)):
            min_price = min(open_prices[i], close_prices[i])
            max_price = max(open_prices[i], close_prices[i])
            low_prices[i] = min(low_prices[i], min_price * 0.999)
            high_prices[i] = max(high_prices[i], max_price * 1.001)
        
        # Generate volume
        volume = np.random.lognormal(10, 1, size=len(timestamps))
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        }, index=timestamps)
        
        data[symbol] = df
    
    return data

def generate_sub_strategy_performance_history() -> Dict[str, Dict[str, float]]:
    """
    Generate synthetic performance metrics for sub-strategies
    
    Returns:
        Dictionary of strategy_type -> performance metrics
    """
    strategies = [
        'trend_following', 'counter_trend', 'breakout',
        'range_trading', 'momentum', 'retracement'
    ]
    
    performance = {}
    
    for strategy in strategies:
        # Generate random performance metrics with some bias to create variation
        win_rate = np.random.uniform(0.4, 0.7)
        profit_factor = np.random.uniform(1.0, 3.0)
        sharpe_ratio = np.random.uniform(0.5, 2.5)
        
        # Add some bias based on strategy type
        if strategy == 'trend_following':
            win_rate *= 1.2  # Better win rate for trend following
            profit_factor *= 1.2
        elif strategy == 'counter_trend':
            sharpe_ratio *= 1.2  # Better risk-adjusted for counter trend
        elif strategy == 'breakout':
            profit_factor *= 1.3  # Higher profit factor for breakout
        elif strategy == 'range_trading':
            win_rate *= 1.1  # Better win rate for range trading
            
        # Ensure values are within realistic bounds
        win_rate = min(win_rate, 0.85)
        profit_factor = min(profit_factor, 4.0)
        sharpe_ratio = min(sharpe_ratio, 3.0)
        
        performance[strategy] = {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'drawdown': np.random.uniform(0.05, 0.25),
            'total_trades': int(np.random.uniform(50, 500)),
            'avg_trade_pips': np.random.uniform(5, 30) if 'JPY' not in strategy else np.random.uniform(0.5, 3.0),
            'avg_holding_time': np.random.uniform(2, 48)  # Hours
        }
    
    return performance

def test_strategy_factory_integration():
    """Test that the strategy is properly registered in the factory"""
    factory = StrategyFactory()
    
    # Check if AlgorithmicMetaStrategy is in available strategies
    available_strategies = factory.get_available_strategies()
    assert 'algorithmic_meta' in available_strategies, "AlgorithmicMetaStrategy not registered in factory"
    
    # Create an instance through the factory
    meta_strategy = factory.create_strategy('algorithmic_meta')
    assert isinstance(meta_strategy, AlgorithmicMetaStrategy), "Factory did not create AlgorithmicMetaStrategy instance"
    
    print("✅ Strategy Factory integration verified!")
    return meta_strategy

def test_regime_detection(strategy: AlgorithmicMetaStrategy, data: Dict[str, pd.DataFrame]):
    """Test the regime detection logic"""
    for symbol, ohlcv in data.items():
        # Detect regime
        regime = strategy._detect_market_regime(ohlcv, symbol)
        
        # Verify it's a valid regime
        valid_regimes = list(MarketRegime)
        assert regime in valid_regimes, f"Invalid regime detected: {regime}"
        
        print(f"Detected regime for {symbol}: {regime.name}")
    
    print("✅ Regime detection verified!")

def test_strategy_weighting(strategy: AlgorithmicMetaStrategy, data: Dict[str, pd.DataFrame]):
    """Test the strategy weighting logic"""
    # Detect regimes first
    current_regimes = {}
    for symbol, ohlcv in data.items():
        regime = strategy._detect_market_regime(ohlcv, symbol)
        current_regimes[symbol] = regime
    
    # Calculate weights
    weights = strategy._calculate_strategy_weights(data, current_regimes)
    
    # Verify weights
    assert isinstance(weights, dict), "Weights should be a dictionary"
    assert len(weights) > 0, "No weights calculated"
    
    # Verify weights sum to 1.0
    total_weight = sum(weights.values())
    assert abs(total_weight - 1.0) < 0.0001, f"Weights don't sum to 1.0: {total_weight}"
    
    # Print the weights
    print("Strategy weights:")
    for strategy_type, weight in weights.items():
        print(f"  {strategy_type}: {weight:.4f}")
    
    print("✅ Strategy weighting verified!")
    return weights

def test_sub_strategies(strategy: AlgorithmicMetaStrategy):
    """Test that sub-strategies are initialized correctly"""
    # Check if sub-strategies were initialized
    assert strategy.sub_strategies is not None
    assert len(strategy.sub_strategies) > 0
    
    # Print the initialized sub-strategies
    print(f"Number of sub-strategies: {len(strategy.sub_strategies)}")
    print("Initialized sub-strategies:")
    for strategy_type, sub_strategy in strategy.sub_strategies.items():
        print(f"  - {strategy_type}")
    
    print("✅ Sub-strategies initialization verified!")

def test_signal_generation(strategy: AlgorithmicMetaStrategy, data: Dict[str, pd.DataFrame]):
    """Test signal generation logic"""
    # Generate signals for current time
    current_time = pd.Timestamp.now(tz=pytz.UTC)
    signals = strategy.generate_signals(data, current_time)
    
    # Don't assert signals exist, as they may not for every test run
    # depending on conditions and thresholds
    if signals:
        print(f"Generated {len(signals)} signals:")
        for symbol, signal in signals.items():
            print(f"  {symbol}: {signal['direction']} signal with strength {signal['strength']:.4f}")
            print(f"    Sub-strategies: {signal['sub_strategies']}")
            print(f"    Weights: {signal['weights']}")
            print(f"    Resolved via: {signal['conflict_resolution']}")
    else:
        print("No signals generated in this test run")
    
    print("✅ Signal generation executed successfully")
    return signals

def print_weights(weights: Dict[str, float]):
    """Print strategy weights in a formatted way"""
    print("\nStrategy weights:")
    print("-" * 30)
    for strategy_type, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"{strategy_type.ljust(20)}: {weight:.4f}")
    print("-" * 30)
    
    # Verify weights sum to approximately 1.0
    total = sum(weights.values())
    print(f"Total weight: {total:.4f} (should be ~1.0)")
    
    print("✅ Strategy weights calculated successfully")

def main():
    print("="*80)
    print("ALGORITHMIC META-STRATEGY TEST")
    print("="*80)
    
    # Test strategy factory integration
    meta_strategy = test_strategy_factory_integration()
    
    # Test sub-strategies initialization
    print("\nTesting sub-strategies initialization...")
    test_sub_strategies(meta_strategy)
    
    # Generate synthetic data for testing
    symbols = ['EURUSD', 'USDJPY', 'GBPUSD']
    print(f"\nGenerating synthetic data for {symbols}...")
    data = generate_synthetic_data(symbols)
    print(f"✅ Generated {len(data)} data series")
    
    # Test regime detection
    print("\nTesting regime detection...")
    test_regime_detection(meta_strategy, data)
    
    # Test strategy weighting
    print("\nTesting strategy weighting...")
    weights = test_strategy_weighting(meta_strategy, data)
    print_weights(weights)
    
    # Test signal generation
    print("\nTesting signal generation...")
    signals = test_signal_generation(meta_strategy, data)
    
    print("\n"+"="*80)
    print("TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
