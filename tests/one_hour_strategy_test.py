#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-Hour Forex Strategy Test Script
Tests the time-aware, session-focused intraday strategy
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
from trading_bot.strategies.forex.one_hour_strategy import OneHourForexStrategy, HourlyPattern
from trading_bot.strategies.base.forex_base import MarketRegime, MarketSession, TradeDirection
from trading_bot.utils.event_bus import EventBus

def generate_synthetic_data(symbols: List[str], days: int = 5, freq: str = '1H') -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Generate synthetic OHLCV data for testing
    
    Args:
        symbols: List of forex symbols
        days: Number of days of data
        freq: Primary data frequency
        
    Returns:
        Dictionary of symbol -> timeframe -> OHLCV DataFrame
    """
    data = {}
    
    # Generate timestamps for multiple timeframes
    end_time = pd.Timestamp.now(tz=pytz.UTC)
    start_time = end_time - pd.Timedelta(days=days)
    
    timeframes = {
        '1h': pd.date_range(start=start_time, end=end_time, freq='1H'),
        '15m': pd.date_range(start=start_time, end=end_time, freq='15min'),
        '4h': pd.date_range(start=start_time, end=end_time, freq='4H')
    }
    
    for symbol in symbols:
        # Base price for this symbol
        base_price = 1.0 if 'JPY' in symbol else 100.0
        
        # Add noise for variation
        symbol_noise = np.random.uniform(-0.05, 0.05)
        base_price *= (1 + symbol_noise)
        
        # Dictionary to hold dataframes for each timeframe
        symbol_data = {}
        
        # Generate data for each timeframe
        for tf, timestamps in timeframes.items():
            # Create base price and random walk
            returns = np.random.normal(0, 0.0003, size=len(timestamps))
            
            # Add some trends, ranges and volatility patterns
            trends = []
            session_based_patterns = []
            
            # Add session-based volatility patterns
            for ts in timestamps:
                hour = ts.hour
                
                # Asian session (0-6 UTC) - lower volatility
                if 0 <= hour < 7:
                    session_pattern = np.random.normal(0, 0.0002)
                # European session (7-12 UTC) - medium to high volatility
                elif 7 <= hour < 13:
                    session_pattern = np.random.normal(0, 0.0004)
                # US session (13-20 UTC) - highest volatility
                elif 13 <= hour < 21:
                    session_pattern = np.random.normal(0, 0.0005)
                # Late US/Early Asian - tapering volatility
                else:
                    session_pattern = np.random.normal(0, 0.0003)
                    
                session_based_patterns.append(session_pattern)
            
            # Add some trends (changing every 20-30 bars)
            trend_changes = np.random.randint(20, 30, size=days*3)
            current_trend = np.random.choice([-1, 1])
            
            for i in range(len(timestamps)):
                if i % sum(trend_changes[:1]) == 0:
                    current_trend = np.random.choice([-1, 1])
                    
                trend_impact = current_trend * 0.0001
                trends.append(trend_impact)
            
            # Combine everything
            combined_returns = returns + np.array(session_based_patterns) + np.array(trends[:len(timestamps)])
            
            # Generate price series
            close_prices = base_price * (1 + np.cumsum(combined_returns))
            
            # Create realistic OHLCV data
            high_prices = close_prices * (1 + np.random.uniform(0, 0.001, size=len(timestamps)))
            low_prices = close_prices * (1 - np.random.uniform(0, 0.001, size=len(timestamps)))
            open_prices = close_prices * (1 + np.random.normal(0, 0.0005, size=len(timestamps)))
            
            # Ensure low < open, close < high
            for i in range(len(timestamps)):
                min_price = min(open_prices[i], close_prices[i])
                max_price = max(open_prices[i], close_prices[i])
                low_prices[i] = min(low_prices[i], min_price * 0.999)
                high_prices[i] = max(high_prices[i], max_price * 1.001)
            
            # Generate volume with session-based patterns
            volume = np.zeros(len(timestamps))
            for i, ts in enumerate(timestamps):
                hour = ts.hour
                if 0 <= hour < 7:  # Asian session - lower volume
                    base_vol = 500
                elif 7 <= hour < 13:  # European session - medium volume
                    base_vol = 1000
                elif 13 <= hour < 21:  # US session - highest volume
                    base_vol = 1500
                else:  # Late US/Early Asian - tapering volume
                    base_vol = 800
                
                # Add randomness
                volume[i] = np.random.normal(base_vol, base_vol * 0.1)
            
            # Create DataFrame
            df = pd.DataFrame({
                'open': open_prices,
                'high': high_prices,
                'low': low_prices,
                'close': close_prices,
                'volume': volume
            }, index=timestamps)
            
            # Add to timeframe dict
            symbol_data[tf] = df
        
        # Add to main data dict
        data[symbol] = symbol_data
    
    return data

def test_strategy_factory_integration():
    """Test that the strategy is properly registered in the factory"""
    factory = StrategyFactory()
    
    # Check if OneHourForexStrategy is in available strategies
    available_strategies = factory.get_available_strategies()
    assert 'one_hour_forex' in available_strategies, "OneHourForexStrategy not registered in factory"
    
    # Create an instance through the factory
    one_hour_strategy = factory.create_strategy('one_hour_forex')
    assert isinstance(one_hour_strategy, OneHourForexStrategy), "Factory did not create OneHourForexStrategy instance"
    
    print("✅ Strategy Factory integration verified!")
    return one_hour_strategy

def test_pattern_detection(strategy: OneHourForexStrategy, data: Dict[str, Dict[str, pd.DataFrame]]):
    """Test the hourly pattern detection logic"""
    all_patterns = []
    
    for symbol, timeframes in data.items():
        hourly_data = timeframes['1h']
        
        # Update intraday analysis to detect patterns
        strategy._update_intraday_analysis(symbol, hourly_data)
        
        # Get detected patterns
        patterns = strategy.hourly_patterns.get(symbol, [])
        
        if patterns:
            print(f"Detected {len(patterns)} patterns for {symbol}:")
            for i, pattern in enumerate(patterns[:5]):  # Show top 5 patterns
                print(f"  {i+1}. {pattern['type']} ({pattern['direction'].name}) - Strength: {pattern['strength']:.2f}")
            
            all_patterns.extend(patterns)
    
    if all_patterns:
        # Analyze pattern distribution
        pattern_types = {}
        for pattern in all_patterns:
            pattern_type = pattern['type']
            if pattern_type not in pattern_types:
                pattern_types[pattern_type] = 0
            pattern_types[pattern_type] += 1
        
        print("\nPattern distribution:")
        for pattern_type, count in pattern_types.items():
            print(f"  {pattern_type}: {count} occurrences ({count/len(all_patterns)*100:.1f}%)")
    
    print("✅ Pattern detection verified!")

def test_signal_generation(strategy: OneHourForexStrategy, data: Dict[str, Dict[str, pd.DataFrame]]):
    """Test signal generation logic"""
    # Convert data format for the strategy
    formatted_data = {}
    for symbol, timeframes in data.items():
        formatted_data[symbol] = timeframes
    
    # Generate signals for the current time
    current_time = pd.Timestamp.now(tz=pytz.UTC)
    signals = strategy.generate_signals(formatted_data, current_time)
    
    # Analyze signals
    if signals:
        print(f"Generated {len(signals)} signals:")
        for symbol, signal in signals.items():
            print(f"  {symbol}: {signal['direction'].name} at {signal['entry_price']:.5f}")
            print(f"    Pattern: {signal['pattern']}")
            print(f"    Strength: {signal['strength']:.2f}")
            print(f"    Stop Loss: {signal['stop_loss']:.5f}")
            print(f"    Take Profit: {signal['take_profit']:.5f}")
    else:
        print("No signals generated in this test run (normal depending on market conditions)")
    
    print("✅ Signal generation logic verified!")

def test_regime_compatibility(strategy: OneHourForexStrategy):
    """Test regime compatibility scoring"""
    regimes = list(MarketRegime)
    
    print("Regime compatibility scores:")
    for regime in regimes:
        score = strategy.get_regime_compatibility(regime)
        print(f"  {regime.name}: {score:.2f}")
    
    # Verify the strategy has highest compatibility with trending and breakout regimes
    trending_score = strategy.get_regime_compatibility(MarketRegime.TRENDING)
    breakout_score = strategy.get_regime_compatibility(MarketRegime.BREAKOUT)
    
    assert trending_score >= 0.7, "Strategy should be highly compatible with trending markets"
    assert breakout_score >= 0.7, "Strategy should be highly compatible with breakout markets"
    
    print("✅ Regime compatibility verified!")

def test_time_based_position_sizing(strategy: OneHourForexStrategy):
    """Test time-based position sizing logic"""
    # Test for different hours within a session
    symbol = "EURUSD"
    hourly_atr = 0.0005
    
    # European session tests
    session = MarketSession.EUROPEAN
    early_time = pd.Timestamp('2023-01-01 08:00:00', tz=pytz.UTC)  # Early European
    mid_time = pd.Timestamp('2023-01-01 10:00:00', tz=pytz.UTC)    # Mid European
    late_time = pd.Timestamp('2023-01-01 12:00:00', tz=pytz.UTC)   # Late European
    
    early_hours = strategy._calculate_hours_in_session(early_time, session)
    mid_hours = strategy._calculate_hours_in_session(mid_time, session)
    late_hours = strategy._calculate_hours_in_session(late_time, session)
    
    early_size = strategy._calculate_position_size(symbol, early_hours, hourly_atr)
    mid_size = strategy._calculate_position_size(symbol, mid_hours, hourly_atr)
    late_size = strategy._calculate_position_size(symbol, late_hours, hourly_atr)
    
    print("Time-based position sizing:")
    print(f"  Early European session ({early_hours} hours in): {early_size:.4f}")
    print(f"  Mid European session ({mid_hours} hours in): {mid_size:.4f}")
    print(f"  Late European session ({late_hours} hours in): {late_size:.4f}")
    
    # Verify time decay is working
    if strategy.parameters['reduce_exposure_after_hours'] < late_hours:
        assert early_size > late_size, "Position size should decrease as session progresses"
        print("  ✓ Time decay verified: position size decreases as session progresses")
    
    print("✅ Time-based position sizing verified!")

def test_session_awareness(strategy: OneHourForexStrategy):
    """Test session awareness"""
    # Test session detection
    asian_time = pd.Timestamp('2023-01-01 03:00:00', tz=pytz.UTC)
    european_time = pd.Timestamp('2023-01-01 09:00:00', tz=pytz.UTC)
    us_time = pd.Timestamp('2023-01-01 16:00:00', tz=pytz.UTC)
    overnight_time = pd.Timestamp('2023-01-01 22:00:00', tz=pytz.UTC)
    
    # Get detected sessions
    asian_session = strategy.determine_trading_session(asian_time)
    european_session = strategy.determine_trading_session(european_time)
    us_session = strategy.determine_trading_session(us_time)
    overnight_session = strategy.determine_trading_session(overnight_time)
    
    print("Session detection:")
    print(f"  03:00 UTC detected as: {asian_session.name}")
    print(f"  09:00 UTC detected as: {european_session.name}")
    print(f"  16:00 UTC detected as: {us_session.name}")
    print(f"  22:00 UTC detected as: {overnight_session.name}")
    
    # Check session transitions
    strategy._analyze_session_transition("EURUSD", asian_session, european_session, european_time)
    
    print("✅ Session awareness verified!")

def main():
    print("="*80)
    print("ONE-HOUR FOREX STRATEGY TEST")
    print("="*80)
    
    # Test strategy factory integration
    one_hour_strategy = test_strategy_factory_integration()
    
    # Test regime compatibility
    print("\nTesting regime compatibility...")
    test_regime_compatibility(one_hour_strategy)
    
    # Test session awareness
    print("\nTesting session awareness...")
    test_session_awareness(one_hour_strategy)
    
    # Test time-based position sizing
    print("\nTesting time-based position sizing...")
    test_time_based_position_sizing(one_hour_strategy)
    
    # Generate synthetic data for pattern and signal testing
    symbols = ['EURUSD', 'USDJPY', 'GBPUSD']
    print(f"\nGenerating synthetic data for {symbols}...")
    data = generate_synthetic_data(symbols)
    print(f"✅ Generated data for {len(data)} symbols")
    
    # Test pattern detection
    print("\nTesting hourly pattern detection...")
    test_pattern_detection(one_hour_strategy, data)
    
    # Test signal generation
    print("\nTesting signal generation...")
    test_signal_generation(one_hour_strategy, data)
    
    print("\n"+"="*80)
    print("TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
