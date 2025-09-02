#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the Pips-a-Day Forex Strategy
This validates the strategy's target-based trading, session awareness,
and regime compatibility.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Dict, Any, List, Tuple

# Add the project path to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our trading system components
from trading_bot.strategies.forex.pips_a_day_strategy import PipsADayStrategy
from trading_bot.enums.market_enums import MarketRegime, MarketSession, EntryQuality, TradeDirection
from trading_bot.events.event_bus import EventBus

# Initialize EventBus (singleton)
event_bus = EventBus.get_instance()

def generate_synthetic_data(num_days: int = 10, 
                          timeframe: str = '30m', 
                          base_currency_pairs: List[str] = None,
                          volatility_factor: float = 1.0) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Generate synthetic OHLCV data for testing
    
    Args:
        num_days: Number of days to generate
        timeframe: Timeframe for the data
        base_currency_pairs: List of currency pairs to generate data for
        volatility_factor: Volatility multiplier (higher = more volatile)
        
    Returns:
        Dictionary of symbol -> {timeframe -> DataFrame}
    """
    if base_currency_pairs is None:
        base_currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        
    # Set up timeframe in minutes
    if timeframe == '1m':
        minutes = 1
    elif timeframe == '5m':
        minutes = 5
    elif timeframe == '15m':
        minutes = 15
    elif timeframe == '30m':
        minutes = 30
    elif timeframe == '1h':
        minutes = 60
    elif timeframe == '4h':
        minutes = 240
    else:
        minutes = 60
    
    # Calculate number of periods per day
    periods_per_day = 24 * 60 // minutes
    total_periods = periods_per_day * num_days

    # Get current time and round to nearest timeframe
    end_time = datetime.now().replace(microsecond=0)
    end_time = end_time - timedelta(minutes=end_time.minute % minutes, 
                                   seconds=end_time.second)
    
    # Generate dates working backwards from end_time
    dates = [end_time - timedelta(minutes=i*minutes) for i in range(total_periods)]
    dates.reverse()  # Put in chronological order
    
    # Create empty result dictionary
    result = {}
    
    # Generate data for each currency pair
    for pair in base_currency_pairs:
        # Set base price depending on pair
        if 'JPY' in pair:
            base_price = 110.0  # JPY pairs have higher nominal value
            pip_size = 0.01
        else:
            base_price = 1.20  # Non-JPY pairs typically around 1.0-1.5
            pip_size = 0.0001
        
        # Create price series with random walk and some cyclicality
        np.random.seed(hash(pair) % 100)  # Different seed for each pair
        
        # Generate random walk
        random_walk = np.random.normal(0, volatility_factor * pip_size * 30, total_periods).cumsum()
        
        # Add cyclicality - daily cycle
        daily_cycle = np.sin(np.linspace(0, 2*np.pi*num_days, total_periods)) * pip_size * 80 * volatility_factor
        
        # Add cyclicality - session-based volatility
        session_volatility = []
        for i, date in enumerate(dates):
            hour = date.hour
            if 0 <= hour < 8:  # Asian session (less volatile)
                volatility = 0.7
            elif 8 <= hour < 16:  # European session (more volatile)
                volatility = 1.2
            else:  # US session (medium volatile)
                volatility = 1.0
                
            session_volatility.append(volatility)
        
        session_effect = np.array(session_volatility) * pip_size * 20 * volatility_factor
        
        # Combine components
        price_series = base_price + random_walk + daily_cycle + session_effect
        
        # Generate OHLCV data
        ohlcv_data = []
        for i in range(total_periods):
            close = price_series[i]
            high = close + abs(np.random.normal(0, pip_size * 15 * volatility_factor))
            low = close - abs(np.random.normal(0, pip_size * 15 * volatility_factor))
            open_price = price_series[i-1] if i > 0 else close
            volume = np.random.randint(100, 1000)
            
            ohlcv_data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        # Create DataFrame
        df = pd.DataFrame(ohlcv_data, index=dates)
        
        # Add to result
        result[pair] = {timeframe: df}
        
    return result

def test_target_management():
    """
    Test the daily target management functionality
    """
    logger.info("TESTING: Target Management")
    
    # Initialize strategy
    params = {
        'base_currency_pairs': ['EURUSD', 'GBPUSD', 'USDJPY'],
        'daily_pip_target': 30,
        'weekly_target_modifier': 0.8,
        'monthly_target_modifier': 0.7,
        'min_daily_target': 20,
        'max_daily_target': 50,
        'volatility_based_targets': True,
        'target_volatility_factor': 1.2,
        'daily_loss_limit': -20
    }
    
    strategy = PipsADayStrategy(parameters=params)
    
    # Generate synthetic data
    data = generate_synthetic_data(num_days=5, timeframe='1h', 
                                base_currency_pairs=params['base_currency_pairs'])
    
    # Test calculation of daily targets
    strategy._calculate_daily_targets()
    
    # Print targets
    logger.info(f"Daily targets: {strategy.daily_targets}")
    
    # Update volatility factors
    for symbol in params['base_currency_pairs']:
        strategy._update_volatility_factors(symbol, data[symbol]['1h'])
        
    # Recalculate targets after volatility updates
    strategy._calculate_daily_targets()
    
    logger.info(f"Volatility-adjusted targets: {strategy.daily_targets}")
    
    # Test status management
    strategy.current_day_pip_total = 15
    status = strategy._check_daily_status()
    logger.info(f"Status at 15 pips: {status}")
    
    strategy.current_day_pip_total = 31
    status = strategy._check_daily_status()
    logger.info(f"Status at 31 pips: {status}")
    
    strategy.current_day_pip_total = 50
    status = strategy._check_daily_status()
    logger.info(f"Status at 50 pips: {status}")
    
    # Reset and test loss limit
    strategy.current_day_pip_total = 0
    strategy.current_day_status = MarketRegime.NORMAL
    
    strategy.current_day_pip_total = -10
    status = strategy._check_daily_status()
    logger.info(f"Status at -10 pips: {status}")
    
    strategy.current_day_pip_total = -21
    status = strategy._check_daily_status()
    logger.info(f"Status at -21 pips: {status}")
    
    # Test status between days
    strategy._reset_daily_stats()
    logger.info(f"After reset, pip total: {strategy.current_day_pip_total}, status: {strategy.current_day_status}")
    
    # Test statistics tracking
    logger.info("Target management tests completed successfully")

def test_setup_evaluation():
    """
    Test the setup evaluation logic
    """
    logger.info("TESTING: Setup Evaluation")
    
    # Initialize strategy
    params = {
        'base_currency_pairs': ['EURUSD', 'GBPUSD', 'USDJPY'],
        'primary_execution_timeframe': '1h',
        'min_data_lookback': 30,
        'entry_indicators': ['rsi', 'macd', 'bollinger'],
        'filter_indicators': ['moving_averages', 'adx'],
        'confirmation_indicators': ['macd'],
        'required_confirmation_count': 2,
        'min_entry_quality': EntryQuality.MARGINAL.value
    }
    
    strategy = PipsADayStrategy(parameters=params)
    
    # Generate synthetic data with different volatility profiles
    symbols = params['base_currency_pairs']
    trending_data = generate_synthetic_data(num_days=5, timeframe='1h', 
                                          base_currency_pairs=symbols,
                                          volatility_factor=1.5)
    
    ranging_data = generate_synthetic_data(num_days=5, timeframe='1h', 
                                         base_currency_pairs=symbols,
                                         volatility_factor=0.8)
    
    # Test setup evaluation on both datasets
    current_time = pd.Timestamp.now()
    
    # Test trending data
    logger.info("Testing trending data setups:")
    for symbol in symbols:
        quality, direction, strength = strategy._evaluate_setup(
            symbol, trending_data[symbol]['1h'], current_time)
        logger.info(f"{symbol}: Quality={quality.name}, Direction={direction.name}, Strength={strength:.2f}")
        
    # Test ranging data
    logger.info("Testing ranging data setups:")
    for symbol in symbols:
        quality, direction, strength = strategy._evaluate_setup(
            symbol, ranging_data[symbol]['1h'], current_time)
        logger.info(f"{symbol}: Quality={quality.name}, Direction={direction.name}, Strength={strength:.2f}")
        
    # Test position sizing based on quality and strength
    logger.info("Testing position sizing:")
    for quality in [EntryQuality.PREMIUM, EntryQuality.STANDARD, EntryQuality.MARGINAL]:
        for strength in [0.3, 0.6, 0.9]:
            size = strategy._calculate_position_size('EURUSD', quality, strength)
            logger.info(f"Quality={quality.name}, Strength={strength:.1f}, Size={size:.2f}")
            
    logger.info("Setup evaluation tests completed successfully")

def test_signal_generation():
    """
    Test signal generation for different setups
    """
    logger.info("TESTING: Signal Generation")
    
    # Initialize strategy
    params = {
        'base_currency_pairs': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
        'primary_execution_timeframe': '1h',
        'min_data_lookback': 30,
        'entry_indicators': ['rsi', 'macd', 'bollinger'],
        'filter_indicators': ['moving_averages', 'adx'],
        'confirmation_indicators': ['macd'],
        'required_confirmation_count': 2,
        'min_entry_quality': EntryQuality.MARGINAL.value,
        'scale_out_levels': [0.3, 0.6, 0.9],
        'scale_out_percents': [30, 30, 40],
        'scale_out_enabled': True,
        'daily_pip_target': 30,
        'base_stop_loss_pips': 20,
        'min_reward_risk_ratio': 1.5,
        'use_adaptive_stops': True,
        'atr_stop_loss_factor': 1.2,
        'correlation_filter_enabled': True,
        'max_entries_per_day': 5,
        'base_position_size': 0.1,
        'premium_entry_boost': 1.3,
        'trade_asian_session': True,
        'trade_european_session': True,
        'trade_us_session': True,
        'min_daily_range_pips': 30,
        'news_filter_enabled': False
    }
    
    strategy = PipsADayStrategy(parameters=params)
    
    # Generate synthetic data with different profiles
    symbols = params['base_currency_pairs']
    market_data = generate_synthetic_data(num_days=5, timeframe='1h', 
                                       base_currency_pairs=symbols,
                                       volatility_factor=1.2)
    
    # Test signal generation
    current_time = pd.Timestamp.now()
    
    # Register event handler for testing
    received_events = []
    
    def event_handler(event_type, data):
        received_events.append((event_type, data))
        logger.info(f"Received event: {event_type}")
        
    event_bus.subscribe('pips_a_day_signal', event_handler)
    
    # Generate signals
    signals = strategy.generate_signals(market_data, current_time)
    
    # Check output
    if signals:
        logger.info(f"Generated {len(signals)} signals:")
        for symbol, signal in signals.items():
            logger.info(f"{symbol}: {signal['direction'].name} with {signal['setup_quality']} quality")
            logger.info(f"  Entry: {signal['entry_price']:.5f}, SL: {signal['stop_loss']:.5f}, TP: {signal['take_profit']:.5f}")
            logger.info(f"  Target pips: {signal['target_pips']:.1f}, Position size: {signal['position_size']:.3f}")
            
            if 'scale_out_levels' in signal:
                logger.info(f"  Scale-out levels: {len(signal['scale_out_levels'])}")
                for level in signal['scale_out_levels']:
                    logger.info(f"    {level['percent']}% at {level['price']:.5f}")
    else:
        logger.info("No signals generated")
        
    # Test scale-out detection
    if signals:
        # Simulate price movement for scale-out
        symbol = list(signals.keys())[0]
        signal = signals[symbol]
        
        # Create modified data with price reaching first scale-out level
        modified_data = market_data.copy()
        
        if signal['direction'] == TradeDirection.LONG:
            scale_price = signal['entry_price'] + 0.3 * (signal['take_profit'] - signal['entry_price'])
            modified_data[symbol]['1h'].iloc[-1]['close'] = scale_price
        else:
            scale_price = signal['entry_price'] - 0.3 * (signal['entry_price'] - signal['take_profit'])
            modified_data[symbol]['1h'].iloc[-1]['close'] = scale_price
            
        # Check scale-out detection
        strategy.check_scale_out_levels(modified_data, current_time)
        
    # Test trade closed handling
    if signals:
        # Simulate a closed trade
        symbol = list(signals.keys())[0]
        signal = signals[symbol]
        
        trade_data = {
            'id': signal['id'],
            'strategy_name': strategy.__class__.__name__,
            'symbol': symbol,
            'direction': signal['direction'],
            'entry_price': signal['entry_price'],
            'exit_price': signal['take_profit'],
            'pips': signal['target_pips']
        }
        
        # Handle closed trade
        strategy.on_trade_closed(trade_data)
        
        # Check if daily target status was updated
        logger.info(f"After trade close: Pip total = {strategy.current_day_pip_total:.1f}, Status = {strategy.current_day_status}")
    
    # Check events
    logger.info(f"Received {len(received_events)} events during testing")
    
    # Unsubscribe
    event_bus.unsubscribe('pips_a_day_signal', event_handler)
    
    logger.info("Signal generation tests completed successfully")

def test_regime_compatibility():
    """
    Test regime compatibility scoring and optimization
    """
    logger.info("TESTING: Regime Compatibility")
    
    # Initialize strategy
    strategy = PipsADayStrategy()
    
    # Test compatibility across all regimes
    for regime in MarketRegime:
        score = strategy.calculate_regime_compatibility(regime)
        logger.info(f"Regime {regime.name}: Compatibility score = {score:.2f}")
        
    # Test parameter optimization for different regimes
    test_regimes = [
        MarketRegime.TRENDING_BULL,
        MarketRegime.RANGE_BOUND,
        MarketRegime.VOLATILE,
        MarketRegime.BREAKOUT,
        MarketRegime.NORMAL
    ]
    
    for regime in test_regimes:
        # Reset to default parameters
        strategy = PipsADayStrategy()
        
        # Get parameters before optimization
        before_rr = strategy.parameters['min_reward_risk_ratio']
        before_volatility = strategy.parameters.get('target_volatility_factor', 1.0)
        
        # Optimize for regime
        strategy.optimize_for_regime(regime)
        
        # Get parameters after optimization
        after_rr = strategy.parameters['min_reward_risk_ratio']
        after_volatility = strategy.parameters.get('target_volatility_factor', 1.0)
        
        logger.info(f"Regime {regime.name} optimization:")
        logger.info(f"  Reward/Risk: {before_rr:.1f} → {after_rr:.1f}")
        logger.info(f"  Volatility Factor: {before_volatility:.1f} → {after_volatility:.1f}")
        logger.info(f"  Scale-out levels: {strategy.parameters['scale_out_levels']}")
        logger.info(f"  Scale-out percents: {strategy.parameters['scale_out_percents']}")
        
    logger.info("Regime compatibility tests completed successfully")

def main():
    """Run all tests"""
    logger.info("STARTING PIPS-A-DAY STRATEGY TESTS")
    
    test_target_management()
    logger.info("-" * 50)
    
    test_setup_evaluation()
    logger.info("-" * 50)
    
    test_signal_generation()
    logger.info("-" * 50)
    
    test_regime_compatibility()
    logger.info("-" * 50)
    
    logger.info("ALL TESTS COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()
