#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retracement Strategy Test

This script tests the Retracement Strategy implementation and integration
with the strategy factory and selection system.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_test_data() -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic price data for testing retracement patterns.
    
    Returns:
        Dictionary of symbol -> OHLCV DataFrame with synthetic price data
    """
    logger.info("Generating synthetic test data...")
    
    # Common parameters
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
    periods = 200
    now = pd.Timestamp.now()
    dates = pd.date_range(end=now, periods=periods, freq='1H')
    
    # Generate data for each symbol
    data = {}
    
    for symbol in symbols:
        # Set initial price based on symbol
        if symbol == 'EURUSD':
            base_price = 1.10
        elif symbol == 'GBPUSD':
            base_price = 1.25
        elif symbol == 'USDJPY':
            base_price = 110.0
        else:
            base_price = 1.0
            
        # Create synthetic price series with trend and retracement
        np.random.seed(42 + hash(symbol) % 100)  # Different seed for each symbol
        
        # Create price array
        prices = []
        current_price = base_price
        
        # Generate an uptrend with retracements
        for i in range(periods):
            if i < 50:  # Initial uptrend
                change = np.random.normal(0.0001, 0.0005)
                current_price *= (1 + change)
            elif i < 70:  # First retracement (approximately 50%)
                change = np.random.normal(-0.0001, 0.0005)
                current_price *= (1 + change)
            elif i < 120:  # Second uptrend
                change = np.random.normal(0.0002, 0.0005)
                current_price *= (1 + change)
            elif i < 150:  # Second retracement (approximately 38.2%)
                change = np.random.normal(-0.00005, 0.0004)
                current_price *= (1 + change)
            else:  # Final uptrend
                change = np.random.normal(0.0001, 0.0005)
                current_price *= (1 + change)
                
            prices.append(current_price)
        
        # Create DataFrame with OHLCV data
        close_prices = np.array(prices)
        
        # Create open, high, low based on close
        open_prices = close_prices * (1 + np.random.normal(0, 0.0005, periods))
        high_prices = np.maximum(close_prices, open_prices) * (1 + np.abs(np.random.normal(0, 0.001, periods)))
        low_prices = np.minimum(close_prices, open_prices) * (1 - np.abs(np.random.normal(0, 0.001, periods)))
        volumes = np.random.uniform(100, 1000, periods)
        
        df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }, index=dates)
        
        data[symbol] = df
    
    logger.info(f"Generated data for {len(symbols)} symbols")
    return data

def test_retracement_strategy():
    """Test the retracement strategy implementation."""
    try:
        # Import strategy factory and other necessary components
        from trading_bot.strategies.strategy_factory import StrategyFactory
        from trading_bot.strategies.forex.retracement_strategy import ForexRetracementStrategy
        from trading_bot.strategies.strategy_template import MarketRegime
        
        logger.info("Testing retracement strategy implementation...")
        
        # Verify strategy is registered in factory
        factory = StrategyFactory()
        if 'forex_retracement' not in factory.get_available_strategies():
            logger.error("Retracement strategy is not registered in the factory")
            return False
        
        # Create an instance of the strategy
        retracement_strategy = factory.create_strategy('forex_retracement')
        
        if retracement_strategy is None:
            logger.error("Failed to create retracement strategy instance")
            return False
            
        logger.info("Successfully created retracement strategy instance")
        
        # Generate test data
        test_data = generate_test_data()
        
        # Check regime compatibility
        for regime in MarketRegime:
            score = retracement_strategy.get_regime_compatibility_score(regime)
            logger.info(f"Regime compatibility {regime.name}: {score:.2f}")
            
            # Get optimal timeframe for this regime
            if hasattr(retracement_strategy, 'get_optimal_timeframe'):
                optimal_tf = retracement_strategy.get_optimal_timeframe(regime)
                logger.info(f"Optimal timeframe for {regime.name}: {optimal_tf}")
        
        # Test signal generation
        current_time = pd.Timestamp.now()
        signals = retracement_strategy.generate_signals(test_data, current_time)
        
        if signals:
            logger.info(f"Generated {len(signals)} signals:")
            for symbol, signal in signals.items():
                logger.info(f"Signal for {symbol}:")
                logger.info(f"  Direction: {signal['direction'].name}")
                logger.info(f"  Strength: {signal['strength']:.2f}")
                logger.info(f"  Entry: {signal['entry_price']:.5f}")
                logger.info(f"  Stop Loss: {signal['stop_loss']:.5f}")
                logger.info(f"  Take Profit: {signal['take_profit']:.5f}")
                logger.info(f"  Retracement Level: {signal['retracement_level']}")
        else:
            logger.info("No signals generated")
        
        # Test optimization
        optimized = retracement_strategy.optimize(test_data, MarketRegime.TRENDING)
        
        if optimized:
            logger.info("Optimized parameters for TRENDING regime:")
            for param, value in optimized.items():
                logger.info(f"  {param}: {value}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error testing retracement strategy: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_rotation():
    """Test strategy rotation with retracement strategy."""
    try:
        # Import necessary components
        from trading_bot.strategies.forex.strategy_selector import ForexStrategySelector
        from trading_bot.strategies.strategy_template import MarketRegime
        
        logger.info("Testing strategy rotation with retracement strategy...")
        
        # Generate test data
        test_data = generate_test_data()
        
        # Create selector
        selector = ForexStrategySelector()
        
        # Test selection for different regimes
        for regime in [MarketRegime.TRENDING, MarketRegime.RANGING, MarketRegime.VOLATILE]:
            logger.info(f"Testing selection for {regime.name} regime...")
            
            # Force regime for testing
            selector.override_regime = regime
            
            # Select strategy
            selection = selector.select_strategy(test_data)
            
            if selection and 'strategy_type' in selection:
                logger.info(f"Selected strategy for {regime.name}: {selection['strategy_type']}")
                
                # Check if retracement strategy is selected for TRENDING regime
                if regime == MarketRegime.TRENDING and selection['strategy_type'] == 'forex_retracement':
                    logger.info("✓ Retracement strategy correctly selected for TRENDING regime")
                    
                logger.info(f"  Selection reason: {selection.get('reason', 'N/A')}")
                logger.info(f"  Score: {selection.get('score', 0):.2f}")
            else:
                logger.warning(f"No strategy selected for {regime.name}")
                
        return True
        
    except Exception as e:
        logger.error(f"Error testing strategy rotation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    logger.info("Starting Retracement Strategy tests")
    
    # Run tests
    strategy_ok = test_retracement_strategy()
    rotation_ok = test_strategy_rotation()
    
    # Print summary
    logger.info("\nTest Summary:")
    logger.info(f"Strategy Implementation: {'✓ PASS' if strategy_ok else '✗ FAIL'}")
    logger.info(f"Strategy Rotation: {'✓ PASS' if rotation_ok else '✗ FAIL'}")
    
    if strategy_ok and rotation_ok:
        logger.info("\n✓ All Retracement Strategy tests PASSED")
        return 0
    else:
        logger.error("\n✗ Some Retracement Strategy tests FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
