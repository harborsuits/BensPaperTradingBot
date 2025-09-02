#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Migration Test Runner

This script runs tests to verify compatibility between old and new system organization.
It helps ensure functionality parity during the migration process.
"""

import logging
import pandas as pd
import numpy as np
import argparse
import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MigrationTest")

# Import our migration testing framework
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trading_bot.migration.testing import MigrationTester

def generate_test_data(symbols: List[str], days: int = 60, freq: str = '1h') -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic test data for strategies.
    
    Args:
        symbols: List of symbols to generate data for
        days: Number of days of data
        freq: Data frequency
        
    Returns:
        Dictionary mapping symbols to DataFrames
    """
    logger.info(f"Generating test data for {len(symbols)} symbols, {days} days at {freq} frequency")
    
    result = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    for symbol in symbols:
        # Generate date range
        periods = int((end_date - start_date).total_seconds() / (3600 if freq == '1h' else 86400))
        dates = pd.date_range(start=start_date, periods=periods, freq=freq)
        
        # Generate price data with some randomness
        base_price = 100.0
        if 'USD' in symbol:
            base_price = 1.2  # Forex-like price
        
        # Create price series with trend and noise
        np.random.seed(hash(symbol) % 10000)  # Different seed for each symbol
        
        trend = np.cumsum(np.random.normal(0, 0.01, len(dates)))
        noise = np.random.normal(0, 0.02, len(dates))
        price_series = base_price * (1 + trend + noise)
        
        # Create OHLCV data
        df = pd.DataFrame(index=dates)
        df['open'] = price_series
        df['high'] = price_series * (1 + np.random.uniform(0, 0.01, len(dates)))
        df['low'] = price_series * (1 - np.random.uniform(0, 0.01, len(dates)))
        df['close'] = price_series * (1 + np.random.normal(0, 0.005, len(dates)))
        df['volume'] = np.random.uniform(1000, 10000, len(dates))
        
        # Ensure OHLC integrity
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        df['low'] = df[['low', 'open', 'close']].min(axis=1)
        
        # Add some technical indicators
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['rsi'] = 50 + 20 * np.random.normal(0, 1, len(dates))  # Fake RSI for testing
        
        # Add to result
        result[symbol] = df
    
    return result

def test_strategies(tester: MigrationTester, forex_only: bool = False) -> None:
    """
    Test strategy compatibility.
    
    Args:
        tester: Migration tester instance
        forex_only: Whether to test only forex strategies
    """
    # Generate test data
    forex_symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD']
    stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    
    if forex_only:
        logger.info("Testing forex strategies only")
        forex_data = generate_test_data(forex_symbols, days=60, freq='1h')
        
        # Test ForexTrendFollowingStrategy (our known successful implementation)
        tester.test_strategy(
            strategy_name="ForexTrendFollowingStrategy",
            test_data=forex_data,
            parameters={
                'fast_ma_period': 8,
                'slow_ma_period': 21,
                'adx_period': 14,
                'adx_threshold': 25
            }
        )
        
        # Test other forex strategies
        for strategy_name in [
            "ForexBreakoutStrategy",
            "ForexRangeTradingStrategy",
            "ForexMomentumStrategy"
        ]:
            try:
                tester.test_strategy(
                    strategy_name=strategy_name,
                    test_data=forex_data
                )
            except Exception as e:
                logger.error(f"Error testing {strategy_name}: {str(e)}")
    else:
        # Test all strategy types
        forex_data = generate_test_data(forex_symbols, days=60, freq='1h')
        stock_data = generate_test_data(stock_symbols, days=60, freq='1d')
        
        # Forex strategies
        for strategy_name in [
            "ForexTrendFollowingStrategy",
            "ForexBreakoutStrategy",
            "ForexRangeTradingStrategy"
        ]:
            try:
                tester.test_strategy(
                    strategy_name=strategy_name,
                    test_data=forex_data
                )
            except Exception as e:
                logger.error(f"Error testing {strategy_name}: {str(e)}")
        
        # Stock strategies
        for strategy_name in [
            "StockMomentumStrategy",
            "StockMeanReversionStrategy"
        ]:
            try:
                tester.test_strategy(
                    strategy_name=strategy_name,
                    test_data=stock_data
                )
            except Exception as e:
                logger.error(f"Error testing {strategy_name}: {str(e)}")

def test_data_processors(tester: MigrationTester) -> None:
    """
    Test data processor compatibility.
    
    Args:
        tester: Migration tester instance
    """
    # Generate test data
    forex_data = generate_test_data(['EUR/USD'], days=30, freq='1h')['EUR/USD']
    
    # Add some data quality issues for testing
    # Duplicate rows
    forex_data = pd.concat([forex_data, forex_data.iloc[-5:]])
    
    # Missing values
    forex_data.loc[forex_data.index[10:15], 'volume'] = np.nan
    
    # Price outliers
    forex_data.loc[forex_data.index[25], 'high'] = forex_data['high'].iloc[25] * 1.5
    
    # Test processors
    tester.test_data_processor(
        test_data=forex_data,
        symbol="EUR/USD",
        source="test",
        config={
            'detect_price_outliers': True,
            'fill_missing_values': True,
            'remove_duplicates': True
        }
    )

def test_config(tester: MigrationTester) -> None:
    """
    Test configuration compatibility.
    
    Args:
        tester: Migration tester instance
    """
    # Test various config keys
    tester.test_config(
        keys=[
            'data.cleaning.outlier_threshold',
            'strategies.default_parameters',
            'logging.level',
            'broker.default'
        ],
        default_values={
            'data.cleaning.outlier_threshold': 3.0,
            'logging.level': 'INFO',
            'broker.default': 'paper'
        }
    )

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Trading Bot Migration Test")
    parser.add_argument('--test-type', choices=['strategies', 'processors', 'config', 'all'],
                      default='all', help='Type of test to run')
    parser.add_argument('--forex-only', action='store_true',
                      help='Test only forex strategies')
    parser.add_argument('--output', help='Output file for results (JSON)')
    
    args = parser.parse_args()
    
    logger.info(f"Starting migration testing (type: {args.test_type})")
    
    tester = MigrationTester()
    
    if args.test_type == 'strategies' or args.test_type == 'all':
        test_strategies(tester, args.forex_only)
    
    if args.test_type == 'processors' or args.test_type == 'all':
        test_data_processors(tester)
    
    if args.test_type == 'config' or args.test_type == 'all':
        test_config(tester)
    
    # Generate and print report
    report = tester.generate_report()
    print("\n" + report)
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(tester.get_results(), f, indent=2)
        logger.info(f"Results saved to {args.output}")
    
    # Print summary stats
    summary = tester.get_summary()
    success_rate = summary['success_rate'] * 100
    identical_rate = summary['identical_results'] * 100
    
    print(f"\nMigration Testing Summary:")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Identical Results: {identical_rate:.1f}%")
    
    # Performance comparison
    perf_ratio = summary['performance'].get('ratio', 0)
    if perf_ratio > 0:
        if perf_ratio > 1.1:
            print(f"  Performance: New implementation is {perf_ratio:.2f}x slower than old")
        elif perf_ratio < 0.9:
            print(f"  Performance: New implementation is {1/perf_ratio:.2f}x faster than old")
        else:
            print(f"  Performance: Comparable between old and new implementations")
    
    # Exit with success code if all tests passed
    if success_rate == 100 and identical_rate == 100:
        logger.info("All tests passed successfully")
        sys.exit(0)
    else:
        logger.warning(f"Some tests failed or had non-identical results")
        sys.exit(1)

if __name__ == "__main__":
    main()
