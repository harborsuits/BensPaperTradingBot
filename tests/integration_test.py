#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Bot Integration Test

This script tests the integration of the reorganized trading bot components:
1. Strategy organization with proper registration and discovery
2. Data pipeline with separated cleaning and quality assurance
3. Event system connectivity and dashboard integration

Use this to verify the migration has been successful.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import argparse
import time
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IntegrationTest")

def test_strategy_organization(verbose: bool = False) -> bool:
    """
    Test the strategy organization structure.
    
    Args:
        verbose: Whether to print verbose output
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Testing strategy organization...")
    
    success = True
    
    try:
        # Import the registry
        from trading_bot.strategies.factory.strategy_registry import StrategyRegistry
        
        # Get all registered strategies
        all_strategies = StrategyRegistry.get_all_strategy_names()
        logger.info(f"Found {len(all_strategies)} registered strategies")
        
        if verbose:
            logger.info(f"Registered strategies: {', '.join(all_strategies)}")
        
        # Check for ForexTrendFollowingStrategy specifically
        if "ForexTrendFollowingStrategy" not in all_strategies:
            logger.error("ForexTrendFollowingStrategy not found in registry")
            success = False
        else:
            logger.info("ForexTrendFollowingStrategy successfully registered")
            
            # Get strategy metadata
            metadata = StrategyRegistry.get_strategy_metadata("ForexTrendFollowingStrategy")
            if verbose and metadata:
                logger.info(f"ForexTrendFollowingStrategy metadata: {metadata}")
        
        # Test strategy classification
        forex_strategies = StrategyRegistry.get_strategies_by_asset_class("forex")
        logger.info(f"Found {len(forex_strategies)} forex strategies")
        
        trend_strategies = StrategyRegistry.get_strategies_by_type("trend_following")
        logger.info(f"Found {len(trend_strategies)} trend following strategies")
        
        # Test strategy instantiation for ForexTrendFollowingStrategy
        try:
            strategy_class = StrategyRegistry.get_strategy_class("ForexTrendFollowingStrategy")
            if strategy_class:
                strategy = strategy_class()
                logger.info(f"Successfully instantiated {strategy.name}")
                
                # Test basic functionality
                if hasattr(strategy, 'calculate_indicators') and callable(getattr(strategy, 'calculate_indicators')):
                    logger.info("Strategy has calculate_indicators method")
                else:
                    logger.error("Strategy missing calculate_indicators method")
                    success = False
                
                if hasattr(strategy, 'generate_signals') and callable(getattr(strategy, 'generate_signals')):
                    logger.info("Strategy has generate_signals method")
                else:
                    logger.error("Strategy missing generate_signals method")
                    success = False
            else:
                logger.error("Failed to get strategy class")
                success = False
        except Exception as e:
            logger.error(f"Error instantiating strategy: {str(e)}")
            success = False
    
    except ImportError as e:
        logger.error(f"Import error during strategy testing: {str(e)}")
        success = False
    except Exception as e:
        logger.error(f"Unexpected error during strategy testing: {str(e)}")
        success = False
    
    return success

def generate_test_data(symbols: List[str], days: int = 30) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic test data for testing.
    
    Args:
        symbols: List of symbols to generate data for
        days: Number of days of data
        
    Returns:
        Dictionary mapping symbols to DataFrames
    """
    logger.info(f"Generating test data for {len(symbols)} symbols, {days} days")
    
    result = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    for symbol in symbols:
        # Generate date range
        dates = pd.date_range(start=start_date, periods=days)
        
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
        
        # Add some data quality issues to test the pipeline
        
        # Missing values
        if len(df) > 5:
            df.loc[df.index[2], 'volume'] = np.nan
        
        # Price spike (outlier)
        if len(df) > 10:
            df.loc[df.index[8], 'high'] = df.loc[df.index[8], 'high'] * 1.5
        
        # Duplicate row
        if len(df) > 15:
            df = pd.concat([df.iloc[:15], df.iloc[14:15], df.iloc[15:]])
        
        # Add to result
        result[symbol] = df
    
    return result

def test_data_pipeline(verbose: bool = False) -> bool:
    """
    Test the data pipeline with separated cleaning and quality assurance.
    
    Args:
        verbose: Whether to print verbose output
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Testing data pipeline...")
    
    success = True
    
    try:
        # Import the data pipeline
        from trading_bot.data.data_pipeline import create_data_pipeline
        from trading_bot.core.event_system import EventBus
        
        # Create event bus and pipeline
        event_bus = EventBus()
        pipeline = create_data_pipeline(
            config={
                'cleaning': {
                    'standardize_timestamps': True,
                    'normalize_volume': True
                },
                'quality': {
                    'auto_repair': True,
                    'quality_threshold': 80
                }
            },
            event_bus=event_bus
        )
        
        logger.info(f"Created data pipeline: {str(pipeline)}")
        
        # Generate test data
        symbols = ['EUR/USD', 'GBP/USD', 'AAPL', 'MSFT']
        test_data = generate_test_data(symbols)
        
        # Process single DataFrame
        symbol = 'EUR/USD'
        df = test_data[symbol]
        processed_df, metadata = pipeline.process(df, symbol=symbol, source='test')
        
        logger.info(f"Processed {symbol}: {len(processed_df)} rows")
        if verbose:
            logger.info(f"Processing metadata: {metadata}")
        
        # Check if the processing worked
        if 'quality_score' in metadata:
            logger.info(f"Quality score for {symbol}: {metadata['quality_score']}")
        else:
            logger.error("Quality score not found in metadata")
            success = False
        
        if 'issues_detected' in metadata:
            logger.info(f"Issues detected: {metadata['issues_detected']}")
        else:
            logger.error("Issues detected not found in metadata")
            success = False
        
        # Process multiple DataFrames
        processed_dict, metadata = pipeline.process(test_data, source='test')
        
        logger.info(f"Processed {len(processed_dict)} symbols")
        if verbose:
            logger.info(f"Multi-processing metadata: {metadata}")
        
        # Check if all symbols were processed
        if len(processed_dict) != len(test_data):
            logger.error(f"Not all symbols were processed: {len(processed_dict)} != {len(test_data)}")
            success = False
        
        # Test the compatibility layer
        from trading_bot.data.processor_migration import clean_data, validate_data_quality
        
        # Test clean_data
        cleaned_df = clean_data(df, symbol=symbol, source='test')
        logger.info(f"Cleaned {symbol} with compatibility function: {len(cleaned_df)} rows")
        
        # Test validate_data_quality
        validated_df, quality_metrics = validate_data_quality(df, symbol=symbol, source='test')
        logger.info(f"Validated {symbol} with compatibility function: {len(validated_df)} rows")
        if verbose and quality_metrics:
            logger.info(f"Quality metrics: {quality_metrics}")
    
    except ImportError as e:
        logger.error(f"Import error during data pipeline testing: {str(e)}")
        success = False
    except Exception as e:
        logger.error(f"Unexpected error during data pipeline testing: {str(e)}")
        success = False
    
    return success

def test_dashboard_integration(verbose: bool = False) -> bool:
    """
    Test the dashboard integration.
    
    Args:
        verbose: Whether to print verbose output
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Testing dashboard integration...")
    
    success = True
    
    try:
        # Import components
        from trading_bot.data.data_pipeline import create_data_pipeline
        from trading_bot.data.dashboard_connector import connect_to_dashboard
        from trading_bot.core.event_system import EventBus, Event, EventType
        
        # Create event bus
        event_bus = EventBus()
        
        # Create pipeline
        pipeline = create_data_pipeline(
            config={
                'cleaning': {
                    'standardize_timestamps': True,
                    'normalize_volume': True
                },
                'quality': {
                    'auto_repair': True,
                    'quality_threshold': 80
                }
            },
            event_bus=event_bus
        )
        
        # Connect dashboard
        dashboard_connector = connect_to_dashboard(
            event_bus=event_bus,
            pipeline=pipeline,
            auto_start=False  # Don't start automatic updates
        )
        
        logger.info("Connected dashboard")
        
        # Register for dashboard events
        dashboard_updates = []
        
        def on_dashboard_update(event):
            if event.event_type == EventType.DASHBOARD_UPDATE:
                dashboard_updates.append(event.data)
                logger.info("Received dashboard update")
        
        event_bus.register(EventType.DASHBOARD_UPDATE, on_dashboard_update)
        
        # Generate test data and process
        symbols = ['EUR/USD', 'AAPL']
        test_data = generate_test_data(symbols)
        
        # Process data
        for symbol, df in test_data.items():
            processed_df, metadata = pipeline.process(df, symbol=symbol, source='test')
            
            # Publish data processed event
            event = Event(
                event_type=EventType.DATA_PROCESSED,
                data={
                    'symbol': symbol,
                    'metadata': metadata
                }
            )
            event_bus.publish(event)
        
        # Manually trigger dashboard update
        dashboard_connector._push_dashboard_update()
        
        # Give time for events to be processed
        time.sleep(0.1)
        
        # Check if dashboard updates were received
        if dashboard_updates:
            logger.info(f"Received {len(dashboard_updates)} dashboard updates")
            if verbose and dashboard_updates:
                logger.info(f"Dashboard metrics: {dashboard_updates[0]}")
        else:
            logger.error("No dashboard updates received")
            success = False
        
        # Get quality metrics from connector
        metrics = dashboard_connector.get_current_metrics()
        if metrics:
            logger.info("Successfully retrieved quality metrics from dashboard connector")
            if verbose:
                logger.info(f"Current metrics: {metrics}")
        else:
            logger.error("Failed to retrieve quality metrics")
            success = False
        
        # Get quality summary
        summary = dashboard_connector.get_quality_summary()
        if summary:
            logger.info("Successfully retrieved quality summary")
            logger.info(f"Overall quality score: {summary.get('overall_quality', 0)}")
        else:
            logger.error("Failed to retrieve quality summary")
            success = False
    
    except ImportError as e:
        logger.error(f"Import error during dashboard testing: {str(e)}")
        success = False
    except Exception as e:
        logger.error(f"Unexpected error during dashboard testing: {str(e)}")
        success = False
    
    return success

def test_full_integration(verbose: bool = False) -> bool:
    """
    Test a full integration of strategies, data pipeline, and event system.
    
    Args:
        verbose: Whether to print verbose output
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Testing full integration...")
    
    success = True
    
    try:
        # Import components
        from trading_bot.strategies.factory.strategy_registry import StrategyRegistry
        from trading_bot.data.data_pipeline import create_data_pipeline
        from trading_bot.core.event_system import EventBus, Event, EventType
        
        # Create event bus
        event_bus = EventBus()
        
        # Create pipeline
        pipeline = create_data_pipeline(
            config={
                'cleaning': {
                    'standardize_timestamps': True,
                    'normalize_volume': True
                },
                'quality': {
                    'auto_repair': True
                }
            },
            event_bus=event_bus
        )
        
        # Record generated signals
        signals_generated = []
        
        def on_signal_generated(event):
            if event.event_type == EventType.SIGNAL_GENERATED and 'signal' in event.data:
                signals_generated.append(event.data['signal'])
                logger.info(f"Signal generated: {event.data['signal'].get('symbol')} {event.data['signal'].get('signal_type')}")
        
        event_bus.register(EventType.SIGNAL_GENERATED, on_signal_generated)
        
        # Get ForexTrendFollowingStrategy
        strategy_class = StrategyRegistry.get_strategy_class("ForexTrendFollowingStrategy")
        if not strategy_class:
            logger.error("ForexTrendFollowingStrategy not found")
            return False
        
        # Create strategy instance
        strategy = strategy_class()
        
        # Register with event bus
        strategy.register_events(event_bus)
        
        # Generate test data
        symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY']
        test_data = generate_test_data(symbols)
        
        # Process data through pipeline
        processed_data = {}
        for symbol, df in test_data.items():
            processed_df, _ = pipeline.process(df, symbol=symbol, source='test')
            processed_data[symbol] = processed_df
            
            # Simulate timeframe completed event
            event = Event(
                event_type=EventType.TIMEFRAME_COMPLETED,
                data={
                    'symbol': symbol,
                    'timeframe': 'daily',
                    'timestamp': datetime.now()
                }
            )
            
            # Mock market data response
            def mock_request_handler(request_event):
                if request_event.event_type == EventType.MARKET_DATA_REQUEST:
                    return {
                        'data': {symbol: processed_df}
                    }
                return None
            
            # Temporarily override request handler
            original_request = event_bus.request
            event_bus.request = mock_request_handler
            
            # Publish event
            event_bus.publish(event)
            
            # Restore original request handler
            event_bus.request = original_request
        
        # Give time for events to be processed
        time.sleep(0.1)
        
        # Check if signals were generated
        if signals_generated:
            logger.info(f"Generated {len(signals_generated)} signals")
            if verbose:
                for signal in signals_generated:
                    logger.info(f"Signal: {signal.get('symbol')} {signal.get('signal_type')} @ {signal.get('entry_price')}")
        else:
            logger.warning("No signals were generated")
            # Not marking as failure as this depends on the data
        
        logger.info("Full integration test completed")
    
    except ImportError as e:
        logger.error(f"Import error during full integration testing: {str(e)}")
        success = False
    except Exception as e:
        logger.error(f"Unexpected error during full integration testing: {str(e)}")
        success = False
    
    return success

def expand_asset_classes(asset_classes: List[str], base_path: str) -> bool:
    """
    Expand to additional asset classes.
    
    Args:
        asset_classes: List of asset classes to expand to
        base_path: Base path of the project
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Expanding to asset classes: {', '.join(asset_classes)}")
    
    try:
        import subprocess
        
        # Run the expand_asset_classes.py script
        cmd = [
            'python3', 
            os.path.join(base_path, 'expand_asset_classes.py'),
            '--path', base_path,
            '--asset-classes'
        ] + asset_classes
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Error expanding asset classes: {stderr.decode('utf-8')}")
            return False
        
        logger.info(f"Asset class expansion output: {stdout.decode('utf-8')}")
        return True
    
    except Exception as e:
        logger.error(f"Error during asset class expansion: {str(e)}")
        return False

def main() -> int:
    """
    Main entry point.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(description="Trading Bot Integration Test")
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--test', choices=['strategy', 'data', 'dashboard', 'full', 'expand', 'all'],
                      default='all', help='Test to run')
    parser.add_argument('--expand-classes', nargs='+', 
                      choices=['stocks', 'crypto', 'options', 'all'],
                      default=['all'], help='Asset classes to expand to')
    
    args = parser.parse_args()
    
    logger.info("Starting integration tests")
    
    tests_passed = True
    
    # Determine which tests to run
    run_strategy = args.test in ['strategy', 'all']
    run_data = args.test in ['data', 'all']
    run_dashboard = args.test in ['dashboard', 'all']
    run_full = args.test in ['full', 'all']
    run_expand = args.test in ['expand', 'all']
    
    # Run tests
    if run_strategy:
        logger.info("=== STRATEGY ORGANIZATION TEST ===")
        strategy_success = test_strategy_organization(args.verbose)
        tests_passed = tests_passed and strategy_success
        logger.info(f"Strategy test {'PASSED' if strategy_success else 'FAILED'}")
    
    if run_data:
        logger.info("=== DATA PIPELINE TEST ===")
        data_success = test_data_pipeline(args.verbose)
        tests_passed = tests_passed and data_success
        logger.info(f"Data pipeline test {'PASSED' if data_success else 'FAILED'}")
    
    if run_dashboard:
        logger.info("=== DASHBOARD INTEGRATION TEST ===")
        dashboard_success = test_dashboard_integration(args.verbose)
        tests_passed = tests_passed and dashboard_success
        logger.info(f"Dashboard test {'PASSED' if dashboard_success else 'FAILED'}")
    
    if run_full:
        logger.info("=== FULL INTEGRATION TEST ===")
        full_success = test_full_integration(args.verbose)
        tests_passed = tests_passed and full_success
        logger.info(f"Full integration test {'PASSED' if full_success else 'FAILED'}")
    
    if run_expand:
        logger.info("=== EXPAND ASSET CLASSES ===")
        asset_classes = []
        if 'all' in args.expand_classes:
            asset_classes = ['stocks', 'crypto', 'options']
        else:
            asset_classes = args.expand_classes
        
        expand_success = expand_asset_classes(asset_classes, os.path.dirname(os.path.abspath(__file__)))
        tests_passed = tests_passed and expand_success
        logger.info(f"Asset class expansion {'PASSED' if expand_success else 'FAILED'}")
    
    # Print summary
    logger.info("=== INTEGRATION TEST SUMMARY ===")
    logger.info(f"Overall result: {'PASSED' if tests_passed else 'FAILED'}")
    
    if tests_passed:
        logger.info("\nAll integration tests passed! Your trading bot migration is successful.")
        logger.info("\nNext steps:")
        logger.info("1. Start using the new structure in production")
        logger.info("2. Continue expanding to additional asset classes")
        logger.info("3. Consider adding more strategies to the new structure")
        
        return 0
    else:
        logger.error("\nSome integration tests failed. Please review the logs and fix the issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
