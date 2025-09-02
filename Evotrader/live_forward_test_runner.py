#!/usr/bin/env python3
"""
Live Forward Test Runner

This module implements a paper trading environment for forward testing
trading strategies against proprietary firm evaluation criteria.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import datetime
from typing import Dict, List, Any, Optional, Union
import logging
import yaml
import importlib.util
from pathlib import Path
import time

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import project modules
from test_status_tracker import TestStatusTracker
try:
    from prop_strategy_registry import PropStrategyRegistry
except ImportError:
    print("PropStrategyRegistry not available. Some functionality will be limited.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('live_forward_tester')


class PropFirmForwardTester:
    """
    Forward testing environment for proprietary trading strategies.
    
    This class simulates live trading to evaluate strategy performance
    against strict proprietary firm criteria.
    """
    
    def __init__(self, 
                risk_profile_path: Optional[str] = None,
                output_dir: str = "./forward_test_results",
                registry_path: Optional[str] = None):
        """
        Initialize the forward tester.
        
        Args:
            risk_profile_path: Path to risk profile YAML file
            output_dir: Directory for output files
            registry_path: Path to strategy registry
        """
        self.output_dir = output_dir
        self.risk_profile_path = risk_profile_path
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load risk profile
        self.risk_profile = {}
        if risk_profile_path and os.path.exists(risk_profile_path):
            with open(risk_profile_path, 'r') as f:
                self.risk_profile = yaml.safe_load(f)
        
        # Initialize defaults from risk profile
        self.forward_test_days = self.risk_profile.get('forward_testing', {}).get('target_days', 10)
        self.min_test_days = self.risk_profile.get('forward_testing', {}).get('min_days', 5)
        self.required_passing_days = self.risk_profile.get('forward_testing', {}).get('required_passing_days', 4)
        
        # Initialize registry connection if path provided
        self.registry = None
        if registry_path:
            try:
                self.registry = PropStrategyRegistry(registry_path)
            except Exception as e:
                logger.error(f"Failed to initialize registry: {e}")
        
        # Initialize test trackers
        self.active_tests = {}  # strategy_id -> test_status_tracker
        self.completed_tests = {}  # strategy_id -> test_results
        
        # Initialize strategy storage
        self.strategies = {}  # strategy_id -> strategy_instance
        
        # Initialize market data cache
        self.market_data_cache = {}  # symbol -> dataframe
        
        logger.info("Forward tester initialized")
    
    def load_strategy(self, strategy_path: str, strategy_id: Optional[str] = None) -> str:
        """
        Load a strategy from file.
        
        Args:
            strategy_path: Path to strategy file
            strategy_id: Optional strategy ID
            
        Returns:
            Strategy ID
        """
        try:
            # Extract module name from path
            module_name = Path(strategy_path).stem
            
            # Load module
            spec = importlib.util.spec_from_file_location(module_name, strategy_path)
            strategy_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(strategy_module)
            
            # Find strategy class in module
            strategy_class = None
            for attr_name in dir(strategy_module):
                attr = getattr(strategy_module, attr_name)
                if isinstance(attr, type) and hasattr(attr, 'generate_signals'):
                    strategy_class = attr
                    break
            
            if strategy_class is None:
                raise ValueError(f"No valid strategy class found in {strategy_path}")
            
            # Create strategy instance
            strategy = strategy_class()
            
            # Generate ID if not provided
            if strategy_id is None:
                strategy_id = f"{module_name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Store strategy
            self.strategies[strategy_id] = strategy
            
            logger.info(f"Loaded strategy {strategy_id} from {strategy_path}")
            
            return strategy_id
            
        except Exception as e:
            logger.error(f"Failed to load strategy from {strategy_path}: {e}")
            raise
    
    def load_strategy_from_registry(self, strategy_id: str) -> bool:
        """
        Load a strategy from the registry.
        
        Args:
            strategy_id: ID of strategy in registry
            
        Returns:
            True if successful, False otherwise
        """
        if self.registry is None:
            logger.error("Registry not initialized")
            return False
        
        try:
            # Get strategy from registry
            strategy_data = self.registry.get_strategy(strategy_id)
            
            if strategy_data is None:
                logger.error(f"Strategy not found in registry: {strategy_id}")
                return False
            
            # Create strategy instance
            strategy_type = strategy_data['strategy_type']
            parameters = strategy_data['parameters']
            
            # Dynamic import of strategy module
            module_name = f"advanced_strategies"  # Assuming strategies are in this module
            
            try:
                module = importlib.import_module(module_name)
                strategy_class = getattr(module, strategy_type)
                strategy = strategy_class(**parameters)
            except (ImportError, AttributeError) as e:
                logger.error(f"Failed to load strategy class: {e}")
                return False
            
            # Store strategy
            self.strategies[strategy_id] = strategy
            
            # Update registry status
            self.registry.update_strategy_status(
                strategy_id=strategy_id,
                new_status=self.registry.STATUS_UNDER_TEST,
                reason="Starting forward test"
            )
            
            logger.info(f"Loaded strategy {strategy_id} from registry")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load strategy from registry: {e}")
            return False
    
    def get_market_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Get market data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            days: Number of days of data to retrieve
            
        Returns:
            DataFrame with market data
        """
        try:
            # Check cache first
            if symbol in self.market_data_cache:
                logger.info(f"Using cached data for {symbol}")
                return self.market_data_cache[symbol]
            
            # Calculate date range
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=days)
            
            # Try to import yfinance
            try:
                import yfinance as yf
                
                # Get data from Yahoo Finance
                data = yf.download(
                    symbol,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    progress=False
                )
                
                # Rename columns
                data.columns = [col.lower() for col in data.columns]
                
                # Cache data
                self.market_data_cache[symbol] = data
                
                logger.info(f"Downloaded market data for {symbol}")
                
                return data
                
            except ImportError:
                logger.warning("yfinance not available. Using random data for testing.")
                
                # Generate random data
                index = pd.date_range(start=start_date, end=end_date, freq='B')
                data = pd.DataFrame(index=index)
                
                # Generate OHLCV data
                base_price = 100.0
                daily_volatility = 0.02
                
                # Start with a random price
                price = base_price
                
                # Generate OHLCV data
                data['open'] = 0.0
                data['high'] = 0.0
                data['low'] = 0.0
                data['close'] = 0.0
                data['volume'] = 0.0
                
                for i, date in enumerate(index):
                    # Random walk for closing price
                    change_pct = np.random.normal(0, daily_volatility)
                    price = price * (1 + change_pct)
                    
                    # Generate OHLC
                    daily_volatility_pct = daily_volatility * price
                    
                    open_price = price * (1 + np.random.normal(0, daily_volatility / 2))
                    high_price = max(open_price, price) * (1 + abs(np.random.normal(0, daily_volatility)))
                    low_price = min(open_price, price) * (1 - abs(np.random.normal(0, daily_volatility)))
                    
                    data.loc[date, 'open'] = open_price
                    data.loc[date, 'high'] = high_price
                    data.loc[date, 'low'] = low_price
                    data.loc[date, 'close'] = price
                    data.loc[date, 'volume'] = np.random.randint(100000, 1000000)
                
                # Cache data
                self.market_data_cache[symbol] = data
                
                logger.info(f"Generated random market data for {symbol}")
                
                return data
                
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            raise
    
    def start_test(self, 
                  strategy_id: str, 
                  symbols: List[str],
                  days: Optional[int] = None,
                  initial_equity: float = 100000.0) -> bool:
        """
        Start a forward test for a strategy.
        
        Args:
            strategy_id: ID of strategy to test
            symbols: List of symbols to test on
            days: Number of days to test for (default from risk profile)
            initial_equity: Initial equity value
            
        Returns:
            True if started successfully, False otherwise
        """
        if strategy_id not in self.strategies:
            logger.error(f"Strategy not found: {strategy_id}")
            return False
        
        if days is None:
            days = self.forward_test_days
        
        # Create test directory
        test_dir = os.path.join(self.output_dir, f"{strategy_id}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
        os.makedirs(test_dir, exist_ok=True)
        
        # Initialize test tracker
        tracker = TestStatusTracker(
            risk_profile_path=self.risk_profile_path,
            output_dir=test_dir,
            strategy_id=strategy_id
        )
        
        # Start tracking
        tracker.start_test(initial_equity)
        
        # Get market data for symbols
        market_data = {}
        for symbol in symbols:
            try:
                data = self.get_market_data(symbol, days=days+30)  # Get extra data for indicators
                market_data[symbol] = data
            except Exception as e:
                logger.error(f"Failed to get market data for {symbol}: {e}")
                return False
        
        # Store test information
        self.active_tests[strategy_id] = {
            'tracker': tracker,
            'symbols': symbols,
            'market_data': market_data,
            'days_remaining': days,
            'start_date': datetime.datetime.now().date(),
            'last_update': datetime.datetime.now().date()
        }
        
        logger.info(f"Started forward test for strategy {strategy_id} on symbols {symbols}")
        
        return True
    
    def process_day(self, strategy_id: str, day: Optional[datetime.date] = None) -> bool:
        """
        Process a single day for a forward test.
        
        Args:
            strategy_id: ID of strategy to process
            day: Date to process (default: today)
            
        Returns:
            True if processed successfully, False otherwise
        """
        if strategy_id not in self.active_tests:
            logger.error(f"No active test for strategy {strategy_id}")
            return False
        
        if strategy_id not in self.strategies:
            logger.error(f"Strategy not found: {strategy_id}")
            return False
        
        # Get test information
        test_info = self.active_tests[strategy_id]
        tracker = test_info['tracker']
        symbols = test_info['symbols']
        market_data = test_info['market_data']
        
        # Use today if no date provided
        if day is None:
            day = datetime.datetime.now().date()
        
        # Check if we have market data for this day
        day_data = {}
        for symbol in symbols:
            if day in market_data[symbol].index:
                day_data[symbol] = market_data[symbol].loc[:day]
            else:
                logger.warning(f"No market data for {symbol} on {day}")
                return False
        
        # Get strategy
        strategy = self.strategies[strategy_id]
        
        # Process each symbol
        daily_pnl_pct = 0.0
        trades = []
        
        for symbol in symbols:
            # Get signals
            try:
                data = day_data[symbol]
                signals = strategy.generate_signals(data)
                
                # Execute orders and calculate P&L
                latest_price = data.iloc[-1]['close']
                prev_price = data.iloc[-2]['close'] if len(data) > 1 else latest_price
                
                # Simple execution model: calculate daily P&L based on signal
                if len(signals) > 0:
                    latest_signal = signals.iloc[-1]
                    
                    # Check if signal is numeric or has a 'position' column
                    if isinstance(latest_signal, (int, float)):
                        position = latest_signal
                    elif hasattr(latest_signal, 'position'):
                        position = latest_signal.position
                    else:
                        position = 0
                    
                    # Calculate P&L for this symbol
                    price_change_pct = (latest_price - prev_price) / prev_price * 100
                    symbol_pnl_pct = position * price_change_pct
                    
                    # Simple position sizing: equal weight across symbols
                    symbol_weight = 1.0 / len(symbols)
                    daily_pnl_pct += symbol_pnl_pct * symbol_weight
                    
                    # Create trade record if position changed
                    if len(signals) > 1:
                        prev_signal = signals.iloc[-2]
                        prev_position = prev_signal if isinstance(prev_signal, (int, float)) else prev_signal.position if hasattr(prev_signal, 'position') else 0
                        
                        if position != prev_position:
                            trade = {
                                'symbol': symbol,
                                'entry_time': data.index[-1],
                                'exit_time': data.index[-1],
                                'direction': position,
                                'entry_price': prev_price,
                                'exit_price': latest_price,
                                'pnl': symbol_pnl_pct * symbol_weight * 100,  # Convert to dollars assuming $100 position
                                'return_pct': symbol_pnl_pct
                            }
                            trades.append(trade)
                
            except Exception as e:
                logger.error(f"Error processing {symbol} for strategy {strategy_id}: {e}")
                continue
        
        # Update tracker with daily P&L
        tracker.track_daily_pnl(day, daily_pnl_pct)
        
        # Record trades
        for trade in trades:
            tracker.record_trade(trade)
        
        # Check compliance
        compliance = self.check_compliance(strategy_id)
        
        # Update test info
        test_info['days_remaining'] -= 1
        test_info['last_update'] = day
        
        # Check if test is complete
        if test_info['days_remaining'] <= 0 or not compliance['within_limits']:
            self.finish_test(strategy_id)
        
        logger.info(f"Processed day {day} for strategy {strategy_id}: P&L = {daily_pnl_pct:.2f}%")
        
        return True
    
    def check_compliance(self, strategy_id: str) -> Dict[str, Any]:
        """
        Check if strategy is compliant with prop firm limits.
        
        Args:
            strategy_id: ID of strategy to check
            
        Returns:
            Compliance results
        """
        if strategy_id not in self.active_tests:
            logger.error(f"No active test for strategy {strategy_id}")
            return {'valid': False, 'within_limits': False}
        
        # Get test information
        test_info = self.active_tests[strategy_id]
        tracker = test_info['tracker']
        
        # Check thresholds
        threshold_check = tracker.check_thresholds()
        
        # Determine if strategy is within limits
        within_limits = True
        
        # Check max drawdown
        if not threshold_check.get('meets_max_drawdown', False):
            logger.warning(f"Strategy {strategy_id} exceeded maximum drawdown")
            within_limits = False
        
        # Check daily loss limit
        if not threshold_check.get('meets_daily_loss', False):
            logger.warning(f"Strategy {strategy_id} exceeded daily loss limit")
            within_limits = False
        
        return {
            'valid': threshold_check.get('valid', False),
            'within_limits': within_limits,
            'threshold_check': threshold_check
        }
    
    def generate_daily_report(self, strategy_id: str) -> str:
        """
        Generate a daily report for a strategy.
        
        Args:
            strategy_id: ID of strategy
            
        Returns:
            Path to report file
        """
        if strategy_id not in self.active_tests:
            logger.error(f"No active test for strategy {strategy_id}")
            return ""
        
        # Get test information
        test_info = self.active_tests[strategy_id]
        tracker = test_info['tracker']
        
        # Generate summary
        summary = tracker.generate_summary()
        
        # Save results
        day = test_info['last_update'].strftime('%Y%m%d')
        report_path = tracker.save_results(f"{strategy_id}_report_{day}.json")
        
        # Generate equity curve plot
        plot_path = tracker.plot_equity_curve(f"{strategy_id}_equity_{day}.png")
        
        return report_path
    
    def finish_test(self, strategy_id: str) -> Dict[str, Any]:
        """
        Finish a forward test and generate final report.
        
        Args:
            strategy_id: ID of strategy
            
        Returns:
            Test results
        """
        if strategy_id not in self.active_tests:
            logger.error(f"No active test for strategy {strategy_id}")
            return {}
        
        # Get test information
        test_info = self.active_tests[strategy_id]
        tracker = test_info['tracker']
        
        # Generate final summary
        summary = tracker.generate_summary()
        
        # Save results
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = tracker.save_results(f"{strategy_id}_final_{timestamp}.json")
        
        # Generate equity curve plot
        plot_path = tracker.plot_equity_curve(f"{strategy_id}_final_{timestamp}.png")
        
        # Determine if test passed
        passed = summary['threshold_check'].get('meets_max_drawdown', False) and \
                summary['threshold_check'].get('meets_daily_loss', False)
        
        # Update registry if available
        if self.registry is not None:
            try:
                new_status = self.registry.STATUS_PROMOTED if passed else self.registry.STATUS_REJECTED
                reason = "Passed forward test" if passed else "Failed forward test"
                
                self.registry.update_strategy_status(
                    strategy_id=strategy_id,
                    new_status=new_status,
                    reason=reason
                )
                
                # Record performance
                self.registry.record_performance(
                    strategy_id=strategy_id,
                    scenario="forward_test",
                    metrics=summary,
                    score=summary.get('total_return_pct', 0) / max(summary.get('max_drawdown', 1), 1),
                    passes_evaluation=passed
                )
                
            except Exception as e:
                logger.error(f"Failed to update registry: {e}")
        
        # Move test to completed
        self.completed_tests[strategy_id] = {
            'summary': summary,
            'results_path': results_path,
            'plot_path': plot_path,
            'passed': passed
        }
        
        # Remove from active tests
        del self.active_tests[strategy_id]
        
        logger.info(f"Finished forward test for strategy {strategy_id}: {'PASSED' if passed else 'FAILED'}")
        
        return self.completed_tests[strategy_id]


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Forward test a trading strategy")
    
    parser.add_argument(
        "--strategy", 
        type=str, 
        required=True,
        help="Path to strategy file or strategy ID in registry"
    )
    
    parser.add_argument(
        "--symbols", 
        type=str, 
        nargs="+",
        default=["SPY"],
        help="Symbols to test on"
    )
    
    parser.add_argument(
        "--days", 
        type=int, 
        default=10,
        help="Number of days to test"
    )
    
    parser.add_argument(
        "--risk-profile", 
        type=str, 
        default="./prop_risk_profile.yaml",
        help="Path to risk profile YAML file"
    )
    
    parser.add_argument(
        "--registry", 
        type=str, 
        default=None,
        help="Path to strategy registry"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="./forward_test_results",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Initialize forward tester
    tester = PropFirmForwardTester(
        risk_profile_path=args.risk_profile,
        output_dir=args.output,
        registry_path=args.registry
    )
    
    # Load strategy
    strategy_id = args.strategy
    
    # Check if this is a path or ID
    if os.path.exists(args.strategy):
        # Load from file
        strategy_id = tester.load_strategy(args.strategy)
    else:
        # Try to load from registry
        if tester.registry is None:
            print("Error: No registry available and strategy path does not exist")
            sys.exit(1)
        
        if not tester.load_strategy_from_registry(args.strategy):
            print(f"Error: Failed to load strategy {args.strategy} from registry")
            sys.exit(1)
    
    # Start test
    if not tester.start_test(strategy_id, args.symbols, args.days):
        print(f"Error: Failed to start test for {strategy_id}")
        sys.exit(1)
    
    # Process days
    for day in range(args.days):
        # Generate date for the test day
        test_date = datetime.datetime.now().date() - datetime.timedelta(days=args.days-day-1)
        
        # Process day
        if not tester.process_day(strategy_id, test_date):
            print(f"Error: Failed to process day {test_date} for {strategy_id}")
            continue
        
        # Generate daily report
        tester.generate_daily_report(strategy_id)
        
        # Simulate passing of time
        if day < args.days - 1:
            time.sleep(1)  # Short sleep for simulation
    
    # Finish test
    results = tester.finish_test(strategy_id)
    
    # Print summary
    if results:
        summary = results['summary']
        print("\n==== FORWARD TEST SUMMARY ====")
        print(f"Strategy: {strategy_id}")
        print(f"Total Return: {summary['total_return_pct']:.2f}%")
        print(f"Max Drawdown: {summary['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
        print(f"Profitable Days: {summary['profitable_days_pct']:.1f}%")
        print(f"Test Result: {'PASSED' if results['passed'] else 'FAILED'}")
        print(f"Results saved to: {results['results_path']}")
        print(f"Equity curve plot: {results['plot_path']}")
    else:
        print("No results available")
