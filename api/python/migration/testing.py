#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Migration Testing Framework

This module provides tools for testing and comparing behavior
between old and new implementations during migration.
"""

import logging
import pandas as pd
import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MigrationTester:
    """
    Test framework for comparing old and new implementations during migration.
    
    This class provides methods to ensure functionality remains consistent
    and performance doesn't regress during the reorganization.
    """
    
    def __init__(self, name: str = "MigrationTester"):
        """
        Initialize the migration tester.
        
        Args:
            name: Tester name
        """
        self.name = name
        self.results = []
        self.current_test = None
    
    def test_strategy(self, 
                     strategy_name: str, 
                     test_data: Dict[str, pd.DataFrame],
                     parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Test a strategy implementation in both old and new systems.
        
        Args:
            strategy_name: Name of the strategy to test
            test_data: Test data as dictionary of DataFrames
            parameters: Optional strategy parameters
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing strategy: {strategy_name}")
        self.current_test = {
            'type': 'strategy',
            'name': strategy_name,
            'timestamp': datetime.now().isoformat(),
            'parameters': parameters,
            'data_shape': {symbol: df.shape for symbol, df in test_data.items()},
            'data_sample': {symbol: df.head(1).to_dict('records') for symbol, df in test_data.items()},
            'results': {}
        }
        
        # Test old implementation
        start_time = time.time()
        try:
            # Import directly to avoid compatibility layer
            old_module_path = f"trading_bot.strategies.{self._snake_case(strategy_name)}"
            old_signals = self._test_strategy_direct(old_module_path, strategy_name, test_data, parameters)
            old_time = time.time() - start_time
            
            self.current_test['results']['old'] = {
                'success': True,
                'time': old_time,
                'signal_count': len(old_signals),
                'signals': old_signals
            }
            logger.info(f"Old implementation: {len(old_signals)} signals in {old_time:.4f}s")
        except Exception as e:
            self.current_test['results']['old'] = {
                'success': False,
                'error': str(e),
                'time': time.time() - start_time
            }
            logger.error(f"Error testing old implementation: {str(e)}")
        
        # Test new implementation
        start_time = time.time()
        try:
            # Use compatibility layer to get new implementation
            from .strategy_compat import create_strategy
            
            # Create the strategy
            strategy = create_strategy(strategy_name, parameters=parameters)
            
            # Generate signals
            new_signals = strategy.generate_signals(test_data)
            new_time = time.time() - start_time
            
            self.current_test['results']['new'] = {
                'success': True,
                'time': new_time,
                'signal_count': len(new_signals),
                'signals': new_signals
            }
            logger.info(f"New implementation: {len(new_signals)} signals in {new_time:.4f}s")
        except Exception as e:
            self.current_test['results']['new'] = {
                'success': False,
                'error': str(e),
                'time': time.time() - start_time
            }
            logger.error(f"Error testing new implementation: {str(e)}")
        
        # Compare results
        if (self.current_test['results'].get('old', {}).get('success', False) and 
            self.current_test['results'].get('new', {}).get('success', False)):
            
            old_signals = self.current_test['results']['old']['signals']
            new_signals = self.current_test['results']['new']['signals']
            
            # Confirm strategy produces identical signals
            self.current_test['comparison'] = self._compare_signals(old_signals, new_signals)
            
            # Add benchmark of ForexTrendFollowingStrategy (our known successful implementation)
            if strategy_name != "ForexTrendFollowingStrategy":
                self._benchmark_against_reference(strategy_name, test_data, parameters)
        
        self.results.append(self.current_test)
        return self.current_test
    
    def test_data_processor(self, 
                          test_data: pd.DataFrame,
                          symbol: str = "TEST",
                          source: str = "test",
                          config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Test a data processor implementation in both old and new systems.
        
        Args:
            test_data: Test data as DataFrame
            symbol: Symbol identifier
            source: Source identifier
            config: Processor configuration
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing data processor for {symbol} from {source}")
        self.current_test = {
            'type': 'data_processor',
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'source': source,
            'data_shape': test_data.shape,
            'data_columns': list(test_data.columns),
            'config': config,
            'results': {}
        }
        
        # Test old implementation
        start_time = time.time()
        try:
            # Import directly to avoid compatibility layer
            from trading_bot.data.processors.data_cleaning_processor import DataCleaningProcessor
            
            old_processor = DataCleaningProcessor(config=config)
            old_result = old_processor.process(test_data.copy())
            old_time = time.time() - start_time
            
            self.current_test['results']['old'] = {
                'success': True,
                'time': old_time,
                'shape': old_result.shape,
                'columns': list(old_result.columns),
                'sample': old_result.head(1).to_dict('records')
            }
            logger.info(f"Old implementation: Processed {old_result.shape[0]} rows in {old_time:.4f}s")
        except Exception as e:
            self.current_test['results']['old'] = {
                'success': False,
                'error': str(e),
                'time': time.time() - start_time
            }
            logger.error(f"Error testing old implementation: {str(e)}")
        
        # Test new implementation
        start_time = time.time()
        try:
            # Use compatibility layer
            from .processor_compat import get_data_processor
            
            # Get combined processor
            processor = get_data_processor("combined", config=config)
            
            # Process data
            new_result = processor.process(test_data.copy(), symbol=symbol, source=source)
            new_time = time.time() - start_time
            
            self.current_test['results']['new'] = {
                'success': True,
                'time': new_time,
                'shape': new_result.shape,
                'columns': list(new_result.columns),
                'sample': new_result.head(1).to_dict('records')
            }
            logger.info(f"New implementation: Processed {new_result.shape[0]} rows in {new_time:.4f}s")
        except Exception as e:
            self.current_test['results']['new'] = {
                'success': False,
                'error': str(e),
                'time': time.time() - start_time
            }
            logger.error(f"Error testing new implementation: {str(e)}")
        
        # Compare results
        if (self.current_test['results'].get('old', {}).get('success', False) and 
            self.current_test['results'].get('new', {}).get('success', False)):
            
            old_df = self.current_test['results']['old'].get('sample')
            new_df = self.current_test['results']['new'].get('sample')
            
            # Compare critical columns
            critical_columns = ['open', 'high', 'low', 'close', 'volume']
            critical_columns = [col for col in critical_columns if col in old_df[0] and col in new_df[0]]
            
            if critical_columns:
                differences = []
                for col in critical_columns:
                    if abs(old_df[0][col] - new_df[0][col]) > 1e-5:
                        differences.append({
                            'column': col,
                            'old_value': old_df[0][col],
                            'new_value': new_df[0][col],
                            'difference': abs(old_df[0][col] - new_df[0][col])
                        })
                
                self.current_test['comparison'] = {
                    'identical': len(differences) == 0,
                    'differences': differences
                }
            
        self.results.append(self.current_test)
        return self.current_test
    
    def test_config(self, 
                   keys: List[str],
                   default_values: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Test configuration access in both old and new systems.
        
        Args:
            keys: List of configuration keys to test
            default_values: Default values for keys if not found
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing configuration for {len(keys)} keys")
        self.current_test = {
            'type': 'config',
            'timestamp': datetime.now().isoformat(),
            'keys': keys,
            'defaults': default_values or {},
            'results': {}
        }
        
        # Test old implementation
        start_time = time.time()
        try:
            # Import directly to avoid compatibility layer
            from trading_bot.config_manager import ConfigManager
            
            old_values = {}
            config = ConfigManager.instance()
            
            for key in keys:
                default = default_values.get(key) if default_values else None
                old_values[key] = config.get(key, default)
                
            old_time = time.time() - start_time
            
            self.current_test['results']['old'] = {
                'success': True,
                'time': old_time,
                'values': old_values
            }
            logger.info(f"Old implementation: Retrieved {len(old_values)} config values in {old_time:.4f}s")
        except Exception as e:
            self.current_test['results']['old'] = {
                'success': False,
                'error': str(e),
                'time': time.time() - start_time
            }
            logger.error(f"Error testing old implementation: {str(e)}")
        
        # Test new implementation
        start_time = time.time()
        try:
            # Use compatibility layer
            from .config_compat import get_config_value
            
            new_values = {}
            for key in keys:
                default = default_values.get(key) if default_values else None
                new_values[key] = get_config_value(key, default)
                
            new_time = time.time() - start_time
            
            self.current_test['results']['new'] = {
                'success': True,
                'time': new_time,
                'values': new_values
            }
            logger.info(f"New implementation: Retrieved {len(new_values)} config values in {new_time:.4f}s")
        except Exception as e:
            self.current_test['results']['new'] = {
                'success': False,
                'error': str(e),
                'time': time.time() - start_time
            }
            logger.error(f"Error testing new implementation: {str(e)}")
        
        # Compare results
        if (self.current_test['results'].get('old', {}).get('success', False) and 
            self.current_test['results'].get('new', {}).get('success', False)):
            
            old_values = self.current_test['results']['old']['values']
            new_values = self.current_test['results']['new']['values']
            
            # Compare values
            differences = []
            for key in keys:
                old_val = old_values.get(key)
                new_val = new_values.get(key)
                
                if old_val != new_val:
                    differences.append({
                        'key': key,
                        'old_value': old_val,
                        'new_value': new_val
                    })
                
            self.current_test['comparison'] = {
                'identical': len(differences) == 0,
                'differences': differences
            }
            
        self.results.append(self.current_test)
        return self.current_test
    
    def _test_strategy_direct(self, 
                            module_path: str, 
                            strategy_name: str, 
                            test_data: Dict[str, pd.DataFrame],
                            parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Test a strategy by direct import."""
        try:
            import importlib
            module = importlib.import_module(module_path)
            strategy_class = getattr(module, strategy_name)
            
            strategy = strategy_class(
                name=strategy_name,
                parameters=parameters or {}
            )
            
            return strategy.generate_signals(test_data)
        except ImportError:
            # Try other module paths
            for prefix in ["trading_bot.strategies.forex.", "trading_bot.strategies."]:
                try:
                    module = importlib.import_module(f"{prefix}{self._snake_case(strategy_name)}")
                    strategy_class = getattr(module, strategy_name)
                    
                    strategy = strategy_class(
                        name=strategy_name,
                        parameters=parameters or {}
                    )
                    
                    return strategy.generate_signals(test_data)
                except (ImportError, AttributeError):
                    continue
        
        raise ValueError(f"Strategy {strategy_name} not found")
    
    def _compare_signals(self, old_signals: Dict[str, Any], new_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Compare signals from old and new implementations."""
        all_symbols = set(list(old_signals.keys()) + list(new_signals.keys()))
        
        differences = []
        for symbol in all_symbols:
            if symbol not in old_signals:
                differences.append({
                    'symbol': symbol,
                    'type': 'missing_in_old'
                })
            elif symbol not in new_signals:
                differences.append({
                    'symbol': symbol,
                    'type': 'missing_in_new'
                })
            else:
                # Compare signal attributes
                old_signal = old_signals[symbol]
                new_signal = new_signals[symbol]
                
                signal_diff = {}
                
                # Check basic attributes
                for attr in ['signal_type', 'entry_price', 'stop_loss', 'take_profit', 'confidence']:
                    old_val = getattr(old_signal, attr, None)
                    new_val = getattr(new_signal, attr, None)
                    
                    if isinstance(old_val, float) and isinstance(new_val, float):
                        # Compare floats with tolerance
                        if abs(old_val - new_val) > 1e-5:
                            signal_diff[attr] = {
                                'old': old_val,
                                'new': new_val,
                                'diff': abs(old_val - new_val)
                            }
                    elif old_val != new_val:
                        signal_diff[attr] = {
                            'old': old_val,
                            'new': new_val
                        }
                
                if signal_diff:
                    differences.append({
                        'symbol': symbol,
                        'type': 'attribute_mismatch',
                        'differences': signal_diff
                    })
        
        return {
            'identical': len(differences) == 0,
            'differences': differences,
            'symbol_count_old': len(old_signals),
            'symbol_count_new': len(new_signals),
            'symbol_count_total': len(all_symbols)
        }
    
    def _benchmark_against_reference(self, 
                                   strategy_name: str, 
                                   test_data: Dict[str, pd.DataFrame],
                                   parameters: Optional[Dict[str, Any]] = None) -> None:
        """
        Benchmark strategy against our ForexTrendFollowingStrategy reference.
        
        Building upon our successful forex trend following pattern as a
        reference point for other strategies.
        """
        try:
            # Try to create the ForexTrendFollowingStrategy as reference
            from .strategy_compat import create_strategy
            
            reference = create_strategy("ForexTrendFollowingStrategy")
            if reference:
                start_time = time.time()
                reference_signals = reference.generate_signals(test_data)
                reference_time = time.time() - start_time
                
                self.current_test['reference_benchmark'] = {
                    'strategy': "ForexTrendFollowingStrategy",
                    'signal_count': len(reference_signals),
                    'time': reference_time
                }
                
                # Compare performance
                new_time = self.current_test['results']['new'].get('time', 0)
                if new_time > 0 and reference_time > 0:
                    self.current_test['reference_benchmark']['performance_ratio'] = new_time / reference_time
        except Exception as e:
            logger.warning(f"Failed to benchmark against reference: {str(e)}")
    
    def _snake_case(self, name: str) -> str:
        """Convert PascalCase to snake_case."""
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get all test results."""
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of test results."""
        summary = {
            'total_tests': len(self.results),
            'test_types': {},
            'success_rate': 0,
            'identical_results': 0,
            'performance': {
                'old_avg': 0,
                'new_avg': 0,
                'ratio': 0
            }
        }
        
        # Count by type
        for test in self.results:
            test_type = test['type']
            if test_type not in summary['test_types']:
                summary['test_types'][test_type] = 0
            summary['test_types'][test_type] += 1
        
        # Success and identity rates
        success_count = 0
        identical_count = 0
        old_time_total = 0
        new_time_total = 0
        
        for test in self.results:
            old_success = test.get('results', {}).get('old', {}).get('success', False)
            new_success = test.get('results', {}).get('new', {}).get('success', False)
            
            if old_success and new_success:
                success_count += 1
                
                if test.get('comparison', {}).get('identical', False):
                    identical_count += 1
                
                old_time = test.get('results', {}).get('old', {}).get('time', 0)
                new_time = test.get('results', {}).get('new', {}).get('time', 0)
                
                if old_time > 0:
                    old_time_total += old_time
                
                if new_time > 0:
                    new_time_total += new_time
        
        if len(self.results) > 0:
            summary['success_rate'] = success_count / len(self.results)
        
        if success_count > 0:
            summary['identical_results'] = identical_count / success_count
        
        if old_time_total > 0 and new_time_total > 0:
            summary['performance']['old_avg'] = old_time_total / success_count
            summary['performance']['new_avg'] = new_time_total / success_count
            summary['performance']['ratio'] = new_time_total / old_time_total
        
        return summary
    
    def generate_report(self, format: str = "text") -> str:
        """
        Generate a report of test results.
        
        Args:
            format: Report format ("text", "html", or "markdown")
            
        Returns:
            Report string
        """
        summary = self.get_summary()
        
        if format == "text":
            report = [
                "Migration Test Report",
                "====================",
                f"Total Tests: {summary['total_tests']}",
                f"Success Rate: {summary['success_rate']*100:.1f}%",
                f"Identical Results: {summary['identical_results']*100:.1f}%",
                "",
                "Test Types:",
            ]
            
            for test_type, count in summary['test_types'].items():
                report.append(f"  - {test_type}: {count}")
            
            report.extend([
                "",
                "Performance:",
                f"  - Old Implementation Avg: {summary['performance']['old_avg']*1000:.2f}ms",
                f"  - New Implementation Avg: {summary['performance']['new_avg']*1000:.2f}ms",
                f"  - Performance Ratio: {summary['performance']['ratio']:.2f}x",
                "",
                "Individual Tests:",
            ])
            
            for i, test in enumerate(self.results):
                report.extend([
                    f"Test #{i+1}: {test.get('name', test.get('type', 'Unknown'))}",
                    f"  - Type: {test['type']}",
                    f"  - Old Implementation: {'✓' if test.get('results', {}).get('old', {}).get('success', False) else '✗'}",
                    f"  - New Implementation: {'✓' if test.get('results', {}).get('new', {}).get('success', False) else '✗'}",
                    f"  - Identical Results: {'✓' if test.get('comparison', {}).get('identical', False) else '✗'}"
                ])
                
                if not test.get('comparison', {}).get('identical', True) and 'differences' in test.get('comparison', {}):
                    diff_count = len(test['comparison']['differences'])
                    report.append(f"  - Differences: {diff_count}")
                
                report.append("")
            
            return "\n".join(report)
            
        elif format == "markdown":
            # Implement markdown format if needed
            return "Markdown format not implemented yet"
            
        elif format == "html":
            # Implement HTML format if needed
            return "HTML format not implemented yet"
            
        else:
            return f"Unknown format: {format}"
