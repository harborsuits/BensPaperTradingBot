"""
Component Testing Framework

Provides utilities for testing individual strategy components.
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import unittest
import tempfile
import uuid

from trading_bot.strategies.modular_strategy_system import (
    StrategyComponent, ComponentType, 
    SignalGeneratorComponent, FilterComponent, 
    PositionSizerComponent, ExitManagerComponent
)
from trading_bot.strategies.components.component_registry import get_component_registry
from trading_bot.strategies.base_strategy import SignalType

logger = logging.getLogger(__name__)

class ComponentTestCase:
    """Base class for component test cases."""
    
    def __init__(self, component: StrategyComponent, test_name: str = ""):
        """
        Initialize test case
        
        Args:
            component: Component to test
            test_name: Test case name
        """
        self.component = component
        self.test_name = test_name or f"Test_{component.component_id}"
        self.component_type = component.component_type
        self.assertions = []
        self.test_data = {}
        self.expected_results = {}
        self.actual_results = {}
        self.passed = False
        self.error_message = ""
    
    def with_data(self, symbol: str, data: pd.DataFrame) -> 'ComponentTestCase':
        """
        Add test data for a symbol
        
        Args:
            symbol: Symbol
            data: Market data
            
        Returns:
            Self for chaining
        """
        self.test_data[symbol] = data
        return self
    
    def with_context(self, context: Dict[str, Any]) -> 'ComponentTestCase':
        """
        Add test context
        
        Args:
            context: Test context
            
        Returns:
            Self for chaining
        """
        if 'context' not in self.test_data:
            self.test_data['context'] = {}
        
        self.test_data['context'].update(context)
        return self
    
    def expect_result(self, expected_result: Any) -> 'ComponentTestCase':
        """
        Set expected test result
        
        Args:
            expected_result: Expected result
            
        Returns:
            Self for chaining
        """
        self.expected_results['result'] = expected_result
        return self
    
    def expect_signal(self, symbol: str, expected_signal: SignalType) -> 'ComponentTestCase':
        """
        Set expected signal for symbol
        
        Args:
            symbol: Symbol
            expected_signal: Expected signal
            
        Returns:
            Self for chaining
        """
        if 'signals' not in self.expected_results:
            self.expected_results['signals'] = {}
        
        self.expected_results['signals'][symbol] = expected_signal
        return self
    
    def expect_filter_pass(self, symbol: str) -> 'ComponentTestCase':
        """
        Expect filter to pass signal for symbol
        
        Args:
            symbol: Symbol
            
        Returns:
            Self for chaining
        """
        if 'filter_pass' not in self.expected_results:
            self.expected_results['filter_pass'] = []
        
        self.expected_results['filter_pass'].append(symbol)
        return self
    
    def expect_filter_block(self, symbol: str) -> 'ComponentTestCase':
        """
        Expect filter to block signal for symbol
        
        Args:
            symbol: Symbol
            
        Returns:
            Self for chaining
        """
        if 'filter_block' not in self.expected_results:
            self.expected_results['filter_block'] = []
        
        self.expected_results['filter_block'].append(symbol)
        return self
    
    def expect_position_size(self, symbol: str, size: float, tolerance: float = 0.01) -> 'ComponentTestCase':
        """
        Set expected position size for symbol
        
        Args:
            symbol: Symbol
            size: Expected position size
            tolerance: Tolerance for size comparison
            
        Returns:
            Self for chaining
        """
        if 'position_sizes' not in self.expected_results:
            self.expected_results['position_sizes'] = {}
        
        self.expected_results['position_sizes'][symbol] = (size, tolerance)
        return self
    
    def expect_exit(self, symbol: str, should_exit: bool = True) -> 'ComponentTestCase':
        """
        Set expected exit decision for symbol
        
        Args:
            symbol: Symbol
            should_exit: Expected exit decision
            
        Returns:
            Self for chaining
        """
        if 'exits' not in self.expected_results:
            self.expected_results['exits'] = {}
        
        self.expected_results['exits'][symbol] = should_exit
        return self
    
    def run(self) -> bool:
        """
        Run the test case
        
        Returns:
            Test result (True if passed)
        """
        try:
            # Run test based on component type
            if self.component_type == ComponentType.SIGNAL_GENERATOR:
                self._test_signal_generator()
            elif self.component_type == ComponentType.FILTER:
                self._test_filter()
            elif self.component_type == ComponentType.POSITION_SIZER:
                self._test_position_sizer()
            elif self.component_type == ComponentType.EXIT_MANAGER:
                self._test_exit_manager()
            
            # Check if all assertions passed
            self.passed = all(assertion['passed'] for assertion in self.assertions)
            return self.passed
        
        except Exception as e:
            self.error_message = str(e)
            self.passed = False
            logger.error(f"Error running test case: {e}")
            return False
    
    def _test_signal_generator(self) -> None:
        """Test signal generator component."""
        component = self.component
        
        # Check component type
        if not isinstance(component, SignalGeneratorComponent):
            raise TypeError(f"Component is not a SignalGeneratorComponent: {component.component_id}")
        
        # Create data dictionary
        data = {symbol: df for symbol, df in self.test_data.items() if isinstance(df, pd.DataFrame)}
        context = self.test_data.get('context', {})
        
        # Generate signals
        signals = component.generate_signals(data, context)
        self.actual_results['signals'] = signals
        
        # Check expected signals
        expected_signals = self.expected_results.get('signals', {})
        
        for symbol, expected_signal in expected_signals.items():
            actual_signal = signals.get(symbol)
            
            self.assertions.append({
                'type': 'signal',
                'symbol': symbol,
                'expected': expected_signal,
                'actual': actual_signal,
                'passed': actual_signal == expected_signal
            })
    
    def _test_filter(self) -> None:
        """Test filter component."""
        component = self.component
        
        # Check component type
        if not isinstance(component, FilterComponent):
            raise TypeError(f"Component is not a FilterComponent: {component.component_id}")
        
        # Create data dictionary
        data = {symbol: df for symbol, df in self.test_data.items() if isinstance(df, pd.DataFrame)}
        context = self.test_data.get('context', {})
        
        # Create input signals
        signals = {}
        for symbol in data.keys():
            signals[symbol] = SignalType.LONG  # Default to LONG for testing
        
        # Apply expected modifications
        pass_symbols = self.expected_results.get('filter_pass', [])
        block_symbols = self.expected_results.get('filter_block', [])
        
        for symbol in pass_symbols:
            if symbol in signals:
                signals[symbol] = SignalType.LONG
        
        for symbol in block_symbols:
            if symbol in signals:
                signals[symbol] = SignalType.LONG  # Will expect to be filtered to FLAT
        
        # Filter signals
        filtered_signals = component.filter_signals(signals, data, context)
        self.actual_results['filtered_signals'] = filtered_signals
        
        # Check expected pass/block
        for symbol in pass_symbols:
            if symbol in signals:
                expected = signals[symbol]
                actual = filtered_signals.get(symbol)
                
                self.assertions.append({
                    'type': 'filter_pass',
                    'symbol': symbol,
                    'expected': expected,
                    'actual': actual,
                    'passed': actual == expected
                })
        
        for symbol in block_symbols:
            if symbol in signals:
                expected = SignalType.FLAT
                actual = filtered_signals.get(symbol)
                
                self.assertions.append({
                    'type': 'filter_block',
                    'symbol': symbol,
                    'expected': expected,
                    'actual': actual,
                    'passed': actual == expected
                })
    
    def _test_position_sizer(self) -> None:
        """Test position sizer component."""
        component = self.component
        
        # Check component type
        if not isinstance(component, PositionSizerComponent):
            raise TypeError(f"Component is not a PositionSizerComponent: {component.component_id}")
        
        # Create data dictionary
        data = {symbol: df for symbol, df in self.test_data.items() if isinstance(df, pd.DataFrame)}
        context = self.test_data.get('context', {})
        
        # Create input signals
        signals = {}
        for symbol in data.keys():
            signals[symbol] = SignalType.LONG  # Default to LONG for testing
        
        # Calculate position sizes
        position_sizes = component.calculate_position_sizes(signals, data, context)
        self.actual_results['position_sizes'] = position_sizes
        
        # Check expected position sizes
        expected_sizes = self.expected_results.get('position_sizes', {})
        
        for symbol, (expected_size, tolerance) in expected_sizes.items():
            actual_size = position_sizes.get(symbol, 0.0)
            
            self.assertions.append({
                'type': 'position_size',
                'symbol': symbol,
                'expected': expected_size,
                'actual': actual_size,
                'passed': abs(actual_size - expected_size) <= tolerance
            })
    
    def _test_exit_manager(self) -> None:
        """Test exit manager component."""
        component = self.component
        
        # Check component type
        if not isinstance(component, ExitManagerComponent):
            raise TypeError(f"Component is not an ExitManagerComponent: {component.component_id}")
        
        # Create data dictionary
        data = {symbol: df for symbol, df in self.test_data.items() if isinstance(df, pd.DataFrame)}
        context = self.test_data.get('context', {})
        
        # Create positions
        positions = {}
        for symbol in data.keys():
            # Create dummy position
            positions[symbol] = {
                'symbol': symbol,
                'position_type': 'long',
                'entry_price': data[symbol]['close'].iloc[-10],
                'current_price': data[symbol]['close'].iloc[-1],
                'entry_time': datetime.now() - timedelta(days=1),
                'quantity': 1
            }
        
        # Calculate exits
        exits = component.calculate_exits(positions, data, context)
        self.actual_results['exits'] = exits
        
        # Check expected exits
        expected_exits = self.expected_results.get('exits', {})
        
        for symbol, expected_exit in expected_exits.items():
            actual_exit = exits.get(symbol, False)
            
            self.assertions.append({
                'type': 'exit',
                'symbol': symbol,
                'expected': expected_exit,
                'actual': actual_exit,
                'passed': actual_exit == expected_exit
            })
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get test results
        
        Returns:
            Test results
        """
        results = {
            'test_name': self.test_name,
            'component_id': self.component.component_id,
            'component_type': self.component_type.name,
            'passed': self.passed,
            'assertions': self.assertions,
            'actual_results': self.actual_results
        }
        
        if self.error_message:
            results['error'] = self.error_message
        
        return results
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert test case to dictionary
        
        Returns:
            Dictionary representation
        """
        test_dict = {
            'test_name': self.test_name,
            'component_id': self.component.component_id,
            'component_type': self.component_type.name,
            'expected_results': self.expected_results
        }
        
        # Don't serialize test data directly
        test_dict['has_test_data'] = {symbol: True for symbol in self.test_data.keys()}
        
        if 'context' in self.test_data:
            test_dict['context'] = self.test_data['context']
        
        return test_dict
    
    def save(self, file_path: str) -> bool:
        """
        Save test case to file
        
        Args:
            file_path: File path
            
        Returns:
            Success flag
        """
        try:
            test_dict = self.to_dict()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(test_dict, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error saving test case: {e}")
            return False

class ComponentTestSuite:
    """Test suite for component testing."""
    
    def __init__(self, name: str = ""):
        """
        Initialize test suite
        
        Args:
            name: Test suite name
        """
        self.name = name or f"ComponentTestSuite_{uuid.uuid4().hex[:8]}"
        self.test_cases = []
        self.test_results = []
    
    def add_test_case(self, test_case: ComponentTestCase) -> None:
        """
        Add test case to suite
        
        Args:
            test_case: Test case
        """
        self.test_cases.append(test_case)
    
    def run(self) -> Dict[str, Any]:
        """
        Run all test cases
        
        Returns:
            Test results
        """
        self.test_results = []
        
        for test_case in self.test_cases:
            test_case.run()
            self.test_results.append(test_case.get_results())
        
        return self.get_results()
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get test suite results
        
        Returns:
            Test suite results
        """
        # Calculate statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['passed'])
        
        total_assertions = sum(len(result['assertions']) for result in self.test_results)
        passed_assertions = sum(
            sum(1 for assertion in result['assertions'] if assertion['passed']) 
            for result in self.test_results
        )
        
        return {
            'name': self.name,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'total_assertions': total_assertions,
            'passed_assertions': passed_assertions,
            'failed_assertions': total_assertions - passed_assertions,
            'assertion_pass_rate': passed_assertions / total_assertions if total_assertions > 0 else 0,
            'test_results': self.test_results
        }
    
    def load_test_data(self, data_dir: str) -> None:
        """
        Load test data from directory
        
        Args:
            data_dir: Data directory
        """
        pass  # Implement if needed
    
    def save_results(self, file_path: str) -> bool:
        """
        Save test results to file
        
        Args:
            file_path: File path
            
        Returns:
            Success flag
        """
        try:
            results = self.get_results()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error saving test results: {e}")
            return False

class ComponentTestRunner:
    """Test runner for component testing."""
    
    def __init__(self, test_dir: str = None, data_dir: str = None):
        """
        Initialize test runner
        
        Args:
            test_dir: Test directory
            data_dir: Data directory
        """
        self.test_dir = test_dir or os.path.join(
            os.path.dirname(__file__), 
            'test_cases'
        )
        
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(__file__), 
            'test_data'
        )
        
        # Ensure directories exist
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Component registry
        self.registry = get_component_registry()
    
    def run_test_suite(self, test_suite: ComponentTestSuite) -> Dict[str, Any]:
        """
        Run test suite
        
        Args:
            test_suite: Test suite
            
        Returns:
            Test results
        """
        return test_suite.run()
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all tests in test directory
        
        Returns:
            Test results
        """
        test_suite = ComponentTestSuite("AllTests")
        
        # Load and run all test cases
        for root, _, files in os.walk(self.test_dir):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    
                    # Load test case
                    test_case = self.load_test_case(file_path)
                    if test_case:
                        test_suite.add_test_case(test_case)
        
        # Run test suite
        return test_suite.run()
    
    def load_test_case(self, file_path: str) -> Optional[ComponentTestCase]:
        """
        Load test case from file
        
        Args:
            file_path: File path
            
        Returns:
            Test case or None if loading failed
        """
        try:
            # Load test case definition
            with open(file_path, 'r') as f:
                test_def = json.load(f)
            
            # Get component
            component_id = test_def.get('component_id')
            component_type_str = test_def.get('component_type')
            
            if not component_id or not component_type_str:
                logger.error(f"Missing component information in test case: {file_path}")
                return None
            
            # Convert string to enum
            component_type = ComponentType[component_type_str]
            
            # Get component from registry
            component = self.registry.get_component(component_type, component_id)
            if not component:
                logger.error(f"Component not found: {component_id}")
                return None
            
            # Create test case
            test_case = ComponentTestCase(
                component=component,
                test_name=test_def.get('test_name', f"Test_{component_id}")
            )
            
            # Add expected results
            expected_results = test_def.get('expected_results', {})
            
            # Signals
            for symbol, signal in expected_results.get('signals', {}).items():
                test_case.expect_signal(symbol, SignalType[signal] if isinstance(signal, str) else signal)
            
            # Filter pass/block
            for symbol in expected_results.get('filter_pass', []):
                test_case.expect_filter_pass(symbol)
            
            for symbol in expected_results.get('filter_block', []):
                test_case.expect_filter_block(symbol)
            
            # Position sizes
            for symbol, size_info in expected_results.get('position_sizes', {}).items():
                if isinstance(size_info, (list, tuple)) and len(size_info) == 2:
                    test_case.expect_position_size(symbol, size_info[0], size_info[1])
                else:
                    test_case.expect_position_size(symbol, size_info)
            
            # Exits
            for symbol, should_exit in expected_results.get('exits', {}).items():
                test_case.expect_exit(symbol, should_exit)
            
            # Load test data
            has_test_data = test_def.get('has_test_data', {})
            
            for symbol in has_test_data:
                if symbol == 'context':
                    continue
                
                # Look for data file
                data_file = os.path.join(self.data_dir, f"{symbol}.csv")
                if os.path.exists(data_file):
                    df = pd.read_csv(data_file)
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                    
                    test_case.with_data(symbol, df)
            
            # Add context
            if 'context' in test_def:
                test_case.with_context(test_def['context'])
            
            return test_case
        
        except Exception as e:
            logger.error(f"Error loading test case: {e}")
            return None
    
    def save_test_case(self, test_case: ComponentTestCase, file_name: str = None) -> str:
        """
        Save test case to file
        
        Args:
            test_case: Test case
            file_name: File name (optional)
            
        Returns:
            File path
        """
        if not file_name:
            file_name = f"{test_case.test_name}.json"
        
        file_path = os.path.join(self.test_dir, file_name)
        test_case.save(file_path)
        
        return file_path
    
    def save_test_data(self, symbol: str, data: pd.DataFrame) -> str:
        """
        Save test data to file
        
        Args:
            symbol: Symbol
            data: Market data
            
        Returns:
            File path
        """
        file_path = os.path.join(self.data_dir, f"{symbol}.csv")
        
        # Create copy of data
        df = data.copy()
        
        # Handle index
        if isinstance(df.index, pd.DatetimeIndex):
            df.reset_index(inplace=True)
        
        # Save to CSV
        df.to_csv(file_path, index=False)
        
        return file_path
