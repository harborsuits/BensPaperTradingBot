# Trading Bot Test Suite

This directory contains unit and integration tests for the Trading Bot application.

## Test Organization

- `test_real_time_data_processor.py`: Tests for real-time market data processing
- `test_live_trading_dashboard.py`: Tests for the live trading dashboard
- `test_run_dashboard.py`: Tests for the dashboard launcher script
- `test_advanced_regime_detector.py`: Tests for the market regime detector
- `test_backtest.py`: Tests for the backtesting system
- `test_integration.py`: Integration tests for the trading system
- `test_api_endpoints.py`: Tests for API endpoints

## Running Tests

### Prerequisites

Install the required testing dependencies:

```bash
pip install -r trading_bot/tests/requirements_tests.txt
```

### Running All Tests

Use the `run_tests.py` script to run all tests and generate a coverage report:

```bash
python trading_bot/tests/run_tests.py
```

### Running Specific Tests

To run specific test modules:

```bash
# Run a single test module
python trading_bot/tests/run_tests.py --tests test_real_time_data_processor

# Run multiple test modules
python trading_bot/tests/run_tests.py --tests test_real_time_data_processor test_live_trading_dashboard
```

### Running Tests Without Coverage

If you want to run tests without generating coverage reports:

```bash
python trading_bot/tests/run_tests.py --skip-coverage
```

### Command-Line Options

The `run_tests.py` script supports the following options:

- `--test-dir`: Directory containing test files (default: trading_bot/tests)
- `--source-dir`: Source directory to measure coverage for (default: trading_bot)
- `--report-dir`: Directory to save coverage reports (default: reports)
- `--pattern`: Pattern to match test files (default: test_*.py)
- `--skip-coverage`: Skip coverage reporting
- `--tests`: Specific test modules to run

## Coverage Reports

After running tests with coverage, reports are generated in the `reports` directory:

- HTML report: `reports/htmlcov/index.html`
- XML report: `reports/coverage.xml`

## Creating New Tests

When adding new functionality to the trading bot, be sure to add corresponding tests. Follow these guidelines:

1. Name your test files with the prefix `test_`.
2. Use the `unittest` framework for test structure.
3. Mock external dependencies to avoid actual API calls or data source connections.
4. Keep tests independent and self-contained.
5. Include both positive and negative test cases.

## Example Test Structure

```python
import unittest
from unittest.mock import Mock, patch

class TestYourComponent(unittest.TestCase):
    def setUp(self):
        # Setup code that runs before each test
        pass
        
    def tearDown(self):
        # Cleanup code that runs after each test
        pass
        
    def test_some_functionality(self):
        # Test code
        self.assertEqual(expected_result, actual_result)
        
    @patch('module.dependency')
    def test_with_mock(self, mock_dependency):
        # Configure mock
        mock_dependency.return_value = 'mocked_result'
        
        # Test code
        result = your_function_that_uses_dependency()
        
        # Assertions
        self.assertEqual('expected_result', result)
        mock_dependency.assert_called_once()
```

## Continuous Integration

The test suite is integrated with the CI system, which runs the tests automatically on code changes. Be sure your tests pass before submitting pull requests. 