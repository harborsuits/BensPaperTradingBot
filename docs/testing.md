# Testing

The BensBot Trading System employs a comprehensive testing strategy to ensure reliability, correctness, and performance of all components.

## Testing Strategy

The testing framework is organized into these key areas:

1. **Unit Tests** - Verify individual components in isolation
2. **Integration Tests** - Test interactions between components
3. **Backtests** - Validate trading strategies on historical data
4. **End-to-End Tests** - Test complete system workflows
5. **Load Tests** - Verify performance under load

## Test Directory Structure

```
tests/
├── unit/              # Unit tests for individual components
├── integration/       # Integration tests between components
├── cassettes/         # VCR.py recordings for API interactions
├── fixtures/          # Test fixtures and shared test resources
├── data/              # Test data (market data, configs, etc.)
├── stress/            # Load and performance tests
└── e2e/               # End-to-end system tests
```

## Unit Testing

Unit tests verify the correctness of individual components in isolation:

```python
def test_risk_manager_position_sizing():
    """Test that the RiskManager correctly calculates position sizes."""
    portfolio_value = 100000.0
    risk_settings = RiskSettings(max_position_pct=0.05, max_risk_pct=0.01)
    risk_manager = RiskManager(portfolio_value, risk_settings)
    
    # Test basic position sizing
    symbol = "AAPL"
    price = 150.0
    stop_loss_pct = 0.05
    
    max_quantity = risk_manager.calculate_position_size(symbol, price, stop_loss_pct)
    
    # Expected: $1000 risk (1% of $100,000) / ($150 * 5% stop) = 133.33 shares
    expected_quantity = int(portfolio_value * risk_settings.max_risk_pct / (price * stop_loss_pct))
    assert max_quantity == expected_quantity
```

## Integration Testing

Integration tests verify that components work correctly together:

### Typed Settings Integration

The `test_settings_roundtrip.py` integration test verifies environment variables properly override file-based settings:

```python
def test_env_override_roundtrip(tmp_path: Path):
    """Test environment variable overrides for typed settings."""
    # Create a sample YAML config
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("risk:\n  max_drawdown: 0.15\nbroker: tradier\n")

    # Set env var that should override the file
    os.environ["RISK__MAX_DRAWDOWN"] = "0.10"

    # Load the config with env override
    s = Settings.model_validate_yaml(cfg_file.read_bytes())
    assert s.risk.max_drawdown == 0.10  # Env var wins over file
```

### Broker Integration Testing

The `test_tradier_broker_integration.py` test verifies the Tradier API interactions using VCR.py:

```python
@tradier_vcr.use_cassette("test_tradier_quotes.yaml")
def test_quotes(tradier_client, test_symbol):
    """Verify that we can retrieve quotes from Tradier API."""
    quotes = tradier_client.get_quotes(test_symbol)
    
    # Check for valid response structure
    assert isinstance(quotes, dict)
    assert test_symbol in quotes
    
    quote = quotes[test_symbol]
    assert "symbol" in quote
    assert "last" in quote
```

## Running Tests

### Basic Test Execution

Run all tests:

```bash
python -m pytest
```

Run a specific test file:

```bash
python -m pytest tests/unit/test_risk_manager.py
```

Run a specific test function:

```bash
python -m pytest tests/unit/test_risk_manager.py::test_risk_manager_position_sizing
```

### Test Tags and Markers

Tests are organized with markers for selective execution:

```bash
# Run only unit tests
python -m pytest -m "unit"

# Run only integration tests
python -m pytest -m "integration"

# Run API-related tests
python -m pytest -m "api"

# Run only backtest tests
python -m pytest -m "backtest"
```

### VCR.py for API Testing

API tests use VCR.py to record API interactions for reliable playback:

```python
# Configure VCR for the Tradier API tests
tradier_vcr = vcr.VCR(
    cassette_library_dir='tests/cassettes',
    record_mode='once',
    match_on=['uri', 'method'],
    filter_headers=['Authorization'],
)

@tradier_vcr.use_cassette('get_positions.yaml')
def test_get_positions(tradier_client):
    """Test retrieving positions from Tradier."""
    positions = tradier_client.get_positions()
    assert isinstance(positions, list)
```

The first time this test runs, it will make real API calls and record the interactions. Future test runs will replay these recordings without making real API calls.

## Test Environment Setup

### Environment Variables

Create a `.env.test` file with test-specific configuration:

```bash
# Test environment settings
TRADIER_SANDBOX=true
INITIAL_CAPITAL=100000
MAX_RISK_PCT=0.01
ENABLE_NOTIFICATIONS=false
```

Load test environment variables in tests:

```python
def pytest_configure(config):
    """Configure test environment."""
    from dotenv import load_dotenv
    load_dotenv('.env.test')
```

### Fixtures

Common fixtures for testing:

```python
@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    return pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [95, 96, 97, 98, 99],
        'close': [102, 103, 104, 105, 106],
        'volume': [1000, 1100, 1200, 1300, 1400]
    }, index=pd.date_range('2023-01-01', periods=5, freq='D'))

@pytest.fixture
def mock_broker():
    """Create a mock broker for testing."""
    broker = MagicMock()
    broker.get_quote.return_value = {'last': 100.0, 'bid': 99.5, 'ask': 100.5}
    broker.get_positions.return_value = []
    return broker
```

## Continuous Integration

Tests are automatically run in the CI pipeline on GitHub Actions:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run tests
        run: |
          python -m pytest tests/unit
```

## Code Coverage

Code coverage is tracked with pytest-cov:

```bash
python -m pytest --cov=trading_bot --cov-report=html tests/
```

Coverage reports are available in the `htmlcov` directory after test execution.

## Best Practices for Testing

1. **Test isolation**: Each test should run independently
2. **Deterministic tests**: Avoid dependencies on external state
3. **Mock external services**: Use mocks for APIs, databases, etc.
4. **Meaningful assertions**: Test the behavior, not the implementation
5. **Test edge cases**: Include boundary conditions and error scenarios
