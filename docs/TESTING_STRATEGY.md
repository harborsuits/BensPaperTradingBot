# BensBot Testing Strategy

This document outlines the comprehensive testing strategy for BensBot, ensuring high-quality code, preventing regressions, and maintaining a robust trading platform.

## Test Organization

The test suite is organized as follows:

```
tests/
├── unit/                  # Unit tests for isolated components
├── integration/           # Integration tests between components
├── brokers/               # Broker-specific tests
│   └── intelligence/      # Broker intelligence tests
├── core/                  # Core trading engine tests
├── persistence/           # Data persistence tests
├── strategies/            # Trading strategy tests
├── cassettes/             # VCR cassettes for API mocking
└── data/                  # Test data files
```

## Test Types

### 1. Unit Tests

Unit tests focus on testing individual components in isolation, with mocked dependencies.

**Key Components to Test:**

- **Strategy Logic**
  - Signal generation from known price data
  - Entry/exit condition evaluation
  - Risk calculation

- **Broker Adapters**
  - Account data retrieval
  - Order placement, cancellation, modification
  - Position and balance information
  - Error handling

- **Multi-Broker Manager**
  - Idempotent order placement
  - Broker failover
  - Asset routing

- **Persistence Layer**
  - Repository CRUD operations
  - Data model conversions
  - Cache invalidation

- **Configuration Management**
  - Config loading and validation
  - Default values
  - Validation errors

### 2. Integration Tests

Integration tests verify the correct interaction between multiple components.

**Key Integration Scenarios:**

- **Database Integration**
  - Repository transaction sequences
  - Crash recovery logic
  - Cache/DB synchronization

- **API Integration**
  - Dashboard API endpoints
  - Authentication and authorization
  - Data retrieval pipelines

- **Broker Integration**
  - End-to-end order flows
  - Market data consumption
  - Event handling across components

- **Event System Integration**
  - Event propagation
  - Event handling by subscribers
  - Event-driven workflows

### 3. Simulation Tests

- **Backtesting Accuracy**
  - Known market conditions with expected outcomes
  - P&L calculation verification
  - Trade execution simulation

- **Stress Testing**
  - High-frequency trading scenarios
  - Large position/order volume handling
  - Recovery from simulated crashes

## Test Data Management

### Static Test Data

- **Market Data Samples**
  - Small, representative price datasets
  - Various market condition samples (trending, volatile, sideways)

- **Configuration Templates**
  - Standard test configurations
  - Edge case configurations

- **Mock API Responses**
  - Recorded broker API responses
  - Error case responses

### Dynamic Test Data

- **Generators**
  - Price series generators
  - Random order generators
  - Event sequence generators

## Testing Infrastructure

### Mocking Strategies

- **External API Mocking**
  - Use VCR.py for recording/replaying API interactions
  - Mock implementation of broker APIs

- **Database Mocking**
  - In-memory MongoDB for repository tests
  - Test-specific Redis instance

- **Time Control**
  - Deterministic time progression for backtesting
  - Mock scheduling for time-based events

### Fixtures

- **Reusable Test Fixtures**
  - Standard broker setup
  - Database connections
  - Test event bus

## Test Execution

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run only unit tests
pytest tests/unit/

# Run with coverage
pytest --cov=trading_bot --cov-report=html

# Run specific test categories
pytest tests/integration/
```

### Continuous Integration

Tests are run automatically via GitHub Actions:
- On every pull request
- On pushes to main branches
- Scheduled weekly for regression testing

## Coverage Goals

- **Target: 80%+ code coverage**
- **Critical components: 90%+ coverage**
  - Broker adapters
  - Order execution logic
  - Risk management
  - Persistence layer

## Best Practices

1. **Test-Driven Development**
   - Write tests before implementing features
   - Use tests to validate design decisions

2. **Test Isolation**
   - Each test should be independent
   - Clean up after tests to avoid state pollution

3. **Realistic Scenarios**
   - Test edge cases and error conditions
   - Simulate real-world trading conditions

4. **Regular Maintenance**
   - Update tests when requirements change
   - Remove obsolete tests
   - Add tests for bug fixes

5. **Documentation**
   - Document test purpose and approach
   - Include example usage in docstrings

## Implementation Priorities

1. **Critical Path Testing**
   - Order execution flow
   - Position management
   - Risk controls

2. **Recovery Testing**
   - Crash recovery scenarios
   - Network interruption handling

3. **Performance Testing**
   - Response time under load
   - Resource utilization

4. **Security Testing**
   - Authentication and authorization
   - Input validation

## Appendix: Test Examples

### Example: Testing Strategy Signal Generation

```python
def test_ma_crossover_strategy_signals():
    # Prepare test data - price series with known crossover points
    prices = pd.DataFrame({
        'close': [10, 11, 12, 11, 10, 9, 8, 7, 8, 9, 10, 11]
    })
    
    # Initialize strategy with known parameters
    strategy = MovingAverageCrossoverStrategy(
        fast_period=2,
        slow_period=4
    )
    
    # Generate signals
    signals = strategy.generate_signals(prices)
    
    # Verify signals occur at expected crossover points
    assert signals.iloc[5]['signal'] == SignalType.SELL
    assert signals.iloc[9]['signal'] == SignalType.BUY
```

### Example: Testing Idempotent Order Placement

```python
def test_idempotent_order_placement(mocker):
    # Create mocks
    mock_broker = mocker.Mock()
    mock_redis = mocker.Mock()
    
    # Set up broker manager with mocked components
    broker_manager = MultiBrokerManager(
        redis_client=mock_redis
    )
    broker_manager.add_broker('test', mock_broker)
    
    # Configure Redis mock behavior
    mock_redis.get.return_value = None  # First call - no existing order
    
    # First order placement
    order_params = {'symbol': 'AAPL', 'quantity': 10, 'price': 150.0}
    broker_manager.place_order_idempotently('test_key', order_params)
    
    # Verify order was placed once
    mock_broker.place_order.assert_called_once_with(order_params)
    
    # Reset and prepare for second call with same key
    mock_broker.place_order.reset_mock()
    mock_redis.get.return_value = 'ORDER123'  # Simulate existing order
    
    # Second order placement with same key
    broker_manager.place_order_idempotently('test_key', order_params)
    
    # Verify second call did not place order
    mock_broker.place_order.assert_not_called()
```
