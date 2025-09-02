# API Reference

This document provides detailed information about the programmatic interfaces available in the Trading Bot system for extension and integration.

## Table of Contents

1. [Backtesting API](#backtesting-api)
2. [Strategy API](#strategy-api)
3. [Risk Management API](#risk-management-api)
4. [Data API](#data-api)
5. [Optimization API](#optimization-api)
6. [Dashboard API](#dashboard-api)
7. [Market Regime API](#market-regime-api)
8. [Event System](#event-system)

## Backtesting API

### UnifiedBacktester

The primary class for backtesting trading strategies.

#### Constructor

```python
UnifiedBacktester(
    initial_capital=100000.0,
    strategies=None,
    start_date=None,
    end_date=None,
    data_source="mock",
    rebalance_frequency="daily",
    trading_cost_pct=0.1,
    data_directory="data",
    min_trade_value=0.0,
    risk_free_rate=0.0,
    benchmark_symbol="SPY",
    enable_risk_management=False,
    risk_config=None,
    debug_mode=False,
    use_mock=False
)
```

#### Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `run_backtest()` | Run the backtest simulation | None | Dict with results |
| `generate_performance_report(metrics=None)` | Generate a performance report | `metrics`: Optional pre-calculated metrics | Dict with performance report |
| `calculate_advanced_metrics(returns, values)` | Calculate advanced performance metrics | `returns`: Series of daily returns<br>`values`: Series of portfolio values | Dict with advanced metrics |
| `calculate_drawdowns(portfolio_values)` | Calculate drawdown series and statistics | `portfolio_values`: Series of portfolio values | Dict with drawdown analysis |
| `plot_drawdowns(portfolio_values=None, save_path=None)` | Plot drawdown chart | `portfolio_values`: Optional values<br>`save_path`: Path to save plot | None |
| `plot_rolling_metrics(window=20, save_path=None)` | Plot rolling performance metrics | `window`: Rolling window size<br>`save_path`: Path to save plot | None |
| `simulate_risk_scenario(scenario_type='market_crash', duration_days=10, severity=0.8)` | Simulate a risk scenario | `scenario_type`: Type of scenario<br>`duration_days`: Duration in days<br>`severity`: Severity factor (0-1) | Dict with scenario results |
| `run_risk_scenarios()` | Run a suite of risk scenarios | None | Dict with scenario results |

#### Events

| Event | Description | Data |
|-------|-------------|------|
| `on_backtest_start` | Fired when backtest starts | Start date, end date, strategies |
| `on_backtest_end` | Fired when backtest completes | Results summary |
| `on_rebalance` | Fired on rebalance events | Date, allocations before/after |
| `on_trade` | Fired when trades are executed | Trade details |

#### Example Usage

```python
from trading_bot.backtesting.unified_backtester import UnifiedBacktester

# Initialize backtester
backtester = UnifiedBacktester(
    initial_capital=100000.0,
    strategies=["trend_following", "momentum"],
    start_date="2022-01-01",
    end_date="2022-12-31",
    rebalance_frequency="weekly",
    enable_risk_management=True
)

# Register event handlers
backtester.on_trade = lambda trade: print(f"Trade executed: {trade}")

# Run backtest
results = backtester.run_backtest()

# Generate reports
report = backtester.generate_performance_report()

# Create visualizations
backtester.plot_drawdowns(save_path="drawdowns.png")
backtester.plot_rolling_metrics(window=20, save_path="rolling_metrics.png")
```

## Strategy API

### BaseStrategy

Abstract base class for implementing trading strategies.

#### Constructor

```python
BaseStrategy(
    name,
    parameters=None
)
```

#### Methods to Implement

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `calculate_signal(market_data, lookback=None)` | Calculate strategy signal | `market_data`: DataFrame with market data<br>`lookback`: Number of bars to look back | Float signal value (-1 to 1) |
| `get_allocation(market_context)` | Get allocation percentage | `market_context`: Dict with market context | Float allocation percentage (0-100) |

#### Optional Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `update(market_data)` | Update strategy with new data | `market_data`: New market data | None |
| `get_parameters()` | Get strategy parameters | None | Dict of parameters |
| `set_parameters(parameters)` | Update strategy parameters | `parameters`: Dict of parameters | None |
| `validate_parameters(parameters)` | Validate parameter values | `parameters`: Dict of parameters | Boolean |

#### Example Implementation

```python
from trading_bot.strategy.base import BaseStrategy
import pandas as pd
import numpy as np

class MovingAverageCrossover(BaseStrategy):
    def __init__(self, parameters=None):
        default_params = {
            "fast_period": 20,
            "slow_period": 50,
            "signal_threshold": 0.2
        }
        
        # Merge default with provided parameters
        merged_params = {**default_params, **(parameters or {})}
        
        super().__init__(name="ma_crossover", parameters=merged_params)
    
    def calculate_signal(self, market_data, lookback=None):
        # Get parameters
        fast_period = self.parameters["fast_period"]
        slow_period = self.parameters["slow_period"]
        
        # Calculate moving averages
        fast_ma = market_data['close'].rolling(window=fast_period).mean()
        slow_ma = market_data['close'].rolling(window=slow_period).mean()
        
        # Calculate crossover signal
        if len(fast_ma) < slow_period:
            return 0.0
        
        # Normalize difference between -1 and 1
        diff = (fast_ma.iloc[-1] - slow_ma.iloc[-1]) / market_data['close'].iloc[-1]
        signal = np.clip(diff * 5, -1, 1)  # Scale and clip
        
        return signal
    
    def get_allocation(self, market_context):
        # Get signal
        signal = self.last_signal if hasattr(self, 'last_signal') else 0
        
        # Convert signal to allocation percentage
        if abs(signal) < self.parameters["signal_threshold"]:
            return 0.0  # No allocation if signal is weak
        
        # Scale allocation based on signal strength
        allocation = signal * 100.0  # Convert to percentage
        
        # Adjust based on market regime
        if market_context.get('market_regime') == 'volatile':
            allocation *= 0.5  # Reduce allocation in volatile regimes
        
        return allocation
```

## Risk Management API

### RiskManager

Class for managing trading risk and enforcing risk controls.

#### Constructor

```python
RiskManager(
    config=None
)
```

#### Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `update_portfolio_value(value, date)` | Update with new portfolio value | `value`: Portfolio value<br>`date`: Current date | None |
| `calculate_drawdown()` | Calculate current drawdown | None | Float drawdown value |
| `calculate_volatility(window=20)` | Calculate realized volatility | `window`: Lookback window | Float volatility value |
| `check_circuit_breakers(current_date)` | Check if circuit breakers are active | `current_date`: Current date | Dict with circuit breaker status |
| `recommend_leverage()` | Get leverage recommendation | None | Float recommended leverage |
| `calculate_max_position_size(portfolio_value, strategy_name)` | Calculate max position size | `portfolio_value`: Current portfolio value<br>`strategy_name`: Strategy to calculate for | Float maximum position size |
| `calculate_position_sizing_adjustment(portfolio_value)` | Calculate position sizing adjustment | `portfolio_value`: Current portfolio value | Float adjustment factor |

#### Example Usage

```python
from trading_bot.risk import RiskManager
from datetime import datetime

# Initialize risk manager
risk_config = {
    "circuit_breakers": {
        "drawdown": {
            "daily": {"threshold": -0.03, "level": 1}
        }
    }
}
risk_manager = RiskManager(config=risk_config)

# Update with portfolio values
risk_manager.update_portfolio_value(100000.0, datetime.now())

# Check circuit breakers
circuit_breaker_status = risk_manager.check_circuit_breakers(datetime.now())
if circuit_breaker_status["active"]:
    print(f"Circuit breaker active: {circuit_breaker_status}")

# Get position sizing adjustment
adjustment = risk_manager.calculate_position_sizing_adjustment(100000.0)
print(f"Position sizing adjustment: {adjustment}")
```

### RiskMonitor

Class for monitoring risk metrics and detecting anomalies.

#### Constructor

```python
RiskMonitor(
    config=None
)
```

#### Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `create_portfolio(portfolio_id)` | Create a new portfolio to monitor | `portfolio_id`: Unique ID for portfolio | None |
| `update_portfolio(portfolio_id, value, return_value, timestamp)` | Update portfolio with new value | `portfolio_id`: Portfolio ID<br>`value`: New value<br>`return_value`: Return<br>`timestamp`: Timestamp | None |
| `detect_anomalies(portfolio_id)` | Detect anomalies in portfolio | `portfolio_id`: Portfolio ID | List of anomalies |
| `run_stress_test(portfolio_id, allocations, strategy_profiles)` | Run stress test on portfolio | `portfolio_id`: Portfolio ID<br>`allocations`: Current allocations<br>`strategy_profiles`: Strategy characteristics | Dict with stress test results |
| `calculate_var(portfolio_id, confidence=0.95)` | Calculate Value-at-Risk | `portfolio_id`: Portfolio ID<br>`confidence`: Confidence level | Float VaR value |
| `calculate_cvar(portfolio_id, confidence=0.95)` | Calculate Conditional VaR | `portfolio_id`: Portfolio ID<br>`confidence`: Confidence level | Float CVaR value |
| `generate_risk_report(portfolio_id)` | Generate risk report | `portfolio_id`: Portfolio ID | Dict with risk metrics |

## Data API

### RealTimeDataManager

Class for retrieving and managing real-time market data.

#### Constructor

```python
RealTimeDataManager(
    symbols,
    config=None
)
```

#### Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `start()` | Start data streaming | None | None |
| `stop()` | Stop data streaming | None | None |
| `get_latest_data(symbol, timeframe='1min')` | Get latest data for symbol | `symbol`: Asset symbol<br>`timeframe`: Bar timeframe | DataFrame with latest data |
| `get_historical_data(symbol, start_date, end_date, timeframe='1day')` | Get historical data | `symbol`: Asset symbol<br>`start_date`: Start date<br>`end_date`: End date<br>`timeframe`: Bar timeframe | DataFrame with historical data |
| `add_symbol(symbol)` | Add symbol to watchlist | `symbol`: Asset symbol | None |
| `remove_symbol(symbol)` | Remove symbol from watchlist | `symbol`: Asset symbol | None |

#### Events

| Event | Description | Data |
|-------|-------------|------|
| `on_bar_update` | Fired when new bar data arrives | Bar data |
| `on_trade` | Fired when new trade data arrives | Trade data |
| `on_quote` | Fired when new quote data arrives | Quote data |
| `on_error` | Fired when an error occurs | Error information |
| `on_regime_change` | Fired when market regime changes | New regime information |

#### Example Usage

```python
from trading_bot.data.real_time_data_processor import RealTimeDataManager
import os

# Setup configuration
config = {
    'data_source': 'alpaca',
    'timeframes': ['1min', '5min', '1day'],
    'alpaca_config': {
        'api_key': os.environ.get('ALPACA_API_KEY'),
        'api_secret': os.environ.get('ALPACA_API_SECRET')
    }
}

# Initialize data manager
data_manager = RealTimeDataManager(symbols=['SPY', 'QQQ', 'IWM'], config=config)

# Define event handlers
def on_bar_handler(bar_data):
    print(f"New bar: {bar_data['symbol']} - {bar_data['timestamp']} - ${bar_data['price']}")

data_manager.on_bar_update = on_bar_handler

# Start data streaming
data_manager.start()

# Get historical data
historical_data = data_manager.get_historical_data(
    symbol='SPY',
    start_date='2022-01-01',
    end_date='2022-12-31',
    timeframe='1day'
)

# Stop data streaming
data_manager.stop()
```

## Optimization API

### StrategyOptimizer

Class for optimizing strategy parameters.

#### Constructor

```python
StrategyOptimizer(
    strategy_name,
    param_space,
    metric="sharpe_ratio",
    maximize=True,
    backtest_config=None
)
```

#### Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `run_grid_search(parallel=True)` | Run grid search optimization | `parallel`: Enable parallel processing | Dict with optimization results |
| `run_random_search(n_iterations=100, parallel=True)` | Run random search optimization | `n_iterations`: Number of iterations<br>`parallel`: Enable parallel processing | Dict with optimization results |
| `run_bayesian_optimization(n_iterations=50)` | Run Bayesian optimization | `n_iterations`: Number of iterations | Dict with optimization results |
| `get_best_parameters()` | Get best parameters found | None | Dict with best parameters |
| `save_results(filepath)` | Save optimization results | `filepath`: Path to save results | None |
| `load_results(filepath)` | Load optimization results | `filepath`: Path to load results | None |
| `plot_optimization_results(metric=None, save_path=None)` | Plot optimization results | `metric`: Metric to plot<br>`save_path`: Path to save plot | None |

#### Example Usage

```python
from trading_bot.optimization.strategy_optimizer import StrategyOptimizer

# Define parameter space
param_space = {
    "lookback": [10, 20, 30, 40, 50],
    "threshold": [0.5, 1.0, 1.5, 2.0],
    "factor": [0.1, 0.2, 0.3]
}

# Configure backtest settings
backtest_config = {
    "initial_capital": 100000,
    "start_date": "2022-01-01",
    "end_date": "2022-12-31",
    "rebalance_frequency": "weekly"
}

# Create optimizer
optimizer = StrategyOptimizer(
    strategy_name="momentum",
    param_space=param_space,
    metric="sharpe_ratio",
    maximize=True,
    backtest_config=backtest_config
)

# Run grid search
results = optimizer.run_grid_search(parallel=True)

# Get best parameters
best_params = optimizer.get_best_parameters()
print(f"Best parameters: {best_params}")

# Plot results
optimizer.plot_optimization_results(save_path="optimization_results.png")

# Save results
optimizer.save_results("momentum_optimization.json")
```

### MLStrategyOptimizer

Extended optimizer using machine learning to predict parameter performance.

#### Constructor

```python
MLStrategyOptimizer(
    strategy_name,
    param_space,
    metric="sharpe_ratio",
    maximize=True,
    backtest_config=None,
    ml_model="random_forest"
)
```

#### Methods (in addition to StrategyOptimizer methods)

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `train_ml_model(n_samples=200)` | Train ML model on parameter samples | `n_samples`: Number of samples to generate | Model training metrics |
| `predict_performance(parameters)` | Predict performance of parameters | `parameters`: Parameter set to evaluate | Predicted performance |
| `run_ml_guided_optimization(n_iterations=50)` | Run ML-guided optimization | `n_iterations`: Number of iterations | Dict with optimization results |
| `save_model(filepath)` | Save trained ML model | `filepath`: Path to save model | None |
| `load_model(filepath)` | Load trained ML model | `filepath`: Path to load model | None |

## Dashboard API

### Dashboard Components

The dashboard is built with Streamlit and can be extended with custom components.

#### Adding Custom Visualizations

Create a new function in `trading_bot/visualization/custom_charts.py`:

```python
def plot_custom_chart(data, **kwargs):
    """Create a custom visualization."""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    # Add your custom chart logic here
    
    return fig
```

Register it in the dashboard by adding it to the `create_dashboard` function:

```python
def create_dashboard():
    # ... existing dashboard code ...
    
    # Add your custom component
    with st.expander("Custom Analysis", expanded=False):
        if 'portfolio_df' in st.session_state:
            data = st.session_state.portfolio_df
            st.plotly_chart(plot_custom_chart(data), use_container_width=True)
    
    # ... rest of the dashboard code ...
```

#### Dashboard Event Hooks

The dashboard provides event hooks to respond to user actions:

```python
# In your custom dashboard extension
def on_symbol_change(symbol):
    """Handle symbol change event."""
    # Perform custom actions when user changes the selected symbol
    st.session_state.custom_data = load_data_for_symbol(symbol)

def on_timeframe_change(timeframe):
    """Handle timeframe change event."""
    # Perform custom actions when user changes the timeframe
    st.session_state.chart_data = resample_data(st.session_state.raw_data, timeframe)

# Register the event hooks
st.session_state.callbacks = {
    'on_symbol_change': on_symbol_change,
    'on_timeframe_change': on_timeframe_change
}
```

## Market Regime API

### AdvancedMarketRegimeDetector

Class for detecting different market regimes using machine learning and statistical methods.

#### Constructor

```python
AdvancedMarketRegimeDetector(
    config=None,
    model_path=None
)
```

#### Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `fit(market_data)` | Train the regime detection model | `market_data`: DataFrame with market data | None |
| `predict(market_data)` | Predict the current market regime | `market_data`: DataFrame with market data | Dict with regime prediction |
| `detect_regime_change(market_data)` | Detect if regime has changed | `market_data`: DataFrame with market data | Boolean indicating change |
| `compute_features(market_data)` | Compute features for regime detection | `market_data`: DataFrame with market data | DataFrame with features |
| `train_unsupervised_model(features)` | Train unsupervised clustering model | `features`: Feature DataFrame | None |
| `train_supervised_model(features, labels)` | Train supervised classification model | `features`: Feature DataFrame<br>`labels`: Known regime labels | None |
| `save_model(filepath)` | Save trained model | `filepath`: Path to save model | None |
| `load_model(filepath)` | Load trained model | `filepath`: Path to load model | None |
| `plot_regime_detection(market_data, predictions=None)` | Plot regime detection visualization | `market_data`: Market data<br>`predictions`: Optional regime predictions | Matplotlib figure |

#### Example Usage

```python
from trading_bot.optimization.advanced_market_regime_detector import AdvancedMarketRegimeDetector
import pandas as pd

# Load market data
market_data = pd.read_csv("market_data.csv", parse_dates=["date"], index_col="date")

# Initialize detector
detector = AdvancedMarketRegimeDetector()

# Train model
detector.fit(market_data)

# Predict current regime
current_regime = detector.predict(market_data.tail(30))
print(f"Current market regime: {current_regime['regime']} (confidence: {current_regime['confidence']:.2f})")

# Save model
detector.save_model("market_regime_model.pkl")

# Visualize regimes
detector.plot_regime_detection(market_data)
```

### StrategyRegimeRotator

Class for rotating strategies based on detected market regimes.

#### Constructor

```python
StrategyRegimeRotator(
    strategies,
    initial_weights=None,
    regime_config=None,
    lookback_window=20,
    rebalance_frequency='weekly',
    max_allocation_change=0.20
)
```

#### Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `update_market_regime(market_data)` | Update current market regime | `market_data`: Recent market data | Dict with regime information |
| `optimize_weights(market_context)` | Optimize strategy weights for regime | `market_context`: Current market context | Dict with optimized weights |
| `rotate_strategies(market_context, force_rotation=False)` | Rotate strategies based on regime | `market_context`: Market context<br>`force_rotation`: Force rotation regardless of schedule | Dict with rotation result |
| `update_portfolio_value(portfolio_value)` | Update current portfolio value | `portfolio_value`: New portfolio value | None |
| `analyze_strategy_performance_by_regime(backtest_results)` | Analyze how strategies perform in different regimes | `backtest_results`: Backtest results | Dict with analysis |
| `backtest_regime_rotation(market_data, initial_capital=100000.0)` | Backtest the regime rotation strategy | `market_data`: Historical market data<br>`initial_capital`: Starting capital | Dict with backtest results |
| `plot_regime_rotation_results(results=None)` | Plot regime rotation results | `results`: Rotation results | Matplotlib figure |

## Event System

The trading system uses an event-driven architecture. Here's how to work with it:

### Registering Event Handlers

```python
from trading_bot.core.events import EventEmitter

# Create an event emitter
emitter = EventEmitter()

# Define event handlers
def on_trade_handler(trade_data):
    print(f"Trade executed: {trade_data}")

def on_error_handler(error_data):
    print(f"Error occurred: {error_data}")

# Register event handlers
emitter.on('trade', on_trade_handler)
emitter.on('error', on_error_handler)

# Emit events
emitter.emit('trade', {'symbol': 'SPY', 'quantity': 100, 'price': 450.25})
```

### Creating Custom Events

```python
from trading_bot.core.events import create_event

# Create a custom event
MyCustomEvent = create_event('MyCustomEvent', ['timestamp', 'data'])

# Create an instance of the event
event = MyCustomEvent(timestamp='2023-01-01 12:00:00', data={'value': 42})

# Register a handler for this event
def handle_custom_event(event_data):
    print(f"Custom event at {event_data.timestamp}: {event_data.data}")

emitter.on('MyCustomEvent', handle_custom_event)

# Emit the event
emitter.emit('MyCustomEvent', event)
```

### System-wide Events

These events are emitted by various components throughout the system:

| Event | Source | Description | Data |
|-------|--------|-------------|------|
| `bar_update` | Data Manager | New bar data received | Bar data |
| `trade_executed` | Execution Module | Trade executed | Trade details |
| `portfolio_update` | Portfolio Manager | Portfolio value updated | Portfolio data |
| `risk_alert` | Risk Monitor | Risk threshold exceeded | Alert details |
| `regime_change` | Regime Detector | Market regime changed | New regime |
| `strategy_update` | Strategy Manager | Strategy updates weights | New weights |
| `optimization_complete` | Optimizer | Optimization finished | Optimization results |
| `backtest_progress` | Backtester | Backtest progress update | Progress percentage | 