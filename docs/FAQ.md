# Frequently Asked Questions & Troubleshooting

This document provides answers to common questions and solutions to frequent issues encountered when using the Trading Bot system.

## Table of Contents
- [General Questions](#general-questions)
- [Backtesting](#backtesting)
- [Risk Management](#risk-management)
- [Data and Connectivity](#data-and-connectivity)
- [Dashboard](#dashboard)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

## General Questions

### What hardware requirements does the system have?
The trading system can run on any modern computer with at least 8GB RAM and a multi-core processor. For large backtests with multiple strategies and extensive data, 16GB RAM or more is recommended.

### Does the system support paper trading?
Yes, the system can be configured for paper trading through supported brokers like Alpaca or Interactive Brokers. Configure the execution module with paper trading credentials in your `.env` file.

### Can I run the system in the cloud?
Yes, the system can be deployed to cloud environments like AWS, GCP, or Azure. Docker configurations are available in the `deployment/` directory to facilitate containerized deployment.

### How can I contribute to the project?
Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Backtesting

### Why are my backtest results different from live trading?
Several factors can cause discrepancies:
1. **Data differences** - Backtesting may use different data sources than live trading
2. **Execution assumptions** - Backtests assume idealized execution that may differ from reality
3. **Look-ahead bias** - Check for potential future data leaking into strategy calculations
4. **Trading costs** - Ensure you're accurately modeling commissions, slippage, and market impact

### How do I add a custom strategy to the backtester?
Extend the `BaseStrategy` class and implement the required methods:

```python
from trading_bot.strategy.base import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    def __init__(self, parameters=None):
        super().__init__(name="my_custom_strategy", parameters=parameters)
        
    def calculate_signal(self, market_data, lookback=None):
        # Your signal calculation logic here
        return signal_value
        
    def get_allocation(self, market_context):
        # Your allocation logic here
        return allocation_percentage
```

Then register your strategy in the `strategy_factory.py` file.

### How can I optimize strategy parameters?
Use the `StrategyOptimizer` class:

```python
from trading_bot.optimization.strategy_optimizer import StrategyOptimizer

# Define parameter space
param_space = {
    "lookback": [10, 20, 30, 40, 50],
    "threshold": [0.5, 1.0, 1.5, 2.0],
    "factor": [0.1, 0.2, 0.3]
}

# Create optimizer
optimizer = StrategyOptimizer(
    strategy_name="momentum",
    param_space=param_space,
    metric="sharpe_ratio",
    backtest_config={
        "initial_capital": 100000,
        "start_date": "2022-01-01",
        "end_date": "2022-12-31"
    }
)

# Run grid search
results = optimizer.run_grid_search()
best_params = optimizer.get_best_parameters()
```

### Can I save and load backtest results?
Yes, use the built-in save/load methods:

```python
# Save results
backtester.save_results("my_backtest_results.json")

# Load results
from trading_bot.backtesting.unified_backtester import UnifiedBacktester
backtester = UnifiedBacktester()
backtester.load_results("my_backtest_results.json")
```

## Risk Management

### How do circuit breakers work in the system?
Circuit breakers automatically restrict trading when risk thresholds are breached:

1. When triggered (by drawdown, volatility, or correlation metrics), they limit the magnitude of allocation changes
2. Different levels (1-3) impose increasingly strict limits
3. Circuit breakers remain active for a configurable duration before auto-deactivating

See [Risk Controls Documentation](risk_management/risk_controls.md) for detailed configuration options.

### How can I customize risk limits for my strategies?
Modify the risk configuration when initializing the backtester or risk manager:

```python
risk_config = {
    "circuit_breakers": {
        "drawdown": {
            "daily": {"threshold": -0.05, "level": 1},  # Custom 5% daily drawdown limit
            "weekly": {"threshold": -0.10, "level": 2}  # Custom 10% weekly drawdown limit
        },
        "volatility": {
            "threshold": 0.30,  # Custom 30% volatility threshold
            "level": 2
        }
    },
    "position_sizing": {
        "max_position": 0.40,  # Custom 40% max position
    }
}

backtester = UnifiedBacktester(
    # ... other parameters
    enable_risk_management=True,
    risk_config=risk_config
)
```

### What is the difference between circuit breakers and emergency risk controls?
- **Circuit breakers** are preventative controls that limit allocation changes when risk metrics exceed thresholds.
- **Emergency risk controls** are reactive measures activated when severe anomalies are detected, forcibly reducing allocations to high-risk strategies.

## Data and Connectivity

### What data sources does the system support?
Built-in support includes:
- Alpaca
- Interactive Brokers
- Yahoo Finance
- Alpha Vantage
- CSV files
- Custom data sources (via the DataAdapter interface)

### How do I configure a new data source?
Add your API credentials to the `.env` file or environment variables:

```
# For Alpaca
ALPACA_API_KEY=your_api_key
ALPACA_API_SECRET=your_api_secret

# For Interactive Brokers
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=1
```

Then configure the data manager in your code:

```python
from trading_bot.data.real_time_data_processor import RealTimeDataManager

config = {
    'data_source': 'alpaca',
    'timeframes': ['1min', '5min', '1day'],
    'alpaca_config': {
        'api_key': os.environ.get('ALPACA_API_KEY'),
        'api_secret': os.environ.get('ALPACA_API_SECRET')
    }
}

data_manager = RealTimeDataManager(symbols=['SPY', 'QQQ', 'IWM'], config=config)
```

### What should I do if I get "No data available" errors?
1. Check API credentials and permissions
2. Verify the requested symbols are available in the data source
3. Confirm the date range is valid (some sources have limited historical data)
4. Check network connectivity to the data provider
5. Look for rate-limiting issues (too many requests)

## Dashboard

### How do I customize the dashboard layout?
Modify the dashboard configuration file at `trading_bot/defaults/dashboard_config.json`:

```json
{
  "theme": "dark",
  "default_symbols": ["SPY", "QQQ", "TLT", "GLD"],
  "update_frequency": 5,
  "performance_metrics": [
    "total_return", "sharpe_ratio", "sortino_ratio", "max_drawdown"
  ],
  "chart_options": {
    "show_volume": true,
    "default_indicators": ["sma", "ema", "rsi"]
  }
}
```

### Can I add custom visualizations to the dashboard?
Yes, add a new visualization function to `trading_bot/visualization/custom_charts.py`:

```python
def plot_my_custom_chart(data, **kwargs):
    """Create a custom visualization."""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    # Add your custom chart logic here
    
    return fig
```

Then register it in the dashboard by adding it to the `create_dashboard` function in `trading_bot/visualization/live_trading_dashboard.py`.

### The dashboard won't start. What should I check?
1. Verify Streamlit is installed (`pip install streamlit`)
2. Check port availability (default 8501)
3. Ensure you have the correct path to the dashboard file
4. Check for any error messages in the console
5. Verify all required dependencies are installed

## Performance

### How can I speed up backtests?
1. Use `parallel=True` in the backtester configuration to enable multi-processing
2. Reduce the number of assets or date range
3. Increase rebalance interval (e.g., weekly instead of daily)
4. Use pre-processed data with the `use_cached_data=True` option
5. Run on a machine with more CPU cores and RAM

### What's the recommended way to handle large datasets?
1. Use a database backend (set `data_storage="database"` in configuration)
2. Enable data chunking (`chunk_size=1000`)
3. Configure caching (`enable_caching=True`)
4. For extremely large datasets, consider a distributed setup with Dask

### How can I profile the performance of my strategies?
Use the built-in profiling tools:

```python
from trading_bot.utils.profiling import profile_strategy

# Profile a specific strategy
profile_results = profile_strategy("momentum", sample_data)

# Print profiling results
print(profile_results.summary())
```

## Troubleshooting

### "ImportError: No module named trading_bot"
Add the project root to your Python path:

```python
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```

### "FileNotFoundError" when loading data
1. Check the path is correct and accessible
2. Verify data files exist and have correct permissions
3. Use absolute paths instead of relative paths
4. Ensure the data directory structure matches configuration

### "KeyError" during strategy calculation
1. Verify the required data columns exist in your dataset
2. Check strategy parameters match expected format
3. Ensure data preprocessing is correctly applied
4. Add error handling for missing data in your strategy logic

### Dashboard shows "NaN" or empty values
1. Check data connectivity in the sidebar status
2. Verify symbols are correctly set up
3. Reset the connection by clicking Disconnect then Connect
4. Check the logs for any data processing errors
5. Ensure time ranges are valid for the selected data source

### "Memory Error" during large backtests
1. Reduce the data size or date range
2. Enable data chunking in the backtester
3. Close other applications to free up memory
4. Use a machine with more RAM
5. Enable the low_memory mode in configuration

### Risk controls not activating as expected
1. Verify risk management is enabled (`enable_risk_management=True`)
2. Check threshold configuration matches your expectations
3. Ensure the risk manager is receiving updates on portfolio values
4. Enable debug mode to see detailed logs of risk calculations
5. Manually validate risk metrics calculations

### "ValueError: non-finite values" in performance metrics
1. Check for division by zero in calculations
2. Look for NaN or infinity values in returns data
3. Add validation in strategy logic to handle edge cases
4. Check for extremely small position sizes that might cause precision issues

### How to restart the system after an unexpected shutdown
1. Run the recovery script: `python trading_bot/recovery.py`
2. Check log files for the last state before shutdown
3. Verify data integrity with `trading_bot/utils/data_validation.py`
4. Restart services in the correct order: data > execution > dashboard 