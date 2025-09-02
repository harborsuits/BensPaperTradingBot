# Backtesting Engine

The BensBot Trading System includes a comprehensive backtesting engine for evaluating trading strategies on historical data with realistic simulations of market conditions.

## Overview

The backtesting engine enables:

1. **Strategy Evaluation**: Test trading strategies on historical data
2. **Parameter Optimization**: Identify optimal strategy parameters
3. **Risk Analysis**: Evaluate risk metrics and drawdowns
4. **Performance Comparison**: Compare multiple strategies
5. **Strategy Rotation**: Test dynamic strategy allocation

## Architecture

The backtesting system consists of these core components:

```
UnifiedBacktester
├── DataLoader
├── StrategyManager
├── PositionManager
├── RiskManager
├── PerformanceAnalyzer
└── ReportGenerator
```

## Configuration

The backtester is configured through the typed settings system:

```python
class BacktestSettings(BaseModel):
    """Backtesting configuration."""
    default_symbols: List[str] = Field(default_factory=list)
    default_start_date: str = (datetime.now().replace(year=datetime.now().year-1)).strftime("%Y-%m-%d")
    default_end_date: str = datetime.now().strftime("%Y-%m-%d")
    initial_capital: float = 100000.0
    commission_per_trade: float = 0.0
    slippage_pct: float = 0.0005
    data_source: str = "local"  # local, alpha_vantage, etc.
    
    @validator('default_start_date', 'default_end_date')
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")
        return v
```

## Unified Backtester

The `UnifiedBacktester` is the primary interface for backtesting:

```python
class UnifiedBacktester:
    """Unified backtesting engine for strategy evaluation and rotation."""
    
    def __init__(
        self, 
        initial_capital: float = 100000.0,
        strategies: List[str] = None,
        start_date: str = None,
        end_date: str = None,
        rebalance_frequency: str = "weekly",
        benchmark_symbol: str = "SPY",
        data_dir: str = "data",
        results_path: str = "data/backtest_results",
        use_mock: bool = False,
        risk_free_rate: float = 0.02,
        trading_cost_pct: float = 0.1,
        config_path: Optional[str] = None,
        settings: Optional[Union[BacktestSettings, TradingBotSettings]] = None,
        **kwargs
    ):
        """Initialize the unified backtester."""
        # Loads settings from typed configuration
        if settings:
            if isinstance(settings, TradingBotSettings):
                self.backtest_settings = settings.backtest
                self.risk_settings = settings.risk
            else:
                self.backtest_settings = settings
                self.risk_settings = None
        else:
            # Load config from file if provided
            config = load_config(config_path) if config_path else None
            self.backtest_settings = config.backtest if config else BacktestSettings()
            self.risk_settings = config.risk if config else RiskSettings()
            
        # Initialize with settings
        self.initial_capital = initial_capital or self.backtest_settings.initial_capital
        self.start_date = start_date or self.backtest_settings.default_start_date
        self.end_date = end_date or self.backtest_settings.default_end_date
        # ... additional initialization
```

## Running a Backtest

To run a backtest:

```python
from trading_bot.backtesting.unified_backtester import UnifiedBacktester

# Create backtester with typed settings
backtester = UnifiedBacktester(
    strategies=["momentum", "mean_reversion"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    rebalance_frequency="monthly"
)

# Run the backtest
results = backtester.run()

# Generate performance report
report = backtester.generate_performance_report()

# Save results and visualizations
backtester.save_results("momentum_vs_mean_reversion")
```

## Strategy Rotation Testing

The backtester can evaluate dynamic strategy rotation:

```python
# Configure strategy rotation backtest
rotation_backtest = UnifiedBacktester(
    strategies=["momentum", "trend_following", "mean_reversion", "rsi", "macd"],
    start_date="2022-01-01",
    end_date="2023-12-31",
    rebalance_frequency="weekly",
    rotation_metric="sharpe",  # Select best strategies by Sharpe ratio
    max_active_strategies=2    # Use top 2 strategies at any time
)

# Run rotation backtest
rotation_results = rotation_backtest.run(mode="rotation")
```

## Performance Metrics

The backtester calculates comprehensive performance metrics:

- **Returns**: Total return, annualized return, daily/monthly returns
- **Risk Metrics**: Volatility, Sharpe ratio, Sortino ratio, maximum drawdown
- **Trade Statistics**: Win rate, profit factor, average win/loss
- **Benchmark Comparison**: Alpha, beta, correlation, tracking error

## Market Data Integration

The backtesting engine integrates with multiple market data sources:

- **Local CSV Files**: Fast testing with pre-downloaded data
- **Alpha Vantage API**: Historical price and fundamental data
- **Finnhub**: Alternative data source with real-time capabilities
- **Tradier API**: Market data from your broker

## Risk Management Simulation

The backtester applies the same risk management rules used in live trading:

- **Position Sizing**: Apply max position size rules
- **Portfolio Risk**: Enforce maximum portfolio risk
- **Correlation Limits**: Respect maximum correlated positions
- **Stop-Loss Simulation**: Test stop-loss and take-profit strategies

## Report Generation

The backtester generates comprehensive reports:

- **Performance Summary**: Key metrics and comparison to benchmark
- **Equity Curve**: Visual representation of portfolio value over time
- **Drawdown Analysis**: Magnitude and duration of drawdowns
- **Trade List**: Detailed record of all simulated trades
- **Monthly Returns**: Calendar of monthly performance

## API Integration

The backtesting engine is accessible through the API:

```
POST /api/v1/backtest
{
  "strategies": ["momentum", "trend_following"],
  "symbols": ["AAPL", "MSFT", "GOOG"],
  "start_date": "2023-01-01",
  "end_date": "2023-12-31",
  "initial_capital": 100000
}
```

## Optimization

The backtester supports parameter optimization:

```python
# Define parameter grid for optimization
param_grid = {
    "lookback_period": [10, 20, 30, 40, 50],
    "entry_threshold": [0.01, 0.02, 0.03, 0.04, 0.05],
    "exit_threshold": [0.01, 0.015, 0.02, 0.025, 0.03]
}

# Run grid search optimization
optimization_results = backtester.optimize(
    strategy="momentum",
    param_grid=param_grid,
    metric="sharpe_ratio",  # Optimize for Sharpe ratio
    n_jobs=-1  # Use all available CPU cores
)
```

## Walk-Forward Testing

To validate strategy robustness, the backtester supports walk-forward testing:

```python
# Configure walk-forward test
wf_test = UnifiedBacktester(
    strategies=["momentum"],
    start_date="2020-01-01",
    end_date="2023-12-31"
)

# Run walk-forward analysis with 6-month training, 1-month testing
wf_results = wf_test.walk_forward_analysis(
    train_period=180,  # 180 days training
    test_period=30,    # 30 days testing
    metric="sharpe_ratio"
)
```

## Monte Carlo Simulation

The backtester can run Monte Carlo simulations to estimate strategy robustness:

```python
# Run Monte Carlo simulation
mc_results = backtester.monte_carlo_simulation(
    num_simulations=1000,
    confidence_interval=0.95
)
```
